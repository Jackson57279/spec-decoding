use crate::{
    drafters::Drafter,
    model::{GenerationConfig, ModelResult, TargetModel, TokenId, TokenSequence, greedy_token},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub draft_rounds: usize,
    pub drafted_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub rejected_draft_tokens: usize,
    pub target_forwards: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeOutput {
    pub tokens: TokenSequence,
    pub stats: SpeculativeStats,
}

pub fn speculative_greedy_decode<M: TargetModel, D: Drafter>(
    model: &mut M,
    drafter: &mut D,
    prompt: &[TokenId],
    config: GenerationConfig,
) -> ModelResult<SpeculativeOutput> {
    config.validate()?;

    let mut tokens = TokenSequence::new(prompt.to_vec());
    let mut stats = SpeculativeStats {
        prompt_tokens: prompt.len(),
        generated_tokens: 0,
        draft_rounds: 0,
        drafted_tokens: 0,
        accepted_draft_tokens: 0,
        rejected_draft_tokens: 0,
        target_forwards: 0,
    };

    while stats.generated_tokens < config.max_new_tokens {
        stats.draft_rounds += 1;
        let remaining = config.max_new_tokens - stats.generated_tokens;
        let draft_limit = remaining.min(config.speculative_tokens);
        let draft = drafter.draft(tokens.as_slice(), draft_limit)?;

        if draft.is_empty() {
            let next = target_greedy_token(model, tokens.as_slice(), &mut stats)?;
            tokens.push(next);
            stats.generated_tokens += 1;
            continue;
        }

        let mut accepted_all = true;
        let draft_tokens = &draft.as_slice()[..draft.len().min(remaining)];
        stats.drafted_tokens += draft_tokens.len();

        for &candidate in draft_tokens {
            let verified = target_greedy_token(model, tokens.as_slice(), &mut stats)?;

            if candidate == verified {
                tokens.push(candidate);
                stats.accepted_draft_tokens += 1;
            } else {
                tokens.push(verified);
                stats.rejected_draft_tokens += 1;
                accepted_all = false;
            }

            stats.generated_tokens += 1;

            if !accepted_all || stats.generated_tokens == config.max_new_tokens {
                break;
            }
        }

        if accepted_all && stats.generated_tokens < config.max_new_tokens {
            let next = target_greedy_token(model, tokens.as_slice(), &mut stats)?;
            tokens.push(next);
            stats.generated_tokens += 1;
        }
    }

    Ok(SpeculativeOutput { tokens, stats })
}

fn target_greedy_token<M: TargetModel>(
    model: &mut M,
    prefix: &[TokenId],
    stats: &mut SpeculativeStats,
) -> ModelResult<TokenId> {
    let logits = model.logits_for_prefix(prefix)?;
    stats.target_forwards += 1;
    greedy_token(&logits)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use crate::{
        drafters::{DraftSequence, Drafter},
        model::{GenerationConfig, ModelError, TargetModel, TokenId},
    };

    use super::speculative_greedy_decode;

    struct ScriptedModel {
        tokens: VecDeque<TokenId>,
        prefixes: Vec<Vec<TokenId>>,
    }

    impl ScriptedModel {
        fn new(tokens: Vec<TokenId>) -> Self {
            Self {
                tokens: tokens.into(),
                prefixes: Vec::new(),
            }
        }
    }

    impl TargetModel for ScriptedModel {
        fn vocab_size(&self) -> usize {
            16
        }

        fn logits_for_prefix(&mut self, prefix: &[TokenId]) -> Result<Vec<f32>, ModelError> {
            self.prefixes.push(prefix.to_vec());
            let token = self.tokens.pop_front().ok_or(ModelError::EmptyLogits)?;
            let mut logits = vec![0.0; self.vocab_size()];
            logits[token as usize] = 1.0;
            Ok(logits)
        }
    }

    struct ScriptedDrafter {
        drafts: VecDeque<Vec<TokenId>>,
    }

    impl ScriptedDrafter {
        fn new(drafts: Vec<Vec<TokenId>>) -> Self {
            Self {
                drafts: drafts.into(),
            }
        }
    }

    impl Drafter for ScriptedDrafter {
        fn draft(&mut self, _prefix: &[TokenId], max_tokens: usize) -> Result<DraftSequence, ModelError> {
            let tokens = self.drafts.pop_front().unwrap_or_default();
            Ok(DraftSequence::new(
                tokens.into_iter().take(max_tokens).collect(),
            ))
        }
    }

    #[test]
    fn accepts_drafts_and_adds_bonus_target_token() {
        let mut model = ScriptedModel::new(vec![1, 2, 3, 4]);
        let mut drafter = ScriptedDrafter::new(vec![vec![1, 2], vec![4]]);
        let config = GenerationConfig::greedy(4, 2).expect("valid config");

        let output =
            speculative_greedy_decode(&mut model, &mut drafter, &[10], config).expect("decode");

        assert_eq!(output.tokens.as_slice(), &[10, 1, 2, 3, 4]);
        assert_eq!(output.stats.prompt_tokens, 1);
        assert_eq!(output.stats.generated_tokens, 4);
        assert_eq!(output.stats.draft_rounds, 2);
        assert_eq!(output.stats.drafted_tokens, 3);
        assert_eq!(output.stats.accepted_draft_tokens, 3);
        assert_eq!(output.stats.rejected_draft_tokens, 0);
        assert_eq!(output.stats.target_forwards, 4);
    }

    #[test]
    fn rejects_bad_draft_and_uses_target_token() {
        let mut model = ScriptedModel::new(vec![1, 2, 3]);
        let mut drafter = ScriptedDrafter::new(vec![vec![1, 9], vec![3]]);
        let config = GenerationConfig::greedy(3, 2).expect("valid config");

        let output =
            speculative_greedy_decode(&mut model, &mut drafter, &[5], config).expect("decode");

        assert_eq!(output.tokens.as_slice(), &[5, 1, 2, 3]);
        assert_eq!(output.stats.draft_rounds, 2);
        assert_eq!(output.stats.drafted_tokens, 3);
        assert_eq!(output.stats.accepted_draft_tokens, 2);
        assert_eq!(output.stats.rejected_draft_tokens, 1);
        assert_eq!(output.stats.target_forwards, 3);
    }

    #[test]
    fn falls_back_to_target_when_drafter_returns_empty() {
        let mut model = ScriptedModel::new(vec![7]);
        let mut drafter = ScriptedDrafter::new(vec![Vec::new()]);
        let config = GenerationConfig::greedy(1, 2).expect("valid config");

        let output =
            speculative_greedy_decode(&mut model, &mut drafter, &[6], config).expect("decode");

        assert_eq!(output.tokens.as_slice(), &[6, 7]);
        assert_eq!(output.stats.draft_rounds, 1);
        assert_eq!(output.stats.drafted_tokens, 0);
        assert_eq!(output.stats.accepted_draft_tokens, 0);
        assert_eq!(output.stats.rejected_draft_tokens, 0);
        assert_eq!(output.stats.target_forwards, 1);
    }
}
