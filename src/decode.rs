use crate::model::{
    GenerationConfig, ModelResult, TargetModel, TokenId, TokenSequence, greedy_token,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub model_forwards: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeOutput {
    pub tokens: TokenSequence,
    pub stats: DecodeStats,
}

pub fn greedy_decode<M: TargetModel>(
    model: &mut M,
    prompt: &[TokenId],
    config: GenerationConfig,
) -> ModelResult<DecodeOutput> {
    config.validate()?;

    let mut tokens = TokenSequence::new(prompt.to_vec());
    let mut model_forwards = 0;

    for _ in 0..config.max_new_tokens {
        let logits = model.logits_for_prefix(tokens.as_slice())?;
        model_forwards += 1;
        let next = greedy_token(&logits)?;
        tokens.push(next);
    }

    Ok(DecodeOutput {
        tokens,
        stats: DecodeStats {
            prompt_tokens: prompt.len(),
            generated_tokens: config.max_new_tokens,
            model_forwards,
        },
    })
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use crate::model::{GenerationConfig, ModelError};

    use super::{TargetModel, TokenId, greedy_decode};

    struct ScriptedModel {
        logits: VecDeque<Vec<f32>>,
        prefixes: Vec<Vec<TokenId>>,
    }

    impl ScriptedModel {
        fn new(logits: Vec<Vec<f32>>) -> Self {
            Self {
                logits: logits.into(),
                prefixes: Vec::new(),
            }
        }
    }

    impl TargetModel for ScriptedModel {
        fn vocab_size(&self) -> usize {
            4
        }

        fn logits_for_prefix(&mut self, prefix: &[TokenId]) -> Result<Vec<f32>, ModelError> {
            self.prefixes.push(prefix.to_vec());
            self.logits.pop_front().ok_or(ModelError::EmptyLogits)
        }
    }

    #[test]
    fn appends_greedy_tokens_and_tracks_stats() {
        let mut model = ScriptedModel::new(vec![
            vec![0.0, 5.0, 1.0, 0.5],
            vec![2.0, 1.0, 6.0, 0.5],
            vec![2.0, 1.0, 0.5, 7.0],
        ]);
        let config = GenerationConfig::greedy(3, 2).expect("valid config");

        let output = greedy_decode(&mut model, &[10, 11], config).expect("decode should pass");

        assert_eq!(output.tokens.as_slice(), &[10, 11, 1, 2, 3]);
        assert_eq!(output.stats.prompt_tokens, 2);
        assert_eq!(output.stats.generated_tokens, 3);
        assert_eq!(output.stats.model_forwards, 3);
        assert_eq!(
            model.prefixes,
            vec![vec![10, 11], vec![10, 11, 1], vec![10, 11, 1, 2]]
        );
    }

    #[test]
    fn propagates_model_errors() {
        let mut model = ScriptedModel::new(Vec::new());
        let config = GenerationConfig::greedy(1, 1).expect("valid config");

        let result = greedy_decode(&mut model, &[1], config);

        assert_eq!(result, Err(ModelError::EmptyLogits));
    }
}
