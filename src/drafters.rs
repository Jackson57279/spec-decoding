use crate::model::{ModelError, ModelResult, TokenId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftSequence {
    tokens: Vec<TokenId>,
}

impl DraftSequence {
    pub fn new(tokens: Vec<TokenId>) -> Self {
        Self { tokens }
    }

    pub fn as_slice(&self) -> &[TokenId] {
        &self.tokens
    }

    pub fn into_tokens(self) -> Vec<TokenId> {
        self.tokens
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

pub trait Drafter {
    fn draft(&mut self, prefix: &[TokenId], max_tokens: usize) -> ModelResult<DraftSequence>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptLookupDrafter {
    min_match: usize,
    max_match: usize,
}

impl PromptLookupDrafter {
    pub fn new(min_match: usize, max_match: usize) -> ModelResult<Self> {
        if min_match == 0 {
            return Err(ModelError::InvalidConfig(
                "min_match must be greater than zero",
            ));
        }

        if max_match < min_match {
            return Err(ModelError::InvalidConfig(
                "max_match must be greater than or equal to min_match",
            ));
        }

        Ok(Self {
            min_match,
            max_match,
        })
    }
}

impl Drafter for PromptLookupDrafter {
    fn draft(&mut self, prefix: &[TokenId], max_tokens: usize) -> ModelResult<DraftSequence> {
        if max_tokens == 0 {
            return Err(ModelError::InvalidConfig(
                "max_tokens must be greater than zero",
            ));
        }

        if prefix.len() <= self.min_match {
            return Ok(DraftSequence::new(Vec::new()));
        }

        let max_match = self.max_match.min(prefix.len());

        for match_len in (self.min_match..=max_match).rev() {
            let suffix_start = prefix.len() - match_len;
            let suffix = &prefix[suffix_start..];

            for start in (0..suffix_start).rev() {
                let end = start + match_len;
                if end > suffix_start || &prefix[start..end] != suffix {
                    continue;
                }

                let draft_start = end;
                let draft_end = (draft_start + max_tokens).min(prefix.len());
                if draft_start < draft_end {
                    return Ok(DraftSequence::new(prefix[draft_start..draft_end].to_vec()));
                }
            }
        }

        Ok(DraftSequence::new(Vec::new()))
    }
}

#[cfg(test)]
mod tests {
    use crate::model::ModelError;

    use super::{Drafter, PromptLookupDrafter};

    #[test]
    fn drafts_tokens_after_longest_suffix_match() {
        let mut drafter = PromptLookupDrafter::new(2, 4).expect("valid config");

        let draft = drafter
            .draft(&[7, 8, 9, 1, 2, 3, 4, 1, 2, 3], 3)
            .expect("draft should succeed");

        assert_eq!(draft.as_slice(), &[4, 1, 2]);
    }

    #[test]
    fn prefers_the_newest_matching_prefix() {
        let mut drafter = PromptLookupDrafter::new(2, 2).expect("valid config");

        let draft = drafter
            .draft(&[1, 2, 3, 1, 2, 4, 1, 2], 2)
            .expect("draft should succeed");

        assert_eq!(draft.as_slice(), &[4, 1]);
    }

    #[test]
    fn returns_empty_draft_when_no_match_exists() {
        let mut drafter = PromptLookupDrafter::new(2, 3).expect("valid config");

        let draft = drafter
            .draft(&[1, 2, 3, 4], 2)
            .expect("draft should succeed");

        assert!(draft.is_empty());
    }

    #[test]
    fn validates_configuration_and_request_limits() {
        assert_eq!(
            PromptLookupDrafter::new(0, 2),
            Err(ModelError::InvalidConfig(
                "min_match must be greater than zero"
            ))
        );
        assert_eq!(
            PromptLookupDrafter::new(3, 2),
            Err(ModelError::InvalidConfig(
                "max_match must be greater than or equal to min_match"
            ))
        );

        let mut drafter = PromptLookupDrafter::new(1, 2).expect("valid config");
        assert_eq!(
            drafter.draft(&[1, 1], 0),
            Err(ModelError::InvalidConfig(
                "max_tokens must be greater than zero"
            ))
        );
    }
}
