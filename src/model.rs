pub type TokenId = u32;

pub type ModelResult<T> = Result<T, ModelError>;

#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    EmptyLogits,
    InvalidConfig(&'static str),
    InvalidLogit { index: usize },
    TokenOutOfRange { index: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSequence {
    tokens: Vec<TokenId>,
}

impl TokenSequence {
    pub fn new(tokens: Vec<TokenId>) -> Self {
        Self { tokens }
    }

    pub fn as_slice(&self) -> &[TokenId] {
        &self.tokens
    }

    pub fn push(&mut self, token: TokenId) {
        self.tokens.push(token);
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub speculative_tokens: usize,
    pub temperature: f32,
}

impl GenerationConfig {
    pub fn greedy(max_new_tokens: usize, speculative_tokens: usize) -> ModelResult<Self> {
        let config = Self {
            max_new_tokens,
            speculative_tokens,
            temperature: 0.0,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> ModelResult<()> {
        if self.max_new_tokens == 0 {
            return Err(ModelError::InvalidConfig(
                "max_new_tokens must be greater than zero",
            ));
        }

        if self.speculative_tokens == 0 {
            return Err(ModelError::InvalidConfig(
                "speculative_tokens must be greater than zero",
            ));
        }

        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(ModelError::InvalidConfig(
                "temperature must be finite and non-negative",
            ));
        }

        Ok(())
    }
}

pub trait TargetModel {
    fn vocab_size(&self) -> usize;
    fn logits_for_prefix(&mut self, prefix: &[TokenId]) -> ModelResult<Vec<f32>>;
}

pub fn greedy_token(logits: &[f32]) -> ModelResult<TokenId> {
    let mut best = None;

    for (index, score) in logits.iter().copied().enumerate() {
        if !score.is_finite() {
            return Err(ModelError::InvalidLogit { index });
        }

        match best {
            Some((_, best_score)) if score <= best_score => {}
            _ => best = Some((index, score)),
        }
    }

    let (index, _) = best.ok_or(ModelError::EmptyLogits)?;
    index
        .try_into()
        .map_err(|_| ModelError::TokenOutOfRange { index })
}

#[cfg(test)]
mod tests {
    use super::{GenerationConfig, ModelError, greedy_token};

    #[test]
    fn validates_greedy_generation_config() {
        let config = GenerationConfig::greedy(32, 8).expect("config should be valid");

        assert_eq!(config.max_new_tokens, 32);
        assert_eq!(config.speculative_tokens, 8);
        assert_eq!(config.temperature, 0.0);
    }

    #[test]
    fn rejects_empty_generation_limits() {
        assert_eq!(
            GenerationConfig::greedy(0, 8),
            Err(ModelError::InvalidConfig(
                "max_new_tokens must be greater than zero"
            ))
        );
        assert_eq!(
            GenerationConfig::greedy(32, 0),
            Err(ModelError::InvalidConfig(
                "speculative_tokens must be greater than zero"
            ))
        );
    }

    #[test]
    fn selects_first_max_logit() {
        let token = greedy_token(&[1.0, 3.5, 3.5, 2.0]).expect("logits should be valid");

        assert_eq!(token, 1);
    }

    #[test]
    fn rejects_empty_or_invalid_logits() {
        assert_eq!(greedy_token(&[]), Err(ModelError::EmptyLogits));
        assert_eq!(
            greedy_token(&[0.0, f32::NAN]),
            Err(ModelError::InvalidLogit { index: 1 })
        );
    }
}
