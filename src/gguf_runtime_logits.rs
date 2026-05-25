use crate::model::{ModelError, ModelResult, TokenId};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GgufRuntimeLogits {
    engine: GgufLogitsEngine,
    vocab_size: usize,
}

impl GgufRuntimeLogits {
    pub(crate) fn unavailable(vocab_size: usize) -> Self {
        Self {
            engine: GgufLogitsEngine::Unavailable,
            vocab_size,
        }
    }

    pub(crate) fn logits_for_prefix(&mut self, prefix: &[TokenId]) -> ModelResult<Vec<f32>> {
        let logits = self.engine.logits_for_prefix(prefix)?;
        if logits.len() != self.vocab_size {
            return Err(ModelError::InvalidConfig(
                "gguf logits length must match vocab size",
            ));
        }
        Ok(logits)
    }

    #[cfg(test)]
    pub(crate) fn test_static(vocab_size: usize, logits: Vec<f32>) -> Self {
        Self {
            engine: GgufLogitsEngine::Static(StaticGgufLogitsEngine::new(logits)),
            vocab_size,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum GgufLogitsEngine {
    Unavailable,
    #[cfg(test)]
    Static(StaticGgufLogitsEngine),
}

impl GgufLogitsEngine {
    fn logits_for_prefix(&mut self, prefix: &[TokenId]) -> ModelResult<Vec<f32>> {
        match self {
            Self::Unavailable => Err(ModelError::InvalidConfig(
                "gguf logits engine is not configured",
            )),
            #[cfg(test)]
            Self::Static(engine) => engine.logits_for_prefix(prefix),
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct StaticGgufLogitsEngine {
    logits: Vec<f32>,
    calls: Vec<Vec<TokenId>>,
}

#[cfg(test)]
impl StaticGgufLogitsEngine {
    fn new(logits: Vec<f32>) -> Self {
        Self {
            logits,
            calls: Vec::new(),
        }
    }

    fn logits_for_prefix(&mut self, prefix: &[TokenId]) -> ModelResult<Vec<f32>> {
        self.calls.push(prefix.to_vec());
        Ok(self.logits.clone())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        gguf_runtime_logits::GgufRuntimeLogits,
        model::{ModelError, TokenId},
    };

    #[test]
    fn static_gguf_logits_engine_returns_logits() {
        let mut logits = GgufRuntimeLogits::test_static(2, vec![0.25, 0.75]);

        assert_eq!(logits.logits_for_prefix(&[0, 1]), Ok(vec![0.25, 0.75]));
    }

    #[test]
    fn rejects_wrong_gguf_logits_length() {
        let mut logits = GgufRuntimeLogits::test_static(3, vec![0.25, 0.75]);

        assert_eq!(
            logits.logits_for_prefix(&[0]),
            Err(ModelError::InvalidConfig(
                "gguf logits length must match vocab size"
            ))
        );
    }

    #[test]
    fn unavailable_gguf_logits_engine_fails_explicitly() {
        let mut logits = GgufRuntimeLogits::unavailable(2);
        let prefix: &[TokenId] = &[0];

        assert_eq!(
            logits.logits_for_prefix(prefix),
            Err(ModelError::InvalidConfig(
                "gguf logits engine is not configured"
            ))
        );
    }
}
