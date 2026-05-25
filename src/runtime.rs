use crate::model::{GenerationConfig, ModelError, ModelResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeMode {
    Greedy,
    Speculative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterBackend {
    None,
    PromptLookup,
    FeatureConditionedBlock,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeConfig {
    pub generation: GenerationConfig,
    pub mode: DecodeMode,
    pub drafter: DrafterBackend,
}

impl RuntimeConfig {
    pub fn greedy(max_new_tokens: usize) -> ModelResult<Self> {
        Self::new(
            GenerationConfig::greedy(max_new_tokens, 1)?,
            DecodeMode::Greedy,
            DrafterBackend::None,
        )
    }

    pub fn speculative(
        max_new_tokens: usize,
        speculative_tokens: usize,
        drafter: DrafterBackend,
    ) -> ModelResult<Self> {
        Self::new(
            GenerationConfig::greedy(max_new_tokens, speculative_tokens)?,
            DecodeMode::Speculative,
            drafter,
        )
    }

    pub fn new(
        generation: GenerationConfig,
        mode: DecodeMode,
        drafter: DrafterBackend,
    ) -> ModelResult<Self> {
        let config = Self {
            generation,
            mode,
            drafter,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> ModelResult<()> {
        self.generation.validate()?;

        if self.generation.temperature != 0.0 {
            return Err(ModelError::InvalidConfig(
                "runtime currently supports greedy temperature only",
            ));
        }

        match (self.mode, self.drafter) {
            (DecodeMode::Greedy, DrafterBackend::None) => Ok(()),
            (DecodeMode::Greedy, _) => Err(ModelError::InvalidConfig(
                "greedy mode must not configure a drafter",
            )),
            (DecodeMode::Speculative, DrafterBackend::None) => Err(ModelError::InvalidConfig(
                "speculative mode requires a drafter",
            )),
            (DecodeMode::Speculative, _) => Ok(()),
        }
    }

    pub fn uses_drafter(&self) -> bool {
        self.drafter != DrafterBackend::None
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        model::{GenerationConfig, ModelError},
        runtime::{DecodeMode, DrafterBackend, RuntimeConfig},
    };

    #[test]
    fn builds_greedy_runtime_config() {
        let config = RuntimeConfig::greedy(16).expect("valid config");

        assert_eq!(config.mode, DecodeMode::Greedy);
        assert_eq!(config.drafter, DrafterBackend::None);
        assert_eq!(config.generation.max_new_tokens, 16);
        assert_eq!(config.generation.speculative_tokens, 1);
        assert!(!config.uses_drafter());
    }

    #[test]
    fn builds_speculative_runtime_config() {
        let config = RuntimeConfig::speculative(32, 8, DrafterBackend::FeatureConditionedBlock)
            .expect("valid config");

        assert_eq!(config.mode, DecodeMode::Speculative);
        assert_eq!(config.drafter, DrafterBackend::FeatureConditionedBlock);
        assert_eq!(config.generation.max_new_tokens, 32);
        assert_eq!(config.generation.speculative_tokens, 8);
        assert!(config.uses_drafter());
    }

    #[test]
    fn rejects_mode_and_drafter_mismatches() {
        let greedy_generation = GenerationConfig::greedy(8, 1).expect("valid generation");
        assert_eq!(
            RuntimeConfig::new(
                greedy_generation,
                DecodeMode::Greedy,
                DrafterBackend::PromptLookup,
            ),
            Err(ModelError::InvalidConfig(
                "greedy mode must not configure a drafter"
            ))
        );

        let speculative_generation = GenerationConfig::greedy(8, 2).expect("valid generation");
        assert_eq!(
            RuntimeConfig::new(
                speculative_generation,
                DecodeMode::Speculative,
                DrafterBackend::None,
            ),
            Err(ModelError::InvalidConfig(
                "speculative mode requires a drafter"
            ))
        );
    }

    #[test]
    fn rejects_sampling_temperature_until_sampling_is_supported() {
        let generation = GenerationConfig {
            max_new_tokens: 8,
            speculative_tokens: 2,
            temperature: 0.7,
        };

        assert_eq!(
            RuntimeConfig::new(
                generation,
                DecodeMode::Speculative,
                DrafterBackend::PromptLookup,
            ),
            Err(ModelError::InvalidConfig(
                "runtime currently supports greedy temperature only"
            ))
        );
    }
}
