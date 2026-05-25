use crate::loading::{ModelAssetPaths, WeightFormat};
use crate::model::{ModelError, ModelResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterKind {
    Candle,
    Gguf,
}

impl AdapterKind {
    pub fn expected_weight_format(self) -> WeightFormat {
        match self {
            Self::Candle => WeightFormat::SafeTensors,
            Self::Gguf => WeightFormat::Gguf,
        }
    }

    pub fn validate_assets(self, assets: &ModelAssetPaths) -> ModelResult<()> {
        let actual = assets.weight_format()?;
        if actual != self.expected_weight_format() {
            return Err(ModelError::InvalidConfig(
                "adapter does not support this weight format",
            ));
        }

        Ok(())
    }
}

#[cfg(feature = "candle")]
pub mod candle {
    use crate::adapters::AdapterKind;

    pub const KIND: AdapterKind = AdapterKind::Candle;
}

#[cfg(feature = "gguf")]
pub mod gguf {
    use crate::adapters::AdapterKind;

    pub const KIND: AdapterKind = AdapterKind::Gguf;
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{
        adapters::AdapterKind,
        loading::ModelAssetPaths,
        model::{ModelError, ModelResult},
    };

    fn assets(weight_file: &str) -> ModelResult<ModelAssetPaths> {
        ModelAssetPaths::new(
            "/models/qwen/config.json",
            "/models/qwen/tokenizer.json",
            vec![PathBuf::from(weight_file)],
        )
    }

    #[test]
    fn validates_candle_safetensor_assets() {
        let paths = assets("/models/qwen/model.safetensors").expect("valid assets");

        assert_eq!(AdapterKind::Candle.validate_assets(&paths), Ok(()));
    }

    #[test]
    fn validates_gguf_assets() {
        let paths = assets("/models/qwen/model.gguf").expect("valid assets");

        assert_eq!(AdapterKind::Gguf.validate_assets(&paths), Ok(()));
    }

    #[test]
    fn rejects_adapter_weight_format_mismatch() {
        let paths = assets("/models/qwen/model.gguf").expect("valid assets");

        assert_eq!(
            AdapterKind::Candle.validate_assets(&paths),
            Err(ModelError::InvalidConfig(
                "adapter does not support this weight format"
            ))
        );
    }
}
