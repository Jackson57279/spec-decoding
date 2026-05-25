use crate::loading::{ModelAssetPaths, ModelLoadRequest, WeightFormat};
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterModelPlan {
    pub kind: AdapterKind,
    pub weight_format: WeightFormat,
    pub weight_files: usize,
}

impl AdapterModelPlan {
    pub fn new(kind: AdapterKind, assets: &ModelAssetPaths) -> ModelResult<Self> {
        kind.validate_assets(assets)?;

        Ok(Self {
            kind,
            weight_format: assets.weight_format()?,
            weight_files: assets.weight_files.len(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterLoadPlan {
    pub target: AdapterModelPlan,
    pub draft: Option<AdapterModelPlan>,
}

impl AdapterLoadPlan {
    pub fn target_only(target_kind: AdapterKind, request: &ModelLoadRequest) -> ModelResult<Self> {
        request.validate()?;

        if request.has_draft() {
            return Err(ModelError::InvalidConfig(
                "target-only adapter plan must not include draft assets",
            ));
        }

        Ok(Self {
            target: AdapterModelPlan::new(target_kind, &request.target)?,
            draft: None,
        })
    }

    pub fn with_draft(
        target_kind: AdapterKind,
        draft_kind: AdapterKind,
        request: &ModelLoadRequest,
    ) -> ModelResult<Self> {
        request.validate()?;

        let draft = request.draft.as_ref().ok_or(ModelError::InvalidConfig(
            "draft adapter plan requires draft assets",
        ))?;

        Ok(Self {
            target: AdapterModelPlan::new(target_kind, &request.target)?,
            draft: Some(AdapterModelPlan::new(draft_kind, draft)?),
        })
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
        adapters::{AdapterKind, AdapterLoadPlan, AdapterModelPlan},
        loading::{ModelAssetPaths, ModelLoadRequest, WeightFormat},
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

    #[test]
    fn builds_adapter_model_plans() {
        let paths = assets("/models/qwen/model.safetensors").expect("valid assets");

        assert_eq!(
            AdapterModelPlan::new(AdapterKind::Candle, &paths),
            Ok(AdapterModelPlan {
                kind: AdapterKind::Candle,
                weight_format: WeightFormat::SafeTensors,
                weight_files: 1,
            })
        );
    }

    #[test]
    fn builds_target_only_adapter_load_plans() {
        let request =
            ModelLoadRequest::target_only(assets("/models/qwen/model.safetensors").unwrap());

        let plan = AdapterLoadPlan::target_only(AdapterKind::Candle, &request).unwrap();

        assert_eq!(plan.target.kind, AdapterKind::Candle);
        assert_eq!(plan.target.weight_format, WeightFormat::SafeTensors);
        assert_eq!(plan.draft, None);
    }

    #[test]
    fn builds_target_and_draft_adapter_load_plans() {
        let request = ModelLoadRequest::with_draft(
            assets("/models/qwen/model.safetensors").unwrap(),
            assets("/models/qwen-draft/model.gguf").unwrap(),
        );

        let plan =
            AdapterLoadPlan::with_draft(AdapterKind::Candle, AdapterKind::Gguf, &request).unwrap();

        assert_eq!(plan.target.kind, AdapterKind::Candle);
        assert_eq!(plan.draft.expect("draft plan").kind, AdapterKind::Gguf);
    }

    #[test]
    fn rejects_adapter_load_plan_shape_mismatches() {
        let target_only =
            ModelLoadRequest::target_only(assets("/models/qwen/model.safetensors").unwrap());
        let with_draft = ModelLoadRequest::with_draft(
            assets("/models/qwen/model.safetensors").unwrap(),
            assets("/models/qwen-draft/model.gguf").unwrap(),
        );

        assert_eq!(
            AdapterLoadPlan::with_draft(AdapterKind::Candle, AdapterKind::Gguf, &target_only),
            Err(ModelError::InvalidConfig(
                "draft adapter plan requires draft assets"
            ))
        );
        assert_eq!(
            AdapterLoadPlan::target_only(AdapterKind::Candle, &with_draft),
            Err(ModelError::InvalidConfig(
                "target-only adapter plan must not include draft assets"
            ))
        );
    }
}
