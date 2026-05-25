use std::path::PathBuf;

use crate::{
    adapters::{AdapterLoadPreflight, AdapterLoaderShell, AdapterModelPreflight},
    loading::{ModelAssetPaths, ModelLoadRequest, WeightFormat},
    model::{ModelError, ModelResult},
    weight_metadata::SafeTensorsFileMetadata,
};

#[cfg(feature = "safetensors")]
use crate::weight_metadata::read_safetensors_file_metadata;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterWeightFilePreflight {
    pub path: PathBuf,
    pub format: WeightFormat,
    pub safetensors: Option<SafeTensorsFileMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterModelWeightPreflight {
    pub model: AdapterModelPreflight,
    pub weights: Vec<AdapterWeightFilePreflight>,
}

impl AdapterModelWeightPreflight {
    pub fn from_model_preflight(
        model: AdapterModelPreflight,
        assets: &ModelAssetPaths,
    ) -> ModelResult<Self> {
        Ok(Self {
            model,
            weights: read_weight_file_preflights(assets)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterLoadWeightPreflight {
    pub target: AdapterModelWeightPreflight,
    pub draft: Option<AdapterModelWeightPreflight>,
}

impl AdapterLoadWeightPreflight {
    pub fn from_load_preflight(
        preflight: AdapterLoadPreflight,
        request: &ModelLoadRequest,
    ) -> ModelResult<Self> {
        let draft = match (preflight.draft, &request.draft) {
            (Some(model), Some(assets)) => Some(AdapterModelWeightPreflight::from_model_preflight(
                model, assets,
            )?),
            (None, None) => None,
            _ => {
                return Err(ModelError::InvalidConfig(
                    "adapter preflight and request draft shape must match",
                ));
            }
        };

        Ok(Self {
            target: AdapterModelWeightPreflight::from_model_preflight(
                preflight.target,
                &request.target,
            )?,
            draft,
        })
    }
}

impl AdapterLoaderShell {
    pub fn preflight_target_only_with_weight_metadata(
        self,
        request: &ModelLoadRequest,
    ) -> ModelResult<AdapterLoadWeightPreflight> {
        let preflight = self.preflight_target_only(request)?;
        AdapterLoadWeightPreflight::from_load_preflight(preflight, request)
    }

    pub fn preflight_with_draft_weight_metadata(
        self,
        draft_kind: crate::adapters::AdapterKind,
        request: &ModelLoadRequest,
    ) -> ModelResult<AdapterLoadWeightPreflight> {
        let preflight = self.preflight_with_draft(draft_kind, request)?;
        AdapterLoadWeightPreflight::from_load_preflight(preflight, request)
    }
}

fn read_weight_file_preflights(
    assets: &ModelAssetPaths,
) -> ModelResult<Vec<AdapterWeightFilePreflight>> {
    let format = assets.weight_format()?;

    assets
        .weight_files
        .iter()
        .map(|path| read_weight_file_preflight(path.clone(), format))
        .collect()
}

fn read_weight_file_preflight(
    path: PathBuf,
    format: WeightFormat,
) -> ModelResult<AdapterWeightFilePreflight> {
    Ok(AdapterWeightFilePreflight {
        safetensors: read_optional_safetensors_metadata(&path, format)?,
        path,
        format,
    })
}

fn read_optional_safetensors_metadata(
    path: &PathBuf,
    format: WeightFormat,
) -> ModelResult<Option<SafeTensorsFileMetadata>> {
    match format {
        WeightFormat::SafeTensors => read_safetensors_metadata(path),
        WeightFormat::Gguf => Ok(None),
    }
}

#[cfg(feature = "safetensors")]
fn read_safetensors_metadata(path: &PathBuf) -> ModelResult<Option<SafeTensorsFileMetadata>> {
    read_safetensors_file_metadata(path).map(Some)
}

#[cfg(not(feature = "safetensors"))]
fn read_safetensors_metadata(_path: &PathBuf) -> ModelResult<Option<SafeTensorsFileMetadata>> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{File, create_dir_all, remove_dir_all, write},
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::{
        adapter_weight_preflight::AdapterLoadWeightPreflight,
        adapters::{AdapterKind, AdapterLoaderShell},
        loading::{ModelAssetPaths, ModelLoadRequest, WeightFormat},
    };

    struct TempAssets {
        root: PathBuf,
        config: PathBuf,
        tokenizer: PathBuf,
        weights: PathBuf,
    }

    impl TempAssets {
        fn new(name: &str, weight_name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time should be valid")
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "speclative-diffusion-weight-preflight-{name}-{}-{unique}",
                std::process::id()
            ));
            create_dir_all(&root).expect("temp dir should be created");

            let config = root.join("config.json");
            let tokenizer = root.join("tokenizer.json");
            let weights = root.join(weight_name);

            write(&config, r#"{"model_type":"llama","vocab_size":32000}"#)
                .expect("config should be written");
            write(
                &tokenizer,
                r#"{"model":{"type":"BPE","vocab":{"hello":0,"world":1}}}"#,
            )
            .expect("tokenizer should be written");
            File::create(&weights).expect("weights should be created");

            Self {
                root,
                config,
                tokenizer,
                weights,
            }
        }

        fn paths(&self) -> ModelAssetPaths {
            ModelAssetPaths::new(
                self.config.clone(),
                self.tokenizer.clone(),
                vec![self.weights.clone()],
            )
            .expect("asset paths should be valid")
        }

        #[cfg(feature = "safetensors")]
        fn write_safetensors(&self) {
            let header = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
            let mut bytes = Vec::new();
            bytes.extend((header.len() as u64).to_le_bytes());
            bytes.extend(header);
            bytes.extend([0_u8; 8]);
            write(&self.weights, bytes).expect("safetensors should be written");
        }
    }

    impl Drop for TempAssets {
        fn drop(&mut self) {
            let _ = remove_dir_all(&self.root);
        }
    }

    #[test]
    fn builds_weight_preflight_for_target_only_requests() {
        let target = TempAssets::new("target", "model.safetensors");
        #[cfg(feature = "safetensors")]
        target.write_safetensors();
        let request = ModelLoadRequest::target_only(target.paths());

        let preflight = AdapterLoaderShell::new(AdapterKind::Candle)
            .preflight_target_only_with_weight_metadata(&request)
            .expect("weight preflight should pass");

        assert_eq!(preflight.target.model.plan.kind, AdapterKind::Candle);
        assert_eq!(preflight.target.weights.len(), 1);
        assert_eq!(
            preflight.target.weights[0].format,
            WeightFormat::SafeTensors
        );
        assert_eq!(preflight.draft, None);

        #[cfg(feature = "safetensors")]
        assert_eq!(
            preflight.target.weights[0]
                .safetensors
                .as_ref()
                .expect("safetensors metadata")
                .tensor_count(),
            1
        );

        #[cfg(not(feature = "safetensors"))]
        assert_eq!(preflight.target.weights[0].safetensors, None);
    }

    #[test]
    fn builds_weight_preflight_for_target_and_draft_requests() {
        let target = TempAssets::new("target-draft", "model.safetensors");
        let draft = TempAssets::new("draft", "model.gguf");
        #[cfg(feature = "safetensors")]
        target.write_safetensors();
        let request = ModelLoadRequest::with_draft(target.paths(), draft.paths());

        let preflight = AdapterLoaderShell::new(AdapterKind::Candle)
            .preflight_with_draft_weight_metadata(AdapterKind::Gguf, &request)
            .expect("weight preflight should pass");
        let draft_preflight = preflight.draft.expect("draft weight preflight");

        assert_eq!(
            preflight.target.weights[0].format,
            WeightFormat::SafeTensors
        );
        assert_eq!(draft_preflight.weights[0].format, WeightFormat::Gguf);
        assert_eq!(draft_preflight.weights[0].safetensors, None);
    }

    #[test]
    fn rejects_shape_mismatches_between_preflight_and_request() {
        let target = TempAssets::new("shape", "model.safetensors");
        #[cfg(feature = "safetensors")]
        target.write_safetensors();
        let request = ModelLoadRequest::target_only(target.paths());
        let preflight = AdapterLoaderShell::new(AdapterKind::Candle)
            .preflight_target_only(&request)
            .expect("base preflight should pass");
        let mismatched = ModelLoadRequest::with_draft(request.target.clone(), request.target);

        assert_eq!(
            AdapterLoadWeightPreflight::from_load_preflight(preflight, &mismatched),
            Err(crate::model::ModelError::InvalidConfig(
                "adapter preflight and request draft shape must match"
            ))
        );
    }
}
