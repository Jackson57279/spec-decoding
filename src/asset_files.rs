use std::path::Path;

use crate::{
    loading::{ModelAssetPaths, ModelLoadRequest},
    model::{ModelError, ModelResult},
};

pub fn validate_model_asset_files(assets: &ModelAssetPaths) -> ModelResult<()> {
    validate_file(&assets.config_file, "config file must exist")?;
    validate_file(&assets.tokenizer_file, "tokenizer file must exist")?;

    for weight_file in &assets.weight_files {
        validate_file(weight_file, "weight file must exist")?;
    }

    Ok(())
}

pub fn validate_load_request_files(request: &ModelLoadRequest) -> ModelResult<()> {
    validate_model_asset_files(&request.target)?;

    if let Some(draft) = &request.draft {
        validate_model_asset_files(draft)?;
    }

    Ok(())
}

fn validate_file(path: &Path, message: &'static str) -> ModelResult<()> {
    if path.is_file() {
        return Ok(());
    }

    Err(ModelError::InvalidConfig(message))
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{File, create_dir_all, remove_dir_all},
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::{
        asset_files::{validate_load_request_files, validate_model_asset_files},
        loading::{ModelAssetPaths, ModelLoadRequest},
        model::ModelError,
    };

    struct TempAssets {
        root: PathBuf,
        config: PathBuf,
        tokenizer: PathBuf,
        weights: PathBuf,
    }

    impl TempAssets {
        fn new(name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time should be valid")
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "speclative-diffusion-{name}-{}-{unique}",
                std::process::id()
            ));
            create_dir_all(&root).expect("temp dir should be created");

            Self {
                config: root.join("config.json"),
                tokenizer: root.join("tokenizer.json"),
                weights: root.join("model.safetensors"),
                root,
            }
        }

        fn create_all(&self) {
            File::create(&self.config).expect("config should be created");
            File::create(&self.tokenizer).expect("tokenizer should be created");
            File::create(&self.weights).expect("weights should be created");
        }

        fn paths(&self) -> ModelAssetPaths {
            ModelAssetPaths::new(
                self.config.clone(),
                self.tokenizer.clone(),
                vec![self.weights.clone()],
            )
            .expect("asset paths should be valid")
        }
    }

    impl Drop for TempAssets {
        fn drop(&mut self) {
            let _ = remove_dir_all(&self.root);
        }
    }

    #[test]
    fn validates_existing_asset_files() {
        let assets = TempAssets::new("existing");
        assets.create_all();

        assert_eq!(validate_model_asset_files(&assets.paths()), Ok(()));
    }

    #[test]
    fn rejects_missing_asset_files() {
        let assets = TempAssets::new("missing");
        File::create(&assets.config).expect("config should be created");
        File::create(&assets.tokenizer).expect("tokenizer should be created");

        assert_eq!(
            validate_model_asset_files(&assets.paths()),
            Err(ModelError::InvalidConfig("weight file must exist"))
        );
    }

    #[test]
    fn validates_target_and_draft_request_files() {
        let target = TempAssets::new("target");
        let draft = TempAssets::new("draft");
        target.create_all();
        draft.create_all();
        let request = ModelLoadRequest::with_draft(target.paths(), draft.paths());

        assert_eq!(validate_load_request_files(&request), Ok(()));
    }
}
