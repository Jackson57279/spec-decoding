use std::{env, path::PathBuf};

use crate::{
    adapters::gguf,
    loading::{ModelAssetPaths, ModelLoadRequest},
    model::{ModelError, TargetModel},
};

const CONFIG_ENV: &str = "SPECLATIVE_DIFFUSION_GGUF_SMOKE_CONFIG";
const TOKENIZER_ENV: &str = "SPECLATIVE_DIFFUSION_GGUF_SMOKE_TOKENIZER";
const WEIGHTS_ENV: &str = "SPECLATIVE_DIFFUSION_GGUF_SMOKE_WEIGHTS";

#[test]
fn env_gated_gguf_backend_smoke_binds_real_assets() {
    let Some(paths) = smoke_paths_from_env() else {
        return;
    };
    let expected_weight = paths.weight_files[0].clone();
    let request = ModelLoadRequest::target_only(paths);

    let mut bundle = gguf::loader()
        .load_gguf_runtime_backend_bundle(&request)
        .expect("gguf smoke assets should build a backend bundle");

    assert!(!bundle.has_draft());
    assert!(bundle.target.weight_paths().contains(&expected_weight));
    assert_eq!(
        bundle.target.logits_for_prefix(&[0]),
        Err(ModelError::InvalidConfig(
            "gguf logits evaluator is not implemented"
        ))
    );
}

fn smoke_paths_from_env() -> Option<ModelAssetPaths> {
    let config = env_path(CONFIG_ENV)?;
    let tokenizer = env_path(TOKENIZER_ENV)?;
    let weights = env_path(WEIGHTS_ENV)?;

    Some(
        ModelAssetPaths::new(config, tokenizer, vec![weights])
            .expect("gguf smoke env paths should have valid extensions"),
    )
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name).map(PathBuf::from)
}
