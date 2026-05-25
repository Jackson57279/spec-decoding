pub mod adapter_loaded;
pub mod adapter_runtime_plan;
pub mod adapter_weight_preflight;
pub mod adapters;
pub mod asset_files;
pub mod block_draft;
pub mod config;
pub mod decode;
pub mod drafters;
pub mod loading;
pub mod model;
pub mod runtime;
pub mod spec_decode;
pub mod weight_metadata;

pub const CRATE_NAME: &str = "speclative-diffusion";

#[cfg(test)]
mod tests {
    use super::CRATE_NAME;

    #[test]
    fn exposes_crate_name() {
        assert_eq!(CRATE_NAME, "speclative-diffusion");
    }
}
