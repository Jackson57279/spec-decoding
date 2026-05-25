pub mod adapters;
pub mod block_draft;
pub mod decode;
pub mod drafters;
pub mod loading;
pub mod model;
pub mod runtime;
pub mod spec_decode;

pub const CRATE_NAME: &str = "speclative-diffusion";

#[cfg(test)]
mod tests {
    use super::CRATE_NAME;

    #[test]
    fn exposes_crate_name() {
        assert_eq!(CRATE_NAME, "speclative-diffusion");
    }
}
