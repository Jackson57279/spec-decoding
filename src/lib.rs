pub mod model;

pub const CRATE_NAME: &str = "speclative-diffusion";

#[cfg(test)]
mod tests {
    use super::CRATE_NAME;

    #[test]
    fn exposes_crate_name() {
        assert_eq!(CRATE_NAME, "speclative-diffusion");
    }
}
