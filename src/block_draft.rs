use crate::{
    drafters::DraftSequence,
    model::{ModelError, ModelResult, TokenId},
};

#[derive(Debug, Clone, PartialEq)]
pub struct TargetFeatureWindow {
    rows: usize,
    width: usize,
    values: Vec<f32>,
}

impl TargetFeatureWindow {
    pub fn new(rows: usize, width: usize, values: Vec<f32>) -> ModelResult<Self> {
        if rows == 0 {
            return Err(ModelError::InvalidConfig(
                "feature rows must be greater than zero",
            ));
        }

        if width == 0 {
            return Err(ModelError::InvalidConfig(
                "feature width must be greater than zero",
            ));
        }

        let expected = rows * width;
        if values.len() != expected {
            return Err(ModelError::InvalidConfig(
                "feature values length must equal rows times width",
            ));
        }

        if values.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::InvalidConfig("feature values must be finite"));
        }

        Ok(Self {
            rows,
            width,
            values,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn row(&self, index: usize) -> Option<&[f32]> {
        if index >= self.rows {
            return None;
        }

        let start = index * self.width;
        let end = start + self.width;
        Some(&self.values[start..end])
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BlockDraftRequest<'a> {
    pub prefix: &'a [TokenId],
    pub last_verified: Option<TokenId>,
    pub target_features: &'a TargetFeatureWindow,
    pub max_tokens: usize,
}

impl<'a> BlockDraftRequest<'a> {
    pub fn new(
        prefix: &'a [TokenId],
        target_features: &'a TargetFeatureWindow,
        max_tokens: usize,
    ) -> ModelResult<Self> {
        if max_tokens == 0 {
            return Err(ModelError::InvalidConfig(
                "max_tokens must be greater than zero",
            ));
        }

        Ok(Self {
            prefix,
            last_verified: prefix.last().copied(),
            target_features,
            max_tokens,
        })
    }
}

pub trait BlockDrafter {
    fn draft_block(&mut self, request: BlockDraftRequest<'_>) -> ModelResult<DraftSequence>;
}

pub trait TargetFeatureExtractor {
    fn extract_target_features(&mut self, prefix: &[TokenId]) -> ModelResult<TargetFeatureWindow>;
}

#[cfg(test)]
mod tests {
    use crate::{
        block_draft::{
            BlockDraftRequest, BlockDrafter, TargetFeatureExtractor, TargetFeatureWindow,
        },
        drafters::DraftSequence,
        model::{ModelError, TokenId},
    };

    struct RecordingBlockDrafter {
        seen_last: Option<TokenId>,
        seen_width: usize,
    }

    impl RecordingBlockDrafter {
        fn new() -> Self {
            Self {
                seen_last: None,
                seen_width: 0,
            }
        }
    }

    impl BlockDrafter for RecordingBlockDrafter {
        fn draft_block(
            &mut self,
            request: BlockDraftRequest<'_>,
        ) -> Result<DraftSequence, ModelError> {
            self.seen_last = request.last_verified;
            self.seen_width = request.target_features.width();
            Ok(DraftSequence::new(
                request
                    .prefix
                    .iter()
                    .copied()
                    .rev()
                    .take(request.max_tokens)
                    .collect(),
            ))
        }
    }

    struct PrefixFeatureExtractor;

    impl TargetFeatureExtractor for PrefixFeatureExtractor {
        fn extract_target_features(
            &mut self,
            prefix: &[TokenId],
        ) -> Result<TargetFeatureWindow, ModelError> {
            TargetFeatureWindow::new(
                prefix.len().max(1),
                2,
                prefix
                    .iter()
                    .flat_map(|token| [*token as f32, (*token as f32) + 0.5])
                    .chain((prefix.is_empty()).then_some(0.0))
                    .chain((prefix.is_empty()).then_some(0.5))
                    .collect(),
            )
        }
    }

    #[test]
    fn validates_feature_window_shape() {
        assert_eq!(
            TargetFeatureWindow::new(0, 2, Vec::new()),
            Err(ModelError::InvalidConfig(
                "feature rows must be greater than zero"
            ))
        );
        assert_eq!(
            TargetFeatureWindow::new(1, 0, Vec::new()),
            Err(ModelError::InvalidConfig(
                "feature width must be greater than zero"
            ))
        );
        assert_eq!(
            TargetFeatureWindow::new(2, 3, vec![0.0; 5]),
            Err(ModelError::InvalidConfig(
                "feature values length must equal rows times width"
            ))
        );
        assert_eq!(
            TargetFeatureWindow::new(1, 1, vec![f32::NAN]),
            Err(ModelError::InvalidConfig("feature values must be finite"))
        );
    }

    #[test]
    fn exposes_feature_rows() {
        let window =
            TargetFeatureWindow::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid");

        assert_eq!(window.rows(), 2);
        assert_eq!(window.width(), 3);
        assert_eq!(window.row(0), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(window.row(1), Some(&[4.0, 5.0, 6.0][..]));
        assert_eq!(window.row(2), None);
    }

    #[test]
    fn request_tracks_last_verified_token() {
        let window = TargetFeatureWindow::new(1, 2, vec![0.5, 1.5]).expect("valid");
        let request = BlockDraftRequest::new(&[7, 8, 9], &window, 4).expect("valid");

        assert_eq!(request.last_verified, Some(9));
        assert_eq!(request.max_tokens, 4);
    }

    #[test]
    fn block_drafter_receives_conditioning_request() {
        let window = TargetFeatureWindow::new(1, 2, vec![0.5, 1.5]).expect("valid");
        let request = BlockDraftRequest::new(&[7, 8, 9], &window, 2).expect("valid");
        let mut drafter = RecordingBlockDrafter::new();

        let draft = drafter.draft_block(request).expect("draft");

        assert_eq!(draft.as_slice(), &[9, 8]);
        assert_eq!(drafter.seen_last, Some(9));
        assert_eq!(drafter.seen_width, 2);
    }

    #[test]
    fn feature_extractor_builds_block_request_conditioning() {
        let mut extractor = PrefixFeatureExtractor;
        let features = extractor
            .extract_target_features(&[3, 4])
            .expect("features");
        let request = BlockDraftRequest::new(&[3, 4], &features, 2).expect("request");

        assert_eq!(request.target_features.rows(), 2);
        assert_eq!(request.target_features.width(), 2);
        assert_eq!(request.target_features.row(0), Some(&[3.0, 3.5][..]));
        assert_eq!(request.target_features.row(1), Some(&[4.0, 4.5][..]));
    }

    #[test]
    fn request_rejects_zero_max_tokens() {
        let window = TargetFeatureWindow::new(1, 2, vec![0.5, 1.5]).expect("valid");

        assert_eq!(
            BlockDraftRequest::new(&[1], &window, 0).map(|request| request.max_tokens),
            Err(ModelError::InvalidConfig(
                "max_tokens must be greater than zero"
            ))
        );
    }
}
