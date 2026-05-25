pub type TokenId = u32;

pub type ModelResult<T> = Result<T, ModelError>;

#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    EmptyLogits,
    InvalidConfig(&'static str),
    InvalidLogit { index: usize },
    TokenOutOfRange { index: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSequence {
    tokens: Vec<TokenId>,
}

impl TokenSequence {
    pub fn new(tokens: Vec<TokenId>) -> Self {
        Self { tokens }
    }

    pub fn as_slice(&self) -> &[TokenId] {
        &self.tokens
    }

    pub fn push(&mut self, token: TokenId) {
        self.tokens.push(token);
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub speculative_tokens: usize,
    pub temperature: f32,
}

impl GenerationConfig {
    pub fn greedy(max_new_tokens: usize, speculative_tokens: usize) -> ModelResult<Self> {
        let config = Self {
            max_new_tokens,
            speculative_tokens,
            temperature: 0.0,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> ModelResult<()> {
        if self.max_new_tokens == 0 {
            return Err(ModelError::InvalidConfig(
                "max_new_tokens must be greater than zero",
            ));
        }

        if self.speculative_tokens == 0 {
            return Err(ModelError::InvalidConfig(
                "speculative_tokens must be greater than zero",
            ));
        }

        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(ModelError::InvalidConfig(
                "temperature must be finite and non-negative",
            ));
        }

        Ok(())
    }
}

pub trait TargetModel {
    fn vocab_size(&self) -> usize;
    fn logits_for_prefix(&mut self, prefix: &[TokenId]) -> ModelResult<Vec<f32>>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetBatch {
    prefixes: Vec<TokenSequence>,
}

impl TargetBatch {
    pub fn new(prefixes: Vec<TokenSequence>) -> ModelResult<Self> {
        if prefixes.is_empty() {
            return Err(ModelError::InvalidConfig(
                "target batch must contain at least one prefix",
            ));
        }

        Ok(Self { prefixes })
    }

    pub fn from_prefix_and_draft(
        prefix: &[TokenId],
        draft_tokens: &[TokenId],
    ) -> ModelResult<Self> {
        if draft_tokens.is_empty() {
            return Err(ModelError::InvalidConfig(
                "draft tokens must not be empty for target verification",
            ));
        }

        let mut prefixes = Vec::with_capacity(draft_tokens.len());
        let mut current = prefix.to_vec();

        for token in draft_tokens {
            prefixes.push(TokenSequence::new(current.clone()));
            current.push(*token);
        }

        Self::new(prefixes)
    }

    pub fn as_slice(&self) -> &[TokenSequence] {
        &self.prefixes
    }

    pub fn len(&self) -> usize {
        self.prefixes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prefixes.is_empty()
    }
}

pub trait BatchedTargetModel {
    fn vocab_size(&self) -> usize;
    fn logits_for_prefixes(&mut self, batch: &TargetBatch) -> ModelResult<Vec<Vec<f32>>>;
}

impl<T> BatchedTargetModel for T
where
    T: TargetModel,
{
    fn vocab_size(&self) -> usize {
        TargetModel::vocab_size(self)
    }

    fn logits_for_prefixes(&mut self, batch: &TargetBatch) -> ModelResult<Vec<Vec<f32>>> {
        batch
            .as_slice()
            .iter()
            .map(|prefix| self.logits_for_prefix(prefix.as_slice()))
            .collect()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvCacheState {
    cached_tokens: usize,
}

impl KvCacheState {
    pub fn new(cached_tokens: usize) -> Self {
        Self { cached_tokens }
    }

    pub fn empty() -> Self {
        Self::default()
    }

    pub fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedTargetRequest {
    prefix: TokenSequence,
    cache: KvCacheState,
}

impl CachedTargetRequest {
    pub fn new(prefix: TokenSequence, cache: KvCacheState) -> ModelResult<Self> {
        if cache.cached_tokens() > prefix.len() {
            return Err(ModelError::InvalidConfig(
                "cache length must not exceed prefix length",
            ));
        }

        Ok(Self { prefix, cache })
    }

    pub fn prefix(&self) -> &[TokenId] {
        self.prefix.as_slice()
    }

    pub fn cache(&self) -> KvCacheState {
        self.cache
    }
}

pub trait CachedTargetModel {
    fn vocab_size(&self) -> usize;
    fn logits_with_cache(
        &mut self,
        request: &CachedTargetRequest,
    ) -> ModelResult<(Vec<f32>, KvCacheState)>;
}

impl<T> CachedTargetModel for T
where
    T: TargetModel,
{
    fn vocab_size(&self) -> usize {
        TargetModel::vocab_size(self)
    }

    fn logits_with_cache(
        &mut self,
        request: &CachedTargetRequest,
    ) -> ModelResult<(Vec<f32>, KvCacheState)> {
        let logits = self.logits_for_prefix(request.prefix())?;
        Ok((logits, KvCacheState::new(request.prefix().len())))
    }
}

pub trait Tokenizer {
    fn vocab_size(&self) -> usize;
    fn encode_with_options(
        &self,
        text: &str,
        options: TokenizerEncodeOptions,
    ) -> ModelResult<TokenSequence>;
    fn decode_with_options(
        &self,
        tokens: &[TokenId],
        options: TokenizerDecodeOptions,
    ) -> ModelResult<String>;

    fn encode(&self, text: &str) -> ModelResult<TokenSequence> {
        self.encode_with_options(text, TokenizerEncodeOptions::without_special_tokens())
    }

    fn decode(&self, tokens: &[TokenId]) -> ModelResult<String> {
        self.decode_with_options(tokens, TokenizerDecodeOptions::skip_special_tokens())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizerEncodeOptions {
    add_special_tokens: bool,
}

impl TokenizerEncodeOptions {
    pub fn new(add_special_tokens: bool) -> Self {
        Self { add_special_tokens }
    }

    pub fn with_special_tokens() -> Self {
        Self::new(true)
    }

    pub fn without_special_tokens() -> Self {
        Self::new(false)
    }

    pub fn add_special_tokens(&self) -> bool {
        self.add_special_tokens
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizerDecodeOptions {
    skip_special_tokens: bool,
}

impl TokenizerDecodeOptions {
    pub fn new(skip_special_tokens: bool) -> Self {
        Self {
            skip_special_tokens,
        }
    }

    pub fn skip_special_tokens() -> Self {
        Self::new(true)
    }

    pub fn preserve_special_tokens() -> Self {
        Self::new(false)
    }

    pub fn skip_special_tokens_enabled(&self) -> bool {
        self.skip_special_tokens
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn vocab_size(&self) -> usize {
        256
    }

    fn encode_with_options(
        &self,
        text: &str,
        _options: TokenizerEncodeOptions,
    ) -> ModelResult<TokenSequence> {
        Ok(TokenSequence::new(
            text.as_bytes().iter().copied().map(TokenId::from).collect(),
        ))
    }

    fn decode_with_options(
        &self,
        tokens: &[TokenId],
        _options: TokenizerDecodeOptions,
    ) -> ModelResult<String> {
        let mut bytes = Vec::with_capacity(tokens.len());

        for (index, token) in tokens.iter().copied().enumerate() {
            let byte = token
                .try_into()
                .map_err(|_| ModelError::TokenOutOfRange { index })?;
            bytes.push(byte);
        }

        String::from_utf8(bytes)
            .map_err(|_| ModelError::InvalidConfig("tokens must decode to valid UTF-8"))
    }
}

pub fn greedy_token(logits: &[f32]) -> ModelResult<TokenId> {
    let mut best = None;

    for (index, score) in logits.iter().copied().enumerate() {
        if !score.is_finite() {
            return Err(ModelError::InvalidLogit { index });
        }

        match best {
            Some((_, best_score)) if score <= best_score => {}
            _ => best = Some((index, score)),
        }
    }

    let (index, _) = best.ok_or(ModelError::EmptyLogits)?;
    index
        .try_into()
        .map_err(|_| ModelError::TokenOutOfRange { index })
}

#[cfg(test)]
mod tests {
    use super::{
        BatchedTargetModel, ByteTokenizer, CachedTargetModel, CachedTargetRequest,
        GenerationConfig, KvCacheState, ModelError, ModelResult, TargetBatch, TargetModel, TokenId,
        TokenSequence, Tokenizer, TokenizerDecodeOptions, TokenizerEncodeOptions, greedy_token,
    };

    #[test]
    fn validates_greedy_generation_config() {
        let config = GenerationConfig::greedy(32, 8).expect("config should be valid");

        assert_eq!(config.max_new_tokens, 32);
        assert_eq!(config.speculative_tokens, 8);
        assert_eq!(config.temperature, 0.0);
    }

    #[test]
    fn rejects_empty_generation_limits() {
        assert_eq!(
            GenerationConfig::greedy(0, 8),
            Err(ModelError::InvalidConfig(
                "max_new_tokens must be greater than zero"
            ))
        );
        assert_eq!(
            GenerationConfig::greedy(32, 0),
            Err(ModelError::InvalidConfig(
                "speculative_tokens must be greater than zero"
            ))
        );
    }

    #[test]
    fn selects_first_max_logit() {
        let token = greedy_token(&[1.0, 3.5, 3.5, 2.0]).expect("logits should be valid");

        assert_eq!(token, 1);
    }

    #[test]
    fn rejects_empty_or_invalid_logits() {
        assert_eq!(greedy_token(&[]), Err(ModelError::EmptyLogits));
        assert_eq!(
            greedy_token(&[0.0, f32::NAN]),
            Err(ModelError::InvalidLogit { index: 1 })
        );
    }

    #[test]
    fn byte_tokenizer_round_trips_utf8_text() {
        let tokenizer = ByteTokenizer;
        let encoded = tokenizer.encode("draft").expect("text should encode");

        assert_eq!(tokenizer.vocab_size(), 256);
        assert_eq!(encoded.as_slice(), &[100, 114, 97, 102, 116]);
        assert_eq!(
            tokenizer.decode(encoded.as_slice()),
            Ok(String::from("draft"))
        );
        assert_eq!(
            tokenizer.encode_with_options("draft", TokenizerEncodeOptions::with_special_tokens()),
            Ok(encoded)
        );
        assert_eq!(
            tokenizer.decode_with_options(
                &[100, 114, 97, 102, 116],
                TokenizerDecodeOptions::preserve_special_tokens(),
            ),
            Ok(String::from("draft"))
        );
    }

    #[test]
    fn byte_tokenizer_rejects_out_of_range_token() {
        let tokenizer = ByteTokenizer;

        assert_eq!(
            tokenizer.decode(&[256]),
            Err(ModelError::TokenOutOfRange { index: 0 })
        );
    }

    #[test]
    fn builds_target_batches_from_draft_tokens() {
        let batch =
            TargetBatch::from_prefix_and_draft(&[1, 2], &[3, 4]).expect("batch should be valid");

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.as_slice()[0].as_slice(), &[1, 2]);
        assert_eq!(batch.as_slice()[1].as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn rejects_empty_target_batches() {
        assert_eq!(
            TargetBatch::new(Vec::new()),
            Err(ModelError::InvalidConfig(
                "target batch must contain at least one prefix"
            ))
        );
        assert_eq!(
            TargetBatch::from_prefix_and_draft(&[1, 2], &[]),
            Err(ModelError::InvalidConfig(
                "draft tokens must not be empty for target verification"
            ))
        );
    }

    #[derive(Debug, Default)]
    struct RecordingTarget {
        calls: Vec<Vec<TokenId>>,
    }

    impl TargetModel for RecordingTarget {
        fn vocab_size(&self) -> usize {
            1
        }

        fn logits_for_prefix(&mut self, prefix: &[TokenId]) -> ModelResult<Vec<f32>> {
            self.calls.push(prefix.to_vec());
            Ok(vec![prefix.len() as f32])
        }
    }

    #[test]
    fn provides_sequential_batched_target_fallback() {
        let batch = TargetBatch::new(vec![
            TokenSequence::new(vec![10]),
            TokenSequence::new(vec![10, 11]),
        ])
        .expect("batch should be valid");
        let mut target = RecordingTarget::default();

        let logits = target
            .logits_for_prefixes(&batch)
            .expect("fallback should run");

        assert_eq!(BatchedTargetModel::vocab_size(&target), 1);
        assert_eq!(target.calls, vec![vec![10], vec![10, 11]]);
        assert_eq!(logits, vec![vec![1.0], vec![2.0]]);
    }

    #[test]
    fn validates_cached_target_requests() {
        let request =
            CachedTargetRequest::new(TokenSequence::new(vec![1, 2, 3]), KvCacheState::new(2))
                .expect("cache should be valid");

        assert_eq!(request.prefix(), &[1, 2, 3]);
        assert_eq!(request.cache().cached_tokens(), 2);
        assert_eq!(
            CachedTargetRequest::new(TokenSequence::new(vec![1]), KvCacheState::new(2)),
            Err(ModelError::InvalidConfig(
                "cache length must not exceed prefix length"
            ))
        );
    }

    #[test]
    fn provides_cached_target_fallback() {
        let request =
            CachedTargetRequest::new(TokenSequence::new(vec![20, 21]), KvCacheState::empty())
                .expect("request should be valid");
        let mut target = RecordingTarget::default();

        let (logits, cache) = target
            .logits_with_cache(&request)
            .expect("fallback should run");

        assert_eq!(CachedTargetModel::vocab_size(&target), 1);
        assert_eq!(target.calls, vec![vec![20, 21]]);
        assert_eq!(logits, vec![2.0]);
        assert_eq!(cache.cached_tokens(), 2);
    }
}
