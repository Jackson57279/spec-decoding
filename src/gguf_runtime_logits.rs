use std::path::PathBuf;

use crate::{
    adapter_runtime_plan::{AdapterRuntimeGgufPlan, AdapterTargetRuntimePlan},
    model::{ModelError, ModelResult, TokenId},
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GgufRuntimeLogits {
    engine: GgufLogitsEngine,
    vocab_size: usize,
}

impl GgufRuntimeLogits {
    pub(crate) fn from_runtime_plan(
        plan: &AdapterTargetRuntimePlan,
        vocab_size: usize,
    ) -> ModelResult<Self> {
        if plan.vocab_size != Some(vocab_size) {
            return Err(ModelError::InvalidConfig(
                "gguf logits vocab size must match runtime plan",
            ));
        }

        let plan = PlanBoundGgufLogitsEngine::from_runtime_plan(plan, vocab_size)?;

        Ok(Self {
            engine: GgufLogitsEngine::from_plan_bound(plan),
            vocab_size,
        })
    }

    pub(crate) fn logits_for_prefix(&mut self, prefix: &[TokenId]) -> ModelResult<Vec<f32>> {
        let logits = self.engine.logits_for_prefix(prefix)?;
        if logits.len() != self.vocab_size {
            return Err(ModelError::InvalidConfig(
                "gguf logits length must match vocab size",
            ));
        }
        Ok(logits)
    }

    #[cfg(test)]
    pub(crate) fn test_static(vocab_size: usize, logits: Vec<f32>) -> Self {
        Self {
            engine: GgufLogitsEngine::Static(StaticGgufLogitsEngine::new(logits)),
            vocab_size,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum GgufLogitsEngine {
    #[cfg(feature = "gguf-llama-cpp")]
    LlamaCpp(LlamaCppGgufLogitsEngine),
    #[cfg(not(feature = "gguf-llama-cpp"))]
    PlanBound(PlanBoundGgufLogitsEngine),
    #[cfg(test)]
    Static(StaticGgufLogitsEngine),
}

impl GgufLogitsEngine {
    fn from_plan_bound(plan: PlanBoundGgufLogitsEngine) -> Self {
        #[cfg(feature = "gguf-llama-cpp")]
        {
            Self::LlamaCpp(LlamaCppGgufLogitsEngine::new(plan))
        }

        #[cfg(not(feature = "gguf-llama-cpp"))]
        {
            Self::PlanBound(plan)
        }
    }

    fn logits_for_prefix(&mut self, prefix: &[TokenId]) -> ModelResult<Vec<f32>> {
        match self {
            #[cfg(feature = "gguf-llama-cpp")]
            Self::LlamaCpp(engine) => engine.logits_for_prefix(prefix),
            #[cfg(not(feature = "gguf-llama-cpp"))]
            Self::PlanBound(engine) => {
                let _ = prefix;
                let _ = engine;
                Err(ModelError::InvalidConfig(
                    "gguf logits evaluator is not implemented",
                ))
            }
            #[cfg(test)]
            Self::Static(engine) => engine.logits_for_prefix(prefix),
        }
    }
}

#[cfg(feature = "gguf-llama-cpp")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct LlamaCppGgufLogitsEngine {
    plan: PlanBoundGgufLogitsEngine,
}

#[cfg(feature = "gguf-llama-cpp")]
impl LlamaCppGgufLogitsEngine {
    fn new(plan: PlanBoundGgufLogitsEngine) -> Self {
        Self { plan }
    }

    fn logits_for_prefix(&mut self, prefix: &[TokenId]) -> ModelResult<Vec<f32>> {
        let _ = prefix;
        let _ = &self.plan;
        Err(ModelError::InvalidConfig(
            "gguf logits evaluator is not implemented",
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlanBoundGgufLogitsEngine {
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    weights: Vec<PlanBoundGgufWeight>,
}

impl PlanBoundGgufLogitsEngine {
    fn from_runtime_plan(plan: &AdapterTargetRuntimePlan, vocab_size: usize) -> ModelResult<Self> {
        let model_type = required_text(
            plan.model_type.as_deref(),
            "gguf logits model type is required",
        )?
        .to_owned();
        let hidden_size = required_usize(plan.hidden_size, "gguf logits hidden size is required")?;
        let num_hidden_layers = required_usize(
            plan.num_hidden_layers,
            "gguf logits hidden layer count is required",
        )?;

        if plan.weights.is_empty() {
            return Err(ModelError::InvalidConfig(
                "gguf logits require at least one weight file",
            ));
        }

        let weights = plan
            .weights
            .iter()
            .map(|weight| {
                let gguf = weight.gguf.as_ref().ok_or(ModelError::InvalidConfig(
                    "gguf logits require gguf weight metadata",
                ))?;
                Ok(PlanBoundGgufWeight::from_plan(weight.path.clone(), gguf))
            })
            .collect::<ModelResult<Vec<_>>>()?;

        Ok(Self {
            model_type,
            vocab_size,
            hidden_size,
            num_hidden_layers,
            weights,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlanBoundGgufWeight {
    path: PathBuf,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    header_bytes: usize,
    architecture: Option<String>,
    parsed_tensor_count: usize,
}

impl PlanBoundGgufWeight {
    fn from_plan(path: PathBuf, gguf: &AdapterRuntimeGgufPlan) -> Self {
        Self {
            path,
            version: gguf.version,
            tensor_count: gguf.tensor_count,
            metadata_kv_count: gguf.metadata_kv_count,
            header_bytes: gguf.header_bytes,
            architecture: gguf.architecture.clone(),
            parsed_tensor_count: gguf.parsed_tensor_count,
        }
    }
}

fn required_text<'a>(value: Option<&'a str>, message: &'static str) -> ModelResult<&'a str> {
    match value {
        Some(value) if !value.is_empty() => Ok(value),
        _ => Err(ModelError::InvalidConfig(message)),
    }
}

fn required_usize(value: Option<usize>, message: &'static str) -> ModelResult<usize> {
    match value {
        Some(value) if value > 0 => Ok(value),
        _ => Err(ModelError::InvalidConfig(message)),
    }
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct StaticGgufLogitsEngine {
    logits: Vec<f32>,
    calls: Vec<Vec<TokenId>>,
}

#[cfg(test)]
impl StaticGgufLogitsEngine {
    fn new(logits: Vec<f32>) -> Self {
        Self {
            logits,
            calls: Vec::new(),
        }
    }

    fn logits_for_prefix(&mut self, prefix: &[TokenId]) -> ModelResult<Vec<f32>> {
        self.calls.push(prefix.to_vec());
        Ok(self.logits.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{
        adapter_runtime_plan::{
            AdapterRuntimeGgufPlan, AdapterRuntimeWeightFilePlan, AdapterTargetRuntimePlan,
        },
        adapters::AdapterKind,
        gguf_runtime_logits::GgufRuntimeLogits,
        loading::WeightFormat,
        model::{ModelError, TokenId},
    };

    fn runtime_plan() -> AdapterTargetRuntimePlan {
        AdapterTargetRuntimePlan {
            kind: AdapterKind::Gguf,
            model_type: Some("llama".to_owned()),
            vocab_size: Some(2),
            hidden_size: Some(4),
            num_hidden_layers: Some(2),
            tokenizer_model_type: None,
            tokenizer_vocab_size: None,
            weight_format: WeightFormat::Gguf,
            weights: vec![AdapterRuntimeWeightFilePlan {
                path: PathBuf::from("/tmp/model.gguf"),
                format: WeightFormat::Gguf,
                safetensors: None,
                gguf: Some(AdapterRuntimeGgufPlan {
                    version: 3,
                    tensor_count: 1,
                    metadata_kv_count: 1,
                    header_bytes: 128,
                    architecture: Some("llama".to_owned()),
                    parsed_tensor_count: 1,
                }),
            }],
        }
    }

    #[test]
    fn static_gguf_logits_engine_returns_logits() {
        let mut logits = GgufRuntimeLogits::test_static(2, vec![0.25, 0.75]);

        assert_eq!(logits.logits_for_prefix(&[0, 1]), Ok(vec![0.25, 0.75]));
    }

    #[test]
    fn rejects_wrong_gguf_logits_length() {
        let mut logits = GgufRuntimeLogits::test_static(3, vec![0.25, 0.75]);

        assert_eq!(
            logits.logits_for_prefix(&[0]),
            Err(ModelError::InvalidConfig(
                "gguf logits length must match vocab size"
            ))
        );
    }

    #[test]
    fn binds_gguf_logits_to_runtime_plan() {
        let mut logits =
            GgufRuntimeLogits::from_runtime_plan(&runtime_plan(), 2).expect("plan should bind");
        let prefix: &[TokenId] = &[0];

        match &logits.engine {
            #[cfg(feature = "gguf-llama-cpp")]
            super::GgufLogitsEngine::LlamaCpp(engine) => {
                assert_bound_plan(&engine.plan);
            }
            #[cfg(not(feature = "gguf-llama-cpp"))]
            super::GgufLogitsEngine::PlanBound(engine) => {
                assert_bound_plan(engine);
            }
            _ => panic!("expected plan-bound engine"),
        }

        assert_eq!(
            logits.logits_for_prefix(prefix),
            Err(ModelError::InvalidConfig(
                "gguf logits evaluator is not implemented"
            ))
        );
    }

    fn assert_bound_plan(engine: &super::PlanBoundGgufLogitsEngine) {
        assert_eq!(engine.model_type, "llama");
        assert_eq!(engine.vocab_size, 2);
        assert_eq!(engine.hidden_size, 4);
        assert_eq!(engine.num_hidden_layers, 2);
        assert_eq!(engine.weights.len(), 1);
        assert_eq!(engine.weights[0].architecture.as_deref(), Some("llama"));
    }

    #[test]
    fn rejects_gguf_logits_vocab_mismatch() {
        assert_eq!(
            GgufRuntimeLogits::from_runtime_plan(&runtime_plan(), 3),
            Err(ModelError::InvalidConfig(
                "gguf logits vocab size must match runtime plan"
            ))
        );
    }
}
