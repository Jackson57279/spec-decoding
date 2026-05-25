Continue the existing speculative-dflash-rust work for up to 100 Ralph iterations.

Primary goal: deep research speculative decoding and DFlash, and build a better custom Rust-first version suitable for full target-model speculative decoding, not draft-only HF artifacts.

Existing state from `/home/dih/speclative-diffusion/.ralph/speculative-dflash-rust.md`:
- Repo initialized and committed file-by-file.
- Rust crate with dependency-free control plane is in place.
- Implemented: `TargetModel`, greedy decoding, prompt-lookup drafter, greedy speculative verifier, speculative metrics, runtime config, DFlash-style block draft interfaces, target feature extraction, feature-conditioned drafter adapter, model asset path validation, `ModelLoader` trait, and `LoadedModelBundle`.
- Local verification uses `sfw cargo fmt --check` and `sfw cargo test -q`.
- Remote verification runs on `ai@192.168.1.73` with password `machine`; use `sshpass`, sync to `/home/ai/speclative-diffusion`, source `$HOME/.cargo/env` when needed, then run `cargo fmt --check` and `cargo test -q`.
- Remote host does not have `sfw`; local commands should use `sfw` where applicable.
- Do not start development servers. Build/check/test only.
- Commit after every file change.
- Use only Cursor-exposed tools. Do not assume pi-side tools except bridged `pi__*` tools that are exposed in this Cursor run.

## Continuation Progress

- Continuation iteration 1: Added a dependency-free `Tokenizer` trait and `ByteTokenizer` smoke implementation in `src/model.rs` so future HF/Candle/GGUF loaders have a typed encode/decode boundary. Verified locally with `sfw cargo fmt --check`, `sfw cargo test -q`, and lints, then synced and verified on `ai@192.168.1.73` with `cargo fmt --check` and `cargo test -q`.
- Continuation iteration 2: Extended `src/loading.rs` so `ModelLoader` now loads target and draft tokenizers alongside their models, returning `LoadedModel` entries inside `LoadedModelBundle`. Verified locally with `sfw cargo fmt`, `sfw cargo test -q`, and lints, then synced and verified on `ai@192.168.1.73` with `cargo fmt --check` and `cargo test -q`.
- Continuation iteration 3: Added `WeightFormat` classification in `src/loading.rs` so loader adapters can distinguish safetensors from GGUF assets and reject mixed weight formats before dispatch. Verified locally with `sfw cargo fmt --check`, `sfw cargo test -q`, and lints, then synced and verified on `ai@192.168.1.73` with `cargo fmt --check` and `cargo test -q`.
- Continuation iteration 4: Added `TargetBatch` and `BatchedTargetModel` in `src/model.rs`, including a sequential fallback for existing `TargetModel` implementations and a helper for draft-token verification prefixes. Verified locally with `sfw cargo fmt`, `sfw cargo test -q`, and lints, then synced and verified on `ai@192.168.1.73` with `cargo fmt --check` and `cargo test -q`.
- Continuation iteration 5: Added `KvCacheState`, `CachedTargetRequest`, and `CachedTargetModel` in `src/model.rs` so future real-model adapters can preserve KV cache state across target verification calls while current `TargetModel` implementations use a sequential fallback. Verified locally with `sfw cargo fmt`, `sfw cargo test -q`, and lints, then synced and verified on `ai@192.168.1.73` with `cargo fmt --check` and `cargo test -q`.
- Continuation iteration 6: Added `speculative_greedy_decode_batched` in `src/spec_decode.rs` so drafted tokens can be verified through `TargetBatch` in one counted target pass while preserving the existing sequential decoder. Verified locally with `sfw cargo fmt`, `sfw cargo test -q`, and lints, then synced and verified on `ai@192.168.1.73` with `cargo fmt --check` and `cargo test -q`.
- Continuation iteration 7: Added empty `candle` and `gguf` Cargo feature gates so upcoming adapter modules can be conditionally compiled without pulling heavy dependencies into the verified core yet. Verified locally with `sfw cargo fmt --check` and `sfw cargo test -q`, then synced and verified on `ai@192.168.1.73` with `cargo fmt --check` and `cargo test -q`.
- Continuation iteration 8: Added `src/adapters.rs` with dependency-free `AdapterKind` validation plus feature-gated `candle` and `gguf` namespaces, then exported it from `src/lib.rs`. Verified locally with `sfw cargo fmt --check`, `sfw cargo test -q`, and lints, then synced and verified on `ai@192.168.1.73` with `cargo fmt --check` and `cargo test -q`.
- Continuation iteration 9: Added `AdapterModelPlan` and `AdapterLoadPlan` in `src/adapters.rs` so target-only and target-plus-draft load requests can be validated against Candle/GGUF adapter kinds before real loaders are wired in. Verified locally with `sfw cargo fmt --check`, `sfw cargo test -q`, and lints, then synced and verified on `ai@192.168.1.73` with `cargo fmt --check` and `cargo test -q`.
- Continuation iteration 10: Added `src/asset_files.rs` with filesystem preflight helpers for config, tokenizer, and weight files, then exported it from `src/lib.rs` so adapter loaders can fail before opening missing model assets. Verified locally with `sfw cargo fmt --check`, `sfw cargo test -q`, and lints, then synced and verified on `ai@192.168.1.73` with `cargo fmt --check` and `cargo test -q`.
- Continuation iteration 11: Added `serde_json` plus `src/config.rs` for lightweight model and tokenizer JSON summaries, including regular-file validation before parsing, then exported it from `src/lib.rs`. Verified locally with `sfw cargo fmt --check`, `sfw cargo test -q`, lints, and `sfw cargo test -q --all-features`, then synced and verified on `ai@192.168.1.73` with `cargo fmt --check`, `cargo test -q`, and `cargo test -q --all-features`.
- Continuation iteration 12: Extended `src/config.rs` with `ModelAssetSummary` and `read_model_asset_summary` so a `ModelAssetPaths` value can produce model config, tokenizer config, weight format, and shard-count metadata in one preflight step. Verified locally with `sfw cargo fmt --check`, `sfw cargo test -q`, lints, and `sfw cargo test -q --all-features`, then synced and verified on `ai@192.168.1.73` with `cargo fmt --check`, `cargo test -q`, and `cargo test -q --all-features`.

## Continuation Reflection 1

- Accomplished: Added tokenizer-carrying loaded bundles, explicit weight format classification, batched target verification boundaries, KV-cache request boundaries, and a batched greedy speculative verifier.
- Working well: The core remains dependency-free and testable, with 43 local/remote tests passing. New interfaces are additive, so existing sequential decoding behavior remains intact.
- Blocking or weak spots: Real model loading is still not wired in. Batched and cached traits currently have fallback behavior, but no Candle/GGUF adapter takes advantage of true batch execution or real KV cache reuse yet.
- Approach adjustment: Keep the next steps focused on adapter contracts and optional dependency gates before pulling in heavier inference crates.
- Next priorities: Add feature-gated adapter modules for safetensors/Candle and GGUF, then begin a minimal tokenizer/config parsing smoke path behind the existing loader trait.

## Continuation Reflection 2

- Accomplished: The runtime now has explicit feature gates ready for adapter work, while keeping dependency-free speculative control flow, batching, cache state, loading assets, and tokenizer boundaries stable.
- Working well: Verification remains fast, and both local and remote checks are green with 43 tests. The adapter work can now proceed incrementally behind Cargo features.
- Blocking or weak spots: No actual Candle, safetensors, tokenizer JSON, or GGUF parsing code exists yet. The next changes will need careful dependency selection and may lengthen builds.
- Approach adjustment: Add thin feature-gated adapter skeletons before adding real dependencies, then add only the smallest dependency set needed for config/tokenizer smoke parsing.
- Next priorities: Create adapter module boundaries under `src/`, wire them through the new Cargo features, and keep real model loading behind traits.

## Continuation Reflection 3

- Accomplished: Added adapter dispatch planning, asset file existence preflight, structured JSON summaries, and a combined asset summary path for model/tokenizer metadata.
- Working well: The loader path is now more than extension validation: it can check files, parse lightweight JSON, and verify adapter/format compatibility before heavy inference dependencies are involved. Local and remote checks pass with 59 tests.
- Blocking or weak spots: The project still has no real model weights reader, tokenizer implementation beyond byte-level smoke tests, or Candle/GGUF target model implementation. Dependency scope will start to matter soon.
- Approach adjustment: Keep one more layer of dependency-light preflight around load requests, then add the first real optional dependency only when a minimal end-to-end adapter smoke test needs it.
- Next priorities: Combine adapter plans, filesystem checks, and JSON summaries into a single load preflight report, then use that report to drive the first feature-gated Candle/GGUF loader shell.

Next priorities:
1. Add real model-loading adapter layer for tokenizer/config/weights paths, preferably behind optional Rust dependencies rather than disturbing the verified core.
2. Introduce Hugging Face/Candle or GGUF-backed target model implementations behind `TargetModel`.
3. Add tokenizer encode/decode boundaries.
4. Add KV-cache-aware target inference shape and batched verification abstractions.
5. Add probabilistic/speculative sampling acceptance after greedy path remains stable.
6. Add custom DFlash-style drafter loading and training/export scaffold, keeping Rust as the inference/control-plane owner.
7. Keep tests focused, run local and remote verification each implementation iteration, and update the Ralph task file with progress/reflections.