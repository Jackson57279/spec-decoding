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

Next priorities:
1. Add real model-loading adapter layer for tokenizer/config/weights paths, preferably behind optional Rust dependencies rather than disturbing the verified core.
2. Introduce Hugging Face/Candle or GGUF-backed target model implementations behind `TargetModel`.
3. Add tokenizer encode/decode boundaries.
4. Add KV-cache-aware target inference shape and batched verification abstractions.
5. Add probabilistic/speculative sampling acceptance after greedy path remains stable.
6. Add custom DFlash-style drafter loading and training/export scaffold, keeping Rust as the inference/control-plane owner.
7. Keep tests focused, run local and remote verification each implementation iteration, and update the Ralph task file with progress/reflections.