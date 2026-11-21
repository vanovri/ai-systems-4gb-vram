# Changelog

All notable changes to `ai-systems-4gb-vram` are documented here.

## [1.0.0] - 2026-11-21

### Added
- Complete end-to-end implementation, training, evaluation, visual proof plots, and test suites for 10 AI/ML and Python systems engineered locally on a consumer NVIDIA GTX 1650 Ti (4 GB VRAM):
  1. `01_instagram_two_tower`: PyTorch CUDA FP16 Two-Tower RecSys with Log-Q popularity debiasing (`4.23x` Recall@10, `5.44x` NDCG@10, `-51.5%` viral bias, `0.105 ms` CPU p50, `17/17` pytest).
  2. `02_tiny_model_lora_banking77`: `Qwen3-0.6B` PEFT LoRA fine-tuning on Banking77 (`r=8`, `0.192%` trainable params, `1,305 MiB` peak VRAM, `84.59%` full 3,076 test Exact-Match & Macro-F1, `89.34%` Top-3, `15/15` pytest).
  3. `03_agent_harness`: Minimal pure-Python coding agent harness with 4 sandboxed tools, path-jail, and JSONL traces (`3/3` repo bugs solved, `17/17` pytest).
  4. `04_quantize_your_own_llm`: Pure NumPy group-wise INT8/INT4 LLM quantizer with real 4-bit nibble packing and GPTQ-lite Hessian compensation (`-67.70%` Layer-10 L2 error, WikiText-2 PPL `34.36 -> 34.47 INT8 -> 43.59 INT4 Sym`, cross-project LoRA INT4 `76.10%` Exact-Match in `236 MiB` linear, `16/16` pytest).
  5. `05_neural_pong_world_model`: `1.58M`-param Action-Conditioned U-Net World Model trained on `140k+30k` frames + 8-step DAgger with zero game engine imported at runtime (`101.2 FPS`, `26.6 MiB` VRAM, `0.9401` Dynamic IoU, `0.286 px` ball error, `100.0%` `300/300` counterfactual paddle ordering, `17/17` pytest).
  6. `06_personal_rag_assistant`: Hybrid FAISS `IndexFlatIP` + BM25 + RRF RAG assistant with inline citations and OOS refusal (`Recall@3 = 1.00`, `MRR = 1.00`, `Refusal = 1.00`, `15/15` pytest).
  7. `07_ai_research_agent`: Pluggable search + HTML clean dedup + cited Markdown brief + SQLite TTL cache (`14/14` pytest).
  8. `08_discord_bot`: Modular `discord.py` Cogs bot with Open-Meteo & GitHub embeds, SQLite WAL reminders, and `Dockerfile` (`14/14` pytest).
  9. `09_twenty_one_game`: Shared-counter 21 game with exhaustive `21/21` minimax proof and `100.0%` (`1,000/1,000`) optimal win rate (`14/14` pytest).
  10. `10_multi_agent_system`: Pure-Python CrewAI-style sequential & hierarchical multi-agent system with critic revision loop (`0.7 -> 1.0`, `12/12` pytest).
- Unified `verify_all.ps1` verification sweep (`151/151` pytest PASS), GitHub Actions CPU CI workflow, and pre-push hygiene scanner.
