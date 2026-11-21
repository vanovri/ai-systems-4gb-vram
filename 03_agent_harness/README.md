# Project #3 - Build Your Own Agent Harness (`03_agent_harness`)

Minimal, production-grade coding agent harness built around the formal bounded control loop:

$$m_{t+1} = m_t \oplus (a_t, o_t)$$

The loop stops cleanly when $a_t \notin \mathcal{T}$ (final natural-language answer or unknown tool), when `step >= max_steps` ($T_{\max}$), when doom-loop detection trips on repeated identical calls, or when cumulative tokens exceed `max_token_budget`.

## Architecture

- `harness/agent.py` - `AgentHarness` bounded control loop with doom-loop detection, token budget guard, and formal state transitions.
- `harness/tools.py` - 4 sandboxed core tools (`list_files`, `read_file`, `edit_file` with unified diff, `run_tests` with allowlist & timeout) behind strict `WorkspaceSandbox` path-jail enforcement.
- `harness/llm.py` - `OpenAICompatibleProvider` (works with OpenAI, local `9Router` at `http://127.0.0.1:20128/v1`, or Gemini OpenAI-compat via `OPENAI_BASE_URL`/`OPENAI_API_KEY`/`OPENAI_MODEL`), `ScriptedTestProvider` (deterministic CI replay), and `HeuristicOfflineProvider` (offline observation-driven bugfix demo).
- `harness/trace.py` - JSONL trace logger recording `tool_call`, `tool_result`, `duration_ms`, unified `diff`, and token usage per step.
- `harness/eval.py` & `benchmarks/` - 3 real buggy benchmark mini-repos evaluated in isolated temp copies:
  1. `checkout_cart` - percentage discount bug (`return total - percent` -> `total * (1 - percent / 100)`)
  2. `rate_limiter` - sliding-window off-by-one capacity bug (`>` -> `>=`)
  3. `csv_parser` - RFC-4180 quoted-comma & escaped-quote parsing bug

## Quickstart

```bash
# Run offline benchmark evaluation (all 3 buggy repos -> fail -> agent fixes -> pass)
python -m harness.cli eval --provider offline --out-dir traces

# Run against any target workspace with an OpenAI-compatible endpoint
# (set OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL in environment - never hardcode keys)
python -m harness.cli run --workspace path/to/repo --task "Fix failing unit tests" --provider openai --trace traces/live.jsonl

# Run unit + integration tests
pytest -v
```
