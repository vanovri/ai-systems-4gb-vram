# Project #9 - 21 Number Game (`09_twenty_one_game`)

Classic **21 Bachet-Nim** game built with an immutable pure state-machine engine, three AI difficulty levels (`easy`, `smart`, `unbeatable` modulo-4), an exhaustive game-tree minimax mathematical proof, an interactive ASCII TUI + non-interactive demo CLI, and a 1,000-match simulation benchmark.

## Bug Fixed vs. Original Slide

Many tutorial implementations (including the reference slide) mistakenly maintain **two separate counters** (`player += p`, `computer += rand`), breaking the actual rules of 21.

This implementation enforces the **true 21 Bachet-Nim rules**:
- Both players advance **one single shared counter** of consecutive integers from `1` to `21` (starting at `0`).
- On each turn, the active player must add `1`, `2`, or `3` consecutive numbers (`1 <= count <= min(3, 21 - current)`).
- **`misere` mode (classic 21, default)**: whoever is forced to say **`21` LOSES**.
- **`normal` mode (variant)**: whoever says **`21` WINS**.

## Mathematical Optimality (Modulo-4 Strategy & Minimax Proof)

In classic misère 21 (`target = 21`, `max_step = 3`, cycle `M = max_step + 1 = 4`):
- Safe **P-positions** (states that guarantee a win for the player who just moved, leaving the next player trapped) are all multiples of 4: **`0, 4, 8, 12, 16, 20`**.
- Whenever `current % 4 != 0`, playing `step = (20 - current) % 4` (`= 4 - (current % 4)`) always lands on the next multiple of 4 (`4 -> 8 -> 12 -> 16 -> 20`), forcing the opponent onto `21`.
- Because `0` is already a multiple of 4, **going second** with `unbeatable` AI mathematically guarantees a **100.0% win rate** against any opponent policy!
- In `normal` mode (taking `21` wins), safe P-positions are `1, 5, 9, 13, 17` (`(21 - current) % 4 == 0`), so **going first** (taking `1` initially, then complementing opponent moves to `4`) guarantees a **100.0% win rate**.

Our exhaustive memoized minimax solver (`game/ai.py::verify_unbeatable_against_minimax`) verifies **all 21/21 states** in both `misere` and `normal` modes with **0 mismatches**.

## Verified 1,000-Match Simulation Results (`reports/simulation_1000.json`)

| Scenario | Mode | First Player | AI Difficulty | Opponent | Matches | AI Wins | Win Rate | Avg Turns |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| Unbeatable (2nd) vs Random | `misere` | `human` | `unbeatable` | `random` | 1,000 | 1,000 | **100.0%** | 11.00 |
| Unbeatable (2nd) vs Greedy | `misere` | `human` | `unbeatable` | `greedy` | 1,000 | 1,000 | **100.0%** | 11.00 |
| Unbeatable (2nd) vs Mixed | `misere` | `human` | `unbeatable` | `mixed` | 1,000 | 1,000 | **100.0%** | 11.00 |
| Unbeatable (1st) vs Mixed | `normal` | `ai` | `unbeatable` | `mixed` | 1,000 | 1,000 | **100.0%** | 11.00 |
| Smart (2nd) vs Random | `misere` | `human` | `smart` | `random` | 1,000 | 954 | 95.4% | 11.21 |
| Easy (2nd) vs Random | `misere` | `human` | `easy` | `random` | 1,000 | 530 | 53.0% | 10.85 |

## Project Structure

```text
09_twenty_one_game/
├── game/
│   ├── __init__.py
│   ├── engine.py      # Pure immutable GameState & GameEngine state machine
│   ├── ai.py          # easy / smart / unbeatable modulo-4 AI + exhaustive minimax solver
│   ├── simulation.py  # 1,000-match simulator & benchmark report generator
│   └── cli.py         # Interactive TUI, ASCII progress bar, --auto-demo, --simulate
├── reports/
│   └── simulation_1000.json
├── tests/
│   └── test_game.py   # 14 pytest tests (rules, minimax proof, 1,000-match 100.0% proof)
├── pytest.ini
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install test dependency (game engine uses Python stdlib only)
pip install -r requirements.txt

# Play interactively in terminal
python -m game.cli --mode misere --difficulty unbeatable --first human

# Run non-interactive showcase match with ASCII progress bar & replay log
python -m game.cli --auto-demo

# Run 1,000-match simulation + minimax proof and write JSON report
python -m game.cli --simulate 1000 --output reports/simulation_1000.json

# Run full pytest suite (14 tests)
pytest -v
```
