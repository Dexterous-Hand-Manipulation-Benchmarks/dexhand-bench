# dexhand-bench

Benchmarking reinforcement-learning methods on dexterous bimanual manipulation
with [DexMachina](https://github.com/MandiZhao/dexmachina) (Genesis simulator).

This repository only holds the glue: run scripts, a small shared package
(unified W&B logging, configuration) and tests. Every method and environment
lives in its own fork in the
[Dexterous-Hand-Manipulation-Benchmarks](https://github.com/Dexterous-Hand-Manipulation-Benchmarks)
organization and is included as a git submodule. Our changes sit on top of the
exact upstream commit we started from, so they can be upstreamed later.

| Path | Fork | Upstream |
|---|---|---|
| `algorithms/FlashSAC` | [FlashSAC](https://github.com/Dexterous-Hand-Manipulation-Benchmarks/FlashSAC) | [Holiday-Robot/FlashSAC](https://github.com/Holiday-Robot/FlashSAC) |
| `algorithms/FastTD3` | [FastTD3](https://github.com/Dexterous-Hand-Manipulation-Benchmarks/FastTD3) | [younggyoseo/FastTD3](https://github.com/younggyoseo/FastTD3) |
| `algorithms/reppo` | [reppo](https://github.com/Dexterous-Hand-Manipulation-Benchmarks/reppo) | [cvoelcker/reppo](https://github.com/cvoelcker/reppo) |
| `environments/dexmachina` | [dexmachina](https://github.com/Dexterous-Hand-Manipulation-Benchmarks/dexmachina) | [MandiZhao/dexmachina](https://github.com/MandiZhao/dexmachina) |
| `environments/Genesis` | [Genesis](https://github.com/Dexterous-Hand-Manipulation-Benchmarks/Genesis) | [MandiZhao/Genesis](https://github.com/MandiZhao/Genesis) (`older` branch, Genesis 0.2.1) |

dexmachina's own PPO baseline uses [MandiZhao/rl_games](https://github.com/MandiZhao/rl_games)
unmodified, so it is a pinned dependency rather than a fork.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and git.

```bash
git clone https://github.com/Dexterous-Hand-Manipulation-Benchmarks/dexhand-bench.git
cd dexhand-bench
git submodule update --init      # not --recursive: Genesis' own submodules are not needed
uv sync --extra flashsac         # or fasttd3, reppo, ppo
```

Each method is a uv *extra*. The methods pin incompatible versions (gymnasium,
torch, wandb, jax), so only one extra is installed at a time. Always pass it, and
use `--exact` so leftovers from another method are removed:

```bash
uv run --exact --extra fasttd3 python algorithms/FastTD3/fast_td3/train.py --help
```

When running from inside a submodule, add `--project <repo root>`; otherwise uv
picks up the submodule's own `pyproject.toml`. The scripts below do this for you.

The scripts keep one environment per method (`.venv-<extra>`, via
`UV_PROJECT_ENVIRONMENT`), so jobs of different methods can run at the same time
from one checkout. Each method is installed once there (a few GB each); set
`UV_PROJECT_ENVIRONMENT` yourself to put it elsewhere. Don't switch the plain
`.venv` to another extra while a job uses it.

On NixOS, `nix-shell` (or `direnv allow`) provides the system libraries; uv
still manages Python.

## Configuration

Personal settings go into `configs/local.yaml`, which is git-ignored:

```bash
cp configs/local.example.yaml configs/local.yaml
```

| Key | Meaning |
|---|---|
| `wandb.entity`, `wandb.project` | Where runs are logged. `null` entity = your W&B default. |
| `slurm.partition`, `slurm.account`, `slurm.gres`, `slurm.cpus_per_task`, `slurm.extra_args` | `sbatch` flags for cluster jobs |
| `slurm.jobs.<job>.*` | Per-job overrides (e.g. `time`, a different partition) |

`WANDB_ENTITY` / `WANDB_PROJECT` set in the environment take precedence.
Inspect the merged config with `uv run dexhand-bench config`.
Log in to W&B once with `uv run wandb login`.

## Running

| Script | What it does |
|---|---|
| `scripts/flashsac/train.sh` | FlashSAC run queue (edit `RUN_QUEUE`), then evaluates and renders checkpoints |
| `scripts/flashsac/eval_and_render.sh` | Evaluate/render all finished FlashSAC runs and attach the results to their W&B runs |
| `scripts/flashsac/render_last.sh` | Re-render the latest FlashSAC run |
| `scripts/flashsac/pre_run_test.py` | Short smoke run for every hand/clip before a long sweep |
| `scripts/fasttd3/train.sh`, `scripts/fasttd3/test_run.sh` | FastTD3 sweep / smoke test |
| `scripts/reppo/run_and_eval.sh` | REPPO sweep with evaluation, videos and ADD |
| `scripts/slurm/submit.sh <reppo\|ppo>` | Submit a cluster job with the `slurm` settings from `configs/` |
| `scripts/analysis/get_training_data.py` | Export training curves from W&B to CSV (`--group all` for every method) |

All scripts use the uv environment of their method and the W&B settings from
`configs/`, and can be started from any directory. Run variables of the SLURM
jobs pass through, e.g. `HAND=xhand SEED=1 scripts/slurm/submit.sh ppo`.

## Tests

```bash
uv run --exact --extra flashsac pytest -m "not sim"   # unit tests, ~10 s
uv run --exact --extra flashsac pytest                # + simulator tests, ~30 s (first run ~1 min: kernel compilation)
uv run --exact --extra fasttd3 pytest                 # FastTD3 integration
```

Tests for a method that is not installed are skipped, so run the suite once per
extra you touched.

- **Unit tests** cover the method wrappers (against a fake `BaseEnv`), the
  unified W&B logger and the configuration.
- **Golden tests** (`-m sim`) replay a recorded 205-step rollout on CPU through
  each method's DexMachina integration (env construction, dexmachina physics and
  rewards, wrapper outputs including the time-out at step 199) and compare every
  output. They were recorded from the code before the restructure. If one fails,
  either something changed unintentionally, or the change is intended: then check
  that it makes sense and re-record the fixture in `tests/golden/`, explaining why
  in the commit message.

## Working with the submodules

Changing code in a fork:

```bash
cd algorithms/FlashSAC
git switch main                  # submodules are checked out detached by default
# edit, test, commit
git push origin main
cd ../.. && git add algorithms/FlashSAC && git commit -m "Bump FlashSAC: <what changed>"
```

Push the submodule before the umbrella commit; otherwise others cannot check out
the referenced commit. `git config push.recurseSubmodules check` makes git refuse
such pushes.

Pulling upstream changes into a fork:

```bash
cd environments/dexmachina
git remote add upstream https://github.com/MandiZhao/dexmachina.git
git fetch upstream && git merge upstream/main   # resolve, run the tests, push
```

Adding a new method or environment: fork it into the organization, add it with
`git submodule add <fork-url> algorithms/<Name>`, add an extra (and a
`[tool.uv.sources]` path entry) to `pyproject.toml`, include the extra in
`[tool.uv].conflicts` if its dependencies clash, then run `uv lock`.

## Known issues

These were found while restructuring and are intentionally **not** changed yet,
because fixing them changes results:

- **Time-out bootstrapping differs between methods.** The provided FastTD3 scripts
  disable bootstrapping, so its handling of the terminal observation is latent there.
  FlashSAC treats every episode end on DexMachina as terminal (see below),
  so its `bootstrap_on_timeout` option has no effect.
- **The methods train on different environment configurations.** FlashSAC builds
  the env from dexmachina's argparser defaults, FastTD3 from its own
  `DexMachinaArgs`. They differ in, for example, action mode (residual vs hybrid),
  `group_collisions`, observations (410 vs 510 dims), early-reset threshold, object
  gains and the whole curriculum schedule. Only the reward weights are aligned.
- **FlashSAC maps clips by substring.** Any clip containing "box" becomes
  `box-30-230-s01-01` and any "mixer" clip becomes `mixer-0-150-s01-01`.
- **dexmachina flags every reset as both terminated and timed out**: time-outs,
  early resets and diverged (NaN) envs alike. The methods interpret these flags
  differently, so episode ends are terminal for some methods and bootstrapped for others.
- **REPPO's `debug_render.py` defaults to the old W&B team**, so
  `scripts/reppo/run_and_eval.sh` always passes an entity explicitly.

## License

MIT for the code in this repository. Each submodule keeps its upstream license
(MIT for all methods and dexmachina, Apache-2.0 for Genesis).
