# Assessment-Aware Study Planner

An Intelligent Systems course project combining a trained neural network,
Uniform Cost / Greedy / A* search, and Backtracking / Forward Checking / AC-3.
All algorithms are implemented explicitly in Python. No search or CSP framework
is used. The neural network uses NumPy and includes its forward pass and training.

**Decision support only:** predicted marks are observational what-if estimates.
Adding clicks is a modelling assumption, not evidence that study sessions cause
a particular improvement. This model has substantial prediction error.

## Quick start (Windows, macOS or Linux)

Install Python 3.12. Extract the ZIP, open a terminal in this directory, then:

```bash
python -m venv .venv
```

Activate on Windows PowerShell with `.venv\Scripts\Activate.ps1`, on Windows
Command Prompt with `.venv\Scripts\activate.bat`, or on macOS/Linux with
`source .venv/bin/activate`. Then:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_demo.py
```

The package includes processed real OULAD data, trained weights and measured
results. No download or training is needed for the demo. Output:
`outputs/demo/result.json` and `outputs/demo/cooperation.png`.

## Reproduce from the original dataset

```bash
python data/make_data.py
python nn/train.py
python -m eval.profiles
python -m eval.experiments
python run_demo.py --trace --rejection-demo
```

`make_data.py` downloads the public 46,750,706-byte Figshare ZIP, verifies its
pinned MD5, extracts four named CSVs and processes clicks in chunks. No account
or registration was required in the verified download. Allow roughly 0.7 GB
disk space for raw files. A network failure should be retried later; no fake
data is silently substituted. If you already have the CSVs, use
`python data/make_data.py --raw-dir PATH --local-only`.

Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before Python if your machine
uses many BLAS threads. The provided results used one thread. Small floating
point differences can occur across platforms. The dataset seed is 42, the main
model seed is 42, and the experiment seeds are 0, 1, 2, 3, 4. Seeds control
student validation splits, initialization, minibatches and generated cases.
Experiments cache fitted models under their output directory. To intentionally
retrain with different code/data/settings, use a fresh `--output-dir`.

## Live demonstration commands

```bash
python run_demo.py --trace --rejection-demo
python run_demo.py --predictor baseline --output-dir outputs/demo_baseline
python run_demo.py --algorithm greedy --output-dir outputs/demo_greedy
python run_demo.py --algorithm astar --output-dir outputs/demo_astar
python run_demo.py --csp bt --output-dir outputs/demo_bt
python run_demo.py --csp fc --output-dir outputs/demo_fc
python run_demo.py --target 100 --capacity 0 --output-dir outputs/demo_impossible
```

The live-demo default is profile index 5, target 60, three weekly slots, UCS, FC
and AC-3. The batch of 30 profiles retains the original target 55.
Choose `--profile 0` through `--profile 29`, `--target`, `--capacity`, or
`--max-sessions` as required. Read the JSON traces for full details; terminal
trace output is deliberately limited to 30 events. The stored trace caps are
250 search states and 150 CSP events, but counters cover the whole computation.

## Code map

| Module | Responsibility | Start reading here |
|---|---|---|
| `data` | Download, chunk clicks, chronological split, common-date profiles | `features.py`, `make_data.py` |
| `nn` | 19-input, 24-hidden-unit NN and average-mark baseline | `model.py`, `train.py` |
| `search` | Unit-cost allocation graph and three priority rules | `core.py` |
| `csp` | Binary constraints, recursive placement and independent pruning | `problem.py`, `backtracking.py`, `forward_checking.py`, `ac3.py` |
| `integration` | Prediction curves, search-to-CSP callback and feasible fallback | `planner.py` |
| `eval` | Five-seed experiments, plots and 30 profile results | `experiments.py`, `profiles.py` |
| `tests` | Normal, edge, impossible and exhaustive-reference checks | `test_algorithms.py`, `test_data.py` |

## Modelling choices you must understand

- Course BBB, October presentations only: train 2013J, test 2014J. Remove any
  held-out student from training. Validation holds out entire training students.
- There are **19 inputs**, not exactly 12: twelve weekly activity bins plus
  seven history/context features. The brief says approximately 12. We preserve
  all four weeks and three groups explicitly instead of losing information.
- Forecasts use clicks in the **28 days before the decision date**, rather than
  future clicks immediately before the deadline. Training horizons are 14, 28,
  42, 70 and 112 days. This deliberate adaptation prevents future-information
  leakage and covers the demonstration's 14/70/112-day horizons.
- An earlier mark becomes available 14 days after the later of its deadline
  and submission date. OULAD does not record grade-release dates; this is an
  explicit assumption. Future marks never enter a planning profile.
- A session adds 40 content clicks, spread equally over four bins. Weekly
  capacity (default 3), target (55), and the eight-session-per-assignment bound
  are generated scenario settings. They are not measured student behaviour.
- Positive click-input and output weights ensure that increasing clicks cannot
  decrease the predicted mark. This mathematical constraint is not proof of
  real-world benefit. Training minimizes squared error with projected Adam.
- CSP slots are abstract ordered positions within complete weeks. All sessions
  are available from week 1, one per slot, and same-assignment sessions strictly
  increase in slot order. Deadline weeks use conservative floor division.
  For an actual calendar interpretation, place slots before the end of each
  complete week. This version does not model task release dates or holidays.
- **UCS is optimal within the configured finite session cap.** A* uses a sum of
  per-assignment minimum remaining sessions, which ignores scheduling. Greedy
  uses the same estimate without path cost and has no general optimality claim.
- Because marks depend independently on each assignment's effort and schedule
  constraints only become tighter with added sessions, this minimum version
  has a separable allocation structure. Do not claim Greedy must produce worse
  costs. Some runs legitimately have equal costs or zero-session solutions.
- The NN currently depends on session **count**, not the exact timetable week.
  Search checks marks, then CSP validates timing. The model's 40-click rule
  changes a hypothetical lookback snapshot; it is not a forecast of daily clicks.
- On failure, the fallback enumerates all bounded allocations and returns a
  jointly schedulable plan minimizing summed target shortfall, then session
  count. “No solution” means unreachable **within this session cap and model**.
  A stopped search reports “feasibility unknown” instead of impossibility.

## Results and submission files

`models/metrics.json` contains held-out error and monotonicity checks.
`outputs/experiments/*_raw.csv` contains individual runs. `*_per_seed.csv`
averages cases within each seed; `*_summary.csv` reports the mean and sample
standard deviation of those five seed means. Missing solution costs are NaN,
never zero. Cost comparisons are conditional on success and must be read next
to success rates. Experiment 2 uses clearly labelled generated stress cases.
Experiment 1 uses generated attainable per-assignment thresholds from each
model's predictions at 1..4 sessions to exercise nontrivial search. Its targets
are constructed benchmarks, not educational evidence. Experiment 3 uses the
fixed target choices above. Experiment 3's plan success is measured against each predictor's own estimates,
not against real post-intervention marks. It is not causal validation.

`outputs/profiles/` contains a timetable and predicted marks for every one of
30 real profiles. `docs/` contains the report, presentation, data dictionary,
discussion guide, live-demo script, checklist and contribution template.
Fill in truthful team contributions before submission. The supplied git history
records the actual AI-assisted build, not fictitious student activity.

## Sources and attribution

OULAD: Kuzilek, J., Hlosta, M., and Zdrahal, Z. (2017). *Open University Learning
Analytics dataset*. Scientific Data 4, 170171. DOI: 10.1038/sdata.2017.171.

- Dataset record: https://doi.org/10.6084/m9.figshare.5081998
- Current schema: https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/
- Public archive: https://ndownloader.figshare.com/files/8606371
- Data licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

Processed subsets and example profiles are derived from OULAD. Modifications:
course/presentation selection, filtering, temporal aggregation, engineered
features, splits and generated planning parameters. Demographic tables are
not read. See `docs/data_dictionary.md` for every consumed and generated field.
