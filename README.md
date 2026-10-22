# distribution-shift-mitigation

Detecting, diagnosing and adapting to distribution shift in tabular data.

1. **Shift simulator** (`src/shifts.py`) injects covariate, noise, drift, label, concept and combined shifts at
   several severities into Adult and Covertype. ACS Income (train on California, test on WY/MS/NY/TX) gives a real shift.
2. **Base models** (`src/models.py`): LR, RF, XGBoost (GBT), MLP. We measure how much each shift hurts them.
3. **Base paper** (`src/detect.py`): *Failing Loudly* (Rabanser et al., NeurIPS 2019) reimplemented for tabular
   data: NoRed / PCA / SRP / BBSDs / BBSDh / domain-classifier representations with KS+Bonferroni, MMD,
   chi-squared and binomial tests, plus the paper's labeled harm check.
4. **Our model, D2A** (`src/adapt.py`): Detect (BBSDs+KS) → Harmful? (ATC, no labels) → Diagnose (BBSE + C2ST:
   label vs covariate shift) → Adapt (BBSE prior correction / importance-weighted retraining / CORAL), or keep the
   model when the shift is harmless.

## Run

```
python -m venv .venv && .venv\Scripts\activate
pip install numpy pandas scikit-learn xgboost matplotlib scipy folktables

python -m experiments.run_all --quick        # adult, 1 seed (minutes)
python -m experiments.run_all                # everything, 3 seeds (about an hour on a laptop)
python -m experiments.run_all --full         # 5 seeds, 8000-row pools, n up to 5000 (several hours)
python -m experiments.make_figures           # report/figures/* and results/summary.md
# report: upload the report/ folder to Overleaf (or any LaTeX install) and compile main.tex
```

## Layout

```
src/data.py            dataset loaders, leak-free source/target split
src/shifts.py          shift simulator
src/models.py          the four base models (preprocessing inside the pipeline)
src/detect.py          Failing Loudly detectors (base paper)
src/adapt.py           D2A (ours) and its building blocks
experiments/run_all.py one loop: simulate -> impact -> detection -> malignancy -> adaptation
experiments/make_figures.py  figures, LaTeX tables, results/summary.md
results/*.csv          raw results (one file per table and dataset); results/dev_seed0/ = development run
results/summary.md     every number quoted in the report
report/                IEEE two-column report (main.tex, refs.bib, figures/)
```
