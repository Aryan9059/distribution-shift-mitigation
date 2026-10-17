# Project context: Detecting and adapting to distribution shift (tabular)

Course ML project, team of 4 (beginners in ML, plenty of time, laptop/CPU is fine).
Title: "When ML Models Go Wrong: Detecting and Adapting to Distribution Shift in Real-World Data".
Baseline paper: "Failing Loudly" (Rabanser, Gunnemann, Lipton, NeurIPS 2019), reimplemented on TABULAR data.
Other papers: BBSE (Lipton et al. 2018), C2ST (Lopez-Paz & Oquab 2017), importance weighting (Shimodaira 2000),
KMM (Huang et al. 2007), CORAL (Sun et al. 2016), gradual self-training (Kumar et al. 2020).
Final deliverable: report, slides, GitHub repo, and a Streamlit app (precompute results, app only reads files).

## How I want you to work with me
- I am a beginner. Explain things in simple terms, and say WHY, not just what.
- Do not stick rigidly to the original proposal; do what is best for a strong project.
- Build step by step, and run each step's sanity check before moving on.
- Do not cut scope: we have lots of time.

## Overall story
Models fail silently under shift. (1) Which detectors catch it? (2) Most detected shifts are harmless, so
alarm only when accuracy really drops (malignancy check). (3) Does the shift TYPE tell us how to fix it
(detector-guided adaptation policy)? That policy is the novel contribution.

## Data
- Adult (OpenML "adult" v2): simulated shifts. Drop `fnlwgt`. y = income > 50K.
- Covertype (sklearn fetch_covtype, subsample ~40k): simulated shifts. Labels shifted to 0..6 (XGBoost needs 0-based).
- Folktables ACS Income: REAL shift. Train on one state (CA), test on others (WY, MS, NY, TX), year 2018.
  Treat ACS codes (COW, MAR, OCCP, POBP, RELP, SEX, RAC1P) as categorical strings.
- NOTE: WILDS is not tabular; skip it (or use only with frozen embeddings). Use Folktables/TableShift instead.
- Tilt columns for covariate shift: adult [age, education-num]; covertype [Elevation, Horizontal_Distance_To_Roadways];
  acs [AGEP, WKHP] (verify with X.columns).

## Models (all via sklearn Pipeline = preprocessing + model)
LR, RandomForest(200), XGBoost GBT(200, depth 4, lr 0.1), MLP(64,32, early stopping).
Preprocessing: StandardScaler on numeric, OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50) on categorical.
FIT ONLY ON SOURCE-TRAIN (never on target).

## Data split design (no leakage)
source -> source-train (train models) + source-test (normal accuracy)
target -> pool A (shifted -> target-train, for the upper bound) + pool B (shifted -> target-test, all models scored here)
For Adult/Covertype, source and target are two random stratified halves; the shift is applied on top of the target.
For ACS, source = one state, target = another state (shift is real, no severity dial).

## Shift simulator (src/shifts.py), all with signature (X, y, rng, n, ...)
- identity: no-shift control (REQUIRED for the false-alarm check in Phase 2)
- covariate: exponential tilting of sampling probability on chosen columns; severity s in {0.5,1,1.5,2,3}
- noise: Gaussian noise std = s * col std on fraction delta of rows; s in {0.25,0.5,1,2}
- drift: mean shift of s std on fraction delta of rows; s in {0.25,0.5,1,2}
- label: class priors q = (1-s)*p + s/K; s in {0.25,0.5,0.75,1.0}; sample with replace=True (rare classes)
- concept: rows above median of a column get label changed to another class with prob s; s in {0.1,0.2,0.4}
- combo: covariate then label shift
Use a separate rng per (seed, scenario index): np.random.default_rng((seed, i)). Phase 2 reuses these scenarios exactly.
Hand-check each function: covariate raises mean age, label s=1 gives balanced classes, concept changes labels in the subgroup.

## Phase 1 (current task)
Files: src/data.py, src/models.py, src/shifts.py, experiments/phase1.py, experiments/plot1.py, results/.
Run from project root: python -m experiments.phase1 (use --debug and only "adult" while developing).
Datasets for the full run: adult, covertype, acs:CA>WY, acs:CA>MS, acs:CA>NY, acs:CA>TX; 5 seeds; run overnight.
Output: results/phase1.csv with one row per (dataset, seed, shift, severity, model) and columns
acc_source_test, acc_base, f1_base (macro), acc_upper, f1_upper.
- baseline 1 = source model scored on target-test; baseline 2 (upper bound) = clone(model) retrained on target-train, scored on same target-test.
Figures: accuracy vs severity per (dataset, shift), solid = source model, dashed = upper bound, error bars over seeds;
ACS = grouped bars (acc_source_test, acc_base, acc_upper) per target state.
Sanity checks: severity 0 -> acc_base ~ acc_source_test ~ acc_upper (Adult ~0.85); accuracy mostly falls with severity;
label shift can move accuracy either way (also check macro F1); concept drift hurts all models and the upper bound too;
mild covariate shifts often barely hurt (a real finding: benign shift); ACS source model should lose accuracy on some states, not all.
Common bugs: fitting preprocessing on target; same row in train and test; XGBoost labels not 0-based; mixing seeds.
Done when: phase1.csv, figures, and a short note on which shifts hurt which models and the source-to-upper-bound gap.

## Later phases (plan)
Phase 2 Detection (reuse Phase 1 scenarios as ground truth; alpha=0.05; n in {10,20,50,100,200,500,1000,5000}; >=5 seeds):
 representations: raw, PCA, BBSDs (predict_proba), BBSDh (predict); tests: KS+Bonferroni, chi-squared (categorical, BBSDh),
 PSI, MMD (optional), C2ST (cross-fitted, binomial test). Metrics: precision/recall of alarms, false-positive rate on no-shift (~5%).
 Malignancy check: alarm only if shift detected AND accuracy drop > tau. Estimate drop with a small labeled budget (50-200)
 and with a label-free confidence-based estimate (validate: estimated vs true drop scatter).
 BBSE shift typing: w = C^-1 mu_target (check C well-conditioned); label shift if reweighted source predictions match target.
Phase 3 Adaptation: importance weighting (domain classifier, clip weights, log effective sample size), KMM, BBSE label reweighting,
 CORAL (numeric block only), self-training (needs a GRADUAL sequence of target domains).
 Adaptation gain = (acc_adapt - acc_base)/(acc_upper - acc_base); only report when the gap is above a minimum, show absolute acc too.
Phase 4 Detector-guided policy: label shift -> BBSE weights; covariate + good overlap -> importance weighting;
 covariate + poor overlap -> CORAL / self-training; no alarm or benign -> do nothing. Compare vs no handling, best fixed method, oracle.
Phase 5 Analysis/ablations (tau, label budget, weight clipping, seeds), failure cases.
Website: Streamlit pages = Overview, Detection demo, Adaptation demo, Results explorer, Reproduce.

## Team (from proposal)
Rehan Farishta (data + shift simulation), Aryan Srivastava (detection, BBSE), Aditya Tiwari (adaptation),
Shivam Mishra (baselines, evaluation, error analysis). Everyone writes the report.

## Setup
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn xgboost matplotlib folktables jupyter
