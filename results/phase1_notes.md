# Phase 1 results: how much do shifts hurt, and how far are we from the upper bound?

Setup: 4 models (LR, RF, GBT, MLP), 6 datasets, 5 seeds, data in `results/phase1.csv`
(1000 rows), figures `results/phase1_<dataset>.png` and `results/phase1_acs.png`.
All numbers are accuracy on target-test, averaged over 5 seeds.

- **Baseline 1** = the source model, scored on target-test.
- **Baseline 2 ("upper bound")** = the same model type retrained on target-train, scored on the same target-test.
- **Gap** = upper bound minus baseline 1. It is the most that adapting could recover.

## Sanity checks

| Check | Result |
|---|---|
| No shift: base ≈ source-test ≈ upper | Adult: 0.85-0.87 base (best model GBT 0.871), gap -0.001. Covertype: 0.72-0.81, gap -0.006. OK. |
| Accuracy mostly falls with severity | Yes for noise, drift, label, concept, combo on both datasets. **Exception: Covertype covariate (see below).** |
| Label shift moves accuracy, check macro F1 | Accuracy falls (Adult GBT 0.871 → 0.787 at s=1). F1 falls less (0.809 → 0.782) because balanced classes make the rare class matter more. |
| Concept drift hurts the upper bound too | Yes. Adult GBT 0.840 → 0.748 as s goes 0.1 → 0.4, and the retrained model cannot fix it (gap is negative). |
| Mild covariate shift is benign | Yes on Adult: GBT loses only 2.7 points at s=0.5 and the gap is about -0.007, so retraining does not help. |
| ACS loses accuracy on some states, not all | Loses on all four, but by very different amounts (below). |

## Which shifts hurt which models (Adult / Covertype)

- **Drift (mean shift on half the rows) is the most damaging.** At s=2 every model collapses: Adult 0.54-0.56 (from about 0.85-0.87), Covertype 0.46-0.62. At s≥1 the gap is large (Adult 0.15 to 0.29, Covertype 0.13 to 0.21). This is the clearest case where adapting to the target can pay off.
- **Noise** hurts steadily (Adult GBT 0.871 → 0.760 at s=2). The gap grows with severity, up to about 0.07 on Adult.
- **Label shift** hurts gradually. On Covertype the gap reaches 0.14 at s=1, so label reweighting has room to help. On Adult the gap is small (0.045).
- **Concept drift** hurts every model equally, and the gap is slightly negative. Retraining on target-train does not help, so no covariate or label correction can fix it.
- **Covariate shift** is mostly mild on Adult (0.871 → 0.787 for GBT at s=2) and the gap is about zero or negative. LR is the most fragile at s=3 (0.678 vs 0.74+ for the others).
- **Model ranking:** on Adult, GBT is best under almost every condition. On Covertype, RF is best (0.811 base, and it stays the best across shifts). MLP is the most fragile on Covertype drift (0.475 at s=1 vs RF 0.704).

## Surprises and caveats (please read before Phase 2/3)

1. **Covertype covariate shift makes accuracy go UP** (GBT 0.772 → 0.884 at s=3). Tilting toward high elevation / long roadway distance concentrates the sample on a few cover types, so the task becomes easier (fewer classes in play). The shift is real, but it is not harmful. This is the benign-shift finding, and a good example for Phase 2: a detector will alarm, accuracy will not drop, and the malignancy check should stay quiet.
2. **The upper bound is not always above baseline 1.** For Adult covariate, concept and combo the gap is -0.005 to -0.022. The target-train set (8,000 rows, many repeated by the tilting) carries less information than the source-train set, so retraining on it does not beat the source model. Phase 3's adaptation gain divides by (acc_upper - acc_base), so it must only be reported when the gap is above a minimum (as planned), and absolute accuracy should be shown too.
3. **Adult covariate s=3 is extreme** (mean age about 74, from 39). Treat that point as a stress test, not a realistic shift.
4. **Macro F1 and accuracy disagree on concept drift.** Adult GBT accuracy 0.748 but F1 0.661 at s=0.4, because the changed labels hit the minority class.

## Real shift (ACS Income, trained on CA)

| Target | Source-test | Base (mean over models) | Upper | Gap | Reading |
|---|---|---|---|---|---|
| MS | 0.81 | 0.74 | 0.80 | **+0.06** | Biggest real shift, mostly recoverable. A good adaptation target. |
| WY | 0.81 | 0.75 | 0.75 | ~0 | Accuracy drops 6 points, but retraining on WY does not help. Only about 1,500 target-train rows, so the upper bound is weak, not the shift harmless. |
| NY | 0.81 | 0.78 | 0.78 | ~0 | Small drop (2-3 points), benign. |
| TX | 0.81 | 0.79 | 0.79 | +0.01 | Smallest drop (2 points). Close to benign. |

All four states lose some accuracy, so "should lose on some states, not all" holds in the weaker sense that NY and TX are nearly benign. The model ranking barely matters on ACS (models are within about 1 point of each other); the state matters far more than the model. Seed variation is small (std about 0.004-0.013), but note that the 40k-row CA/state subsample is fixed (seed 0), so the error bars understate the real uncertainty.

## Takeaways for the next phases

- Not all shifts need fixing: Covertype covariate and mild Adult covariate hurt little or even help. This supports the "alarm only when accuracy drops" idea (Phase 2).
- Shift type matters: drift, noise and label shift leave a large gap that adaptation could close; concept drift leaves none. This supports the detector-guided policy (Phase 4).
- MS is the best real-shift test case for adaptation; WY and NY are the best "do nothing" cases.
