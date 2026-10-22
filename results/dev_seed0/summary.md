## Impact at the strongest severity (mean over seeds)

                           acc_source_test  acc_base  acc_oracle
dataset   shift     model                                       
adult     combo     GBT              0.865     0.805       0.750
                    LR               0.842     0.748       0.732
                    MLP              0.848     0.762       0.738
                    RF               0.844     0.755       0.746
          concept   GBT              0.865     0.748       0.724
                    LR               0.842     0.745       0.732
                    MLP              0.848     0.741       0.719
                    RF               0.844     0.730       0.714
          covariate GBT              0.865     0.792       0.646
                    LR               0.842     0.637       0.616
                    MLP              0.848     0.710       0.644
                    RF               0.844     0.694       0.621
          drift     GBT              0.865     0.556       0.858
                    LR               0.842     0.545       0.845
                    MLP              0.848     0.546       0.846
                    RF               0.844     0.532       0.834
          label     GBT              0.865     0.798       0.839
                    LR               0.842     0.779       0.827
                    MLP              0.848     0.814       0.820
                    RF               0.844     0.765       0.822
          noise     GBT              0.865     0.763       0.837
                    LR               0.842     0.771       0.841
                    MLP              0.848     0.769       0.834
                    RF               0.844     0.750       0.839
covertype combo     GBT              0.771     0.701       0.690
                    LR               0.723     0.655       0.639
                    MLP              0.760     0.643       0.666
                    RF               0.815     0.708       0.679
          concept   GBT              0.771     0.624       0.589
                    LR               0.723     0.592       0.574
                    MLP              0.760     0.620       0.571
                    RF               0.815     0.662       0.608
          covariate GBT              0.771     0.870       0.906
                    LR               0.723     0.811       0.852
                    MLP              0.760     0.828       0.862
                    RF               0.815     0.915       0.911
          drift     GBT              0.771     0.545       0.739
                    LR               0.723     0.540       0.717
                    MLP              0.760     0.401       0.712
                    RF               0.815     0.611       0.759
          label     GBT              0.771     0.608       0.705
                    LR               0.723     0.498       0.690
                    MLP              0.760     0.579       0.702
                    RF               0.815     0.621       0.723
          noise     GBT              0.771     0.583       0.631
                    LR               0.723     0.519       0.587
                    MLP              0.760     0.602       0.594
                    RF               0.815     0.652       0.651

## No shift (identity)

                 acc_source_test  acc_base  acc_oracle
dataset   model                                       
adult     GBT              0.865     0.875       0.869
          LR               0.842     0.854       0.855
          MLP              0.848     0.852       0.860
          RF               0.844     0.846       0.845
covertype GBT              0.771     0.774       0.750
          LR               0.723     0.731       0.723
          MLP              0.760     0.764       0.737
          RF               0.815     0.820       0.773

## ACS real shift

                 acc_source_test  acc_base  acc_oracle
dataset   model                                       
acs:CA>MS GBT              0.810     0.742       0.807
          LR               0.810     0.741       0.798
          MLP              0.813     0.740       0.806
          RF               0.801     0.745       0.791
acs:CA>NY GBT              0.810     0.786       0.780
          LR               0.810     0.784       0.776
          MLP              0.813     0.787       0.784
          RF               0.801     0.782       0.769
acs:CA>TX GBT              0.810     0.790       0.794
          LR               0.810     0.779       0.787
          MLP              0.813     0.786       0.783
          RF               0.801     0.786       0.777
acs:CA>WY GBT              0.810     0.760       0.754
          LR               0.810     0.748       0.755
          MLP              0.813     0.747       0.751
          RF               0.801     0.747       0.755

## Detection rate over all shifted scenarios (Adult + Covertype)

n                    10     20     50     100    200    500    1000
NoRed (KS)          0.318  0.409  0.659  0.705  0.750  0.818  0.886
PCA (KS)            0.341  0.341  0.568  0.659  0.750  0.818  0.864
SRP (KS)            0.341  0.341  0.614  0.705  0.773  0.818  0.841
BBSDs (KS)          0.386  0.341  0.523  0.545  0.591  0.773  0.773
BBSDh (Chi2)        0.386  0.409  0.432  0.500  0.636  0.636  0.682
Classif (Binomial)  0.000  0.000  0.364  0.659  0.727  0.773  0.818
NoRed (MMD)         0.341  0.341  0.568  0.659  0.682  0.773  0.818
PCA (MMD)           0.227  0.364  0.568  0.591  0.705  0.750  0.773
SRP (MMD)           0.182  0.295  0.500  0.705  0.659  0.773  0.818
BBSDs (MMD)         0.136  0.227  0.341  0.455  0.568  0.659  0.750

## False-alarm rate on the no-shift control

n                   10    20    50    100   200   500   1000
NoRed (KS)           0.0   0.0   0.0   0.0   0.0   0.0   0.0
PCA (KS)             0.0   0.0   0.0   0.0   0.5   0.0   0.0
SRP (KS)             0.0   0.5   0.0   0.0   0.0   0.0   0.0
BBSDs (KS)           0.0   0.0   0.0   0.0   0.0   0.5   0.5
BBSDh (Chi2)         0.0   0.0   0.5   0.0   0.0   0.0   0.0
Classif (Binomial)   0.0   0.0   0.0   0.0   0.0   0.0   0.0
NoRed (MMD)          0.0   0.0   0.0   0.0   0.0   0.0   0.0
PCA (MMD)            0.5   0.5   0.0   0.0   0.0   0.0   0.0
SRP (MMD)            0.5   0.0   0.0   0.0   0.5   0.5   0.0
BBSDs (MMD)          0.0   0.0   0.0   0.0   0.5   0.0   0.0

## BBSDs+KS detection rate per shift (identity = false alarms)

n           10     20     50     100    200    500    1000
shift                                                     
identity   0.000  0.000  0.000  0.000  0.000  0.500  0.500
covariate  0.800  0.700  0.800  1.000  1.000  1.000  1.000
noise      0.125  0.000  0.250  0.125  0.125  0.625  0.625
drift      0.250  0.250  0.625  0.500  0.625  0.875  0.875
label      0.250  0.375  0.375  0.625  0.750  1.000  1.000
concept    0.167  0.000  0.167  0.000  0.000  0.000  0.000
combo      0.750  0.750  1.000  1.000  1.000  1.000  1.000

## BBSDs+KS detection rate per shift and severity (all n)

severity    0.10  0.20   0.25  0.40   0.50   0.75   1.00  1.50  2.00  3.00
shift                                                                     
combo        NaN   NaN    NaN   NaN    NaN    NaN  0.857   NaN   1.0   NaN
concept    0.143   0.0    NaN   0.0    NaN    NaN    NaN   NaN   NaN   NaN
covariate    NaN   NaN    NaN   NaN  0.786    NaN  0.714   1.0   1.0   1.0
drift        NaN   NaN  0.143   NaN  0.429    NaN  0.714   NaN   1.0   NaN
label        NaN   NaN  0.500   NaN  0.643  0.643  0.714   NaN   NaN   NaN
noise        NaN   NaN  0.143   NaN  0.286    NaN  0.143   NaN   0.5   NaN

## Malignancy estimates (GBT, all datasets, harmful = true drop > 0.02)

                            MAE of accuracy  harmful? accuracy  harmful recall  benign recall
ATC (0 labels)                        0.055              0.621           0.556          0.846
Paper top-$k$ (100 labels)            0.062              0.741           0.733          0.769
Random (100 labels)                   0.042              0.707           0.711          0.692

## Adaptation accuracy, GBT (mean over seeds and severities)

                     No adapt   BBSE     IW  CORAL  D2A (ours)  Oracle
Adult     identity      0.875  0.871  0.872  0.859       0.875   0.869
          covariate     0.821  0.676  0.824  0.716       0.729   0.774
          noise         0.819  0.813  0.817  0.806       0.819   0.847
          drift         0.704  0.641  0.701  0.818       0.704   0.814
          label         0.823  0.846  0.821  0.737       0.835   0.835
          concept       0.799  0.801  0.799  0.781       0.799   0.785
          combo         0.819  0.697  0.828  0.680       0.680   0.780
Covertype identity      0.774  0.774  0.768  0.735       0.774   0.750
          covariate     0.810  0.760  0.832  0.504       0.810   0.825
          noise         0.667  0.661  0.661  0.646       0.661   0.678
          drift         0.663  0.622  0.666  0.723       0.667   0.734
          label         0.675  0.745  0.667  0.455       0.606   0.710
          concept       0.688  0.688  0.685  0.649       0.688   0.661
          combo         0.672  0.632  0.686  0.478       0.672   0.662
ACS       MS            0.742  0.737  0.727  0.771       0.751   0.807
          NY            0.786  0.784  0.764  0.776       0.781   0.780
          TX            0.790  0.781  0.773  0.794       0.790   0.794
          WY            0.760  0.730  0.729  0.762       0.762   0.754
All       mean          0.760  0.736  0.757  0.705       0.745   0.770

## All models, mean accuracy over all scenarios

       No adapt   BBSE     IW  CORAL  D2A (ours)  Oracle  D2A worst loss
model                                                                   
LR        0.727  0.665  0.728  0.702       0.721   0.755           0.226
RF        0.762  0.733  0.761  0.724       0.751   0.772           0.252
GBT       0.757  0.730  0.754  0.696       0.740   0.771           0.318
MLP       0.730  0.699  0.729  0.713       0.723   0.758           0.235

## Share of scenarios where a method loses > 1 pt vs no-adapt

BBSE          0.504
IW            0.259
CORAL         0.470
D2A (ours)    0.129

## D2A action per true shift (% of runs, GBT)

            none  BBSE     IW  CORAL
shift                               
identity   1.000  0.00  0.000  0.000
covariate  0.600  0.00  0.000  0.400
noise      0.500  0.00  0.500  0.000
drift      0.750  0.00  0.250  0.000
label      0.500  0.25  0.125  0.125
concept    1.000  0.00  0.000  0.000
combo      0.500  0.00  0.000  0.500
real       0.333  0.00  0.000  0.667

## D2A harmful flag vs true harm (GBT)

harmful        False  True 
truly harmful              
False             11      2
True              23     22

## Adult sample rows

   age     education         occupation  hours-per-week   sex  label
0   25          11th  Machine-op-inspct              40  Male  <=50K
1   38       HS-grad    Farming-fishing              50  Male  <=50K
2   28    Assoc-acdm    Protective-serv              40  Male   >50K
3   44  Some-college  Machine-op-inspct              40  Male   >50K
