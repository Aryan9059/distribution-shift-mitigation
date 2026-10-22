## Seeds per dataset

                 impact     detection    malignancy adaptation
dataset                                                       
acs:CA>MS  [0, 1, 2, 3]  [0, 1, 2, 3]  [0, 1, 2, 3]  [1, 2, 3]
acs:CA>NY  [0, 1, 2, 3]  [0, 1, 2, 3]  [0, 1, 2, 3]  [1, 2, 3]
acs:CA>TX  [0, 1, 2, 3]  [0, 1, 2, 3]  [0, 1, 2, 3]  [1, 2, 3]
acs:CA>WY  [0, 1, 2, 3]  [0, 1, 2, 3]  [0, 1, 2, 3]  [1, 2, 3]
adult            [0, 1]        [0, 1]        [0, 1]        [1]
covertype        [0, 1]        [0, 1]        [0, 1]        [1]

## Impact at the strongest severity (mean over seeds)

                           acc_source_test  acc_base  acc_oracle
dataset   shift     model                                       
adult     combo     GBT              0.867     0.793       0.766
                    LR               0.845     0.760       0.752
                    MLP              0.848     0.764       0.740
                    RF               0.843     0.769       0.752
          concept   GBT              0.867     0.744       0.725
                    LR               0.845     0.742       0.723
                    MLP              0.848     0.737       0.722
                    RF               0.843     0.731       0.714
          covariate GBT              0.867     0.760       0.686
                    LR               0.845     0.673       0.694
                    MLP              0.848     0.709       0.694
                    RF               0.843     0.732       0.683
          drift     GBT              0.867     0.554       0.857
                    LR               0.845     0.550       0.845
                    MLP              0.848     0.580       0.840
                    RF               0.843     0.535       0.833
          label     GBT              0.867     0.793       0.836
                    LR               0.845     0.767       0.821
                    MLP              0.848     0.787       0.821
                    RF               0.843     0.762       0.816
          noise     GBT              0.867     0.760       0.835
                    LR               0.845     0.766       0.834
                    MLP              0.848     0.775       0.828
                    RF               0.843     0.752       0.833
covertype combo     GBT              0.770     0.700       0.676
                    LR               0.722     0.629       0.649
                    MLP              0.751     0.638       0.642
                    RF               0.816     0.715       0.648
          concept   GBT              0.770     0.627       0.593
                    LR               0.722     0.593       0.574
                    MLP              0.751     0.608       0.577
                    RF               0.816     0.660       0.610
          covariate GBT              0.770     0.867       0.893
                    LR               0.722     0.811       0.836
                    MLP              0.751     0.825       0.851
                    RF               0.816     0.906       0.899
          drift     GBT              0.770     0.556       0.735
                    LR               0.722     0.506       0.718
                    MLP              0.751     0.469       0.708
                    RF               0.816     0.624       0.754
          label     GBT              0.770     0.615       0.724
                    LR               0.722     0.504       0.685
                    MLP              0.751     0.542       0.726
                    RF               0.816     0.625       0.743
          noise     GBT              0.770     0.590       0.636
                    LR               0.722     0.520       0.590
                    MLP              0.751     0.612       0.601
                    RF               0.816     0.661       0.647

## No shift (identity)

                 acc_source_test  acc_base  acc_oracle
dataset   model                                       
adult     GBT              0.867     0.874       0.869
          LR               0.845     0.855       0.855
          MLP              0.848     0.852       0.857
          RF               0.843     0.847       0.848
covertype GBT              0.770     0.774       0.755
          LR               0.722     0.727       0.722
          MLP              0.751     0.753       0.726
          RF               0.816     0.818       0.771

## ACS real shift

                 acc_source_test  acc_base  acc_oracle
dataset   model                                       
acs:CA>MS GBT              0.810     0.745       0.808
          LR               0.810     0.745       0.800
          MLP              0.812     0.743       0.805
          RF               0.802     0.747       0.792
acs:CA>NY GBT              0.810     0.787       0.782
          LR               0.810     0.783       0.778
          MLP              0.812     0.786       0.784
          RF               0.802     0.781       0.769
acs:CA>TX GBT              0.810     0.789       0.793
          LR               0.810     0.778       0.786
          MLP              0.812     0.786       0.782
          RF               0.802     0.786       0.777
acs:CA>WY GBT              0.810     0.759       0.752
          LR               0.810     0.749       0.751
          MLP              0.812     0.745       0.756
          RF               0.802     0.747       0.750

## Detection rate over all shifted scenarios (Adult + Covertype)

n                    10     20     50     100    200    500    1000
NoRed (KS)          0.330  0.477  0.648  0.727  0.761  0.795  0.852
PCA (KS)            0.307  0.386  0.591  0.693  0.761  0.807  0.841
SRP (KS)            0.352  0.398  0.648  0.727  0.784  0.807  0.830
BBSDs (KS)          0.318  0.375  0.534  0.591  0.648  0.784  0.818
BBSDh (Chi2)        0.330  0.375  0.432  0.545  0.625  0.682  0.739
Classif (Binomial)  0.000  0.000  0.375  0.636  0.739  0.784  0.830
NoRed (MMD)         0.295  0.386  0.614  0.682  0.705  0.784  0.818
PCA (MMD)           0.227  0.352  0.545  0.591  0.693  0.773  0.784
SRP (MMD)           0.250  0.330  0.545  0.648  0.670  0.773  0.795
BBSDs (MMD)         0.091  0.250  0.341  0.466  0.591  0.670  0.773

## False-alarm rate on the no-shift control

n                   10    20    50    100   200   500   1000
NoRed (KS)          0.00  0.00  0.00   0.0  0.00  0.00  0.00
PCA (KS)            0.00  0.00  0.00   0.0  0.25  0.00  0.25
SRP (KS)            0.00  0.25  0.00   0.0  0.00  0.00  0.00
BBSDs (KS)          0.00  0.25  0.00   0.0  0.00  0.25  0.25
BBSDh (Chi2)        0.25  0.25  0.25   0.0  0.00  0.00  0.00
Classif (Binomial)  0.00  0.00  0.00   0.0  0.00  0.00  0.00
NoRed (MMD)         0.00  0.00  0.00   0.0  0.00  0.00  0.00
PCA (MMD)           0.25  0.25  0.00   0.0  0.00  0.00  0.00
SRP (MMD)           0.25  0.00  0.00   0.0  0.25  0.50  0.00
BBSDs (MMD)         0.00  0.25  0.25   0.0  0.25  0.00  0.00

## BBSDs+KS detection rate per shift (identity = false alarms)

n           10     20     50     100    200    500    1000
shift                                                     
identity   0.000  0.250  0.000  0.000  0.000  0.250  0.250
covariate  0.650  0.750  0.850  1.000  1.000  1.000  1.000
noise      0.062  0.062  0.250  0.188  0.312  0.688  0.812
drift      0.312  0.375  0.688  0.625  0.750  0.938  0.938
label      0.188  0.375  0.375  0.625  0.750  0.938  1.000
concept    0.083  0.000  0.083  0.083  0.000  0.000  0.000
combo      0.625  0.625  1.000  1.000  1.000  1.000  1.000

## BBSDs+KS detection rate per shift and severity (all n)

severity    0.10   0.20   0.25  0.40   0.50   0.75   1.00   1.50   2.00  3.00
shift                                                                        
combo        NaN    NaN    NaN   NaN    NaN    NaN  0.786    NaN  1.000   NaN
concept    0.071  0.036    NaN   0.0    NaN    NaN    NaN    NaN    NaN   NaN
covariate    NaN    NaN    NaN   NaN  0.786    NaN  0.750  0.929  1.000   1.0
drift        NaN    NaN  0.393   NaN  0.464    NaN  0.821    NaN  0.964   NaN
label        NaN    NaN  0.429   NaN  0.607  0.571  0.821    NaN    NaN   NaN
noise        NaN    NaN  0.393   NaN  0.250    NaN  0.250    NaN  0.464   NaN

## Malignancy estimates (GBT, all datasets, harmful = true drop > 0.02)

                            MAE of accuracy  harmful? accuracy  harmful recall  benign recall
ATC (0 labels)                        0.053              0.578           0.500          0.846
Paper top-$k$ (100 labels)            0.061              0.716           0.711          0.731
Random (100 labels)                   0.036              0.750           0.767          0.692

## Adaptation accuracy, GBT (mean over seeds and severities)

                     No adapt   BBSE     IW  CORAL  D2A (ours)  Oracle
Adult     identity      0.874  0.876  0.874  0.846       0.876   0.870
          covariate     0.800  0.705  0.809  0.700       0.783   0.798
          noise         0.818  0.809  0.816  0.799       0.813   0.846
          drift         0.755  0.706  0.760  0.817       0.824   0.839
          label         0.823  0.845  0.822  0.753       0.829   0.839
          concept       0.793  0.794  0.794  0.779       0.794   0.784
          combo         0.806  0.721  0.811  0.672       0.811   0.806
Covertype identity      0.774  0.769  0.762  0.736       0.774   0.760
          covariate     0.811  0.777  0.833  0.539       0.833   0.808
          noise         0.675  0.663  0.672  0.648       0.653   0.684
          drift         0.672  0.642  0.668  0.730       0.732   0.729
          label         0.679  0.745  0.678  0.434       0.681   0.720
          concept       0.685  0.682  0.678  0.662       0.685   0.656
          combo         0.710  0.691  0.677  0.524       0.677   0.670
ACS       MS            0.747  0.744  0.733  0.777       0.740   0.809
          NY            0.787  0.783  0.772  0.772       0.777   0.784
          TX            0.788  0.780  0.774  0.795       0.787   0.793
          WY            0.758  0.726  0.735  0.759       0.735   0.750
All       mean          0.764  0.748  0.759  0.708       0.767   0.775

## All models, mean accuracy over all scenarios

       No adapt   BBSE     IW  CORAL  D2A (ours)  D2A+gate  Oracle
model                                                             
LR        0.728  0.681  0.724  0.706       0.740     0.729   0.759
RF        0.768  0.743  0.767  0.731       0.776     0.776   0.773
GBT       0.760  0.742  0.758  0.699       0.765     0.760   0.775
MLP       0.735  0.705  0.736  0.718       0.747     0.741   0.762

## Gain vs no-adapt over all models and scenarios: share losing > 1 pt / share gaining > 1 pt / worst loss

           lose>1pt  gain>1pt  worst   mean
bbse          0.522     0.116 -0.270 -0.030
iw            0.216     0.147 -0.065 -0.002
coral         0.478     0.164 -0.490 -0.035
d2a           0.194     0.254 -0.099  0.009
d2a_gated     0.121     0.129 -0.099  0.004

## Diagnosis accuracy per shift (all models; covariate->IW, label->BBSE, noise/drift->CORAL, identity->none)

shift
covariate    0.900
drift        0.500
identity     0.625
label        0.500
noise        0.625

## D2A action per true shift (% of runs, GBT)

            none   BBSE     IW  CORAL
shift                                
identity   0.500  0.500  0.000  0.000
covariate  0.000  0.000  0.900  0.100
noise      0.000  0.000  0.375  0.625
drift      0.000  0.000  0.500  0.500
label      0.000  0.375  0.625  0.000
concept    0.833  0.167  0.000  0.000
combo      0.000  0.000  1.000  0.000
real       0.417  0.000  0.583  0.000

## D2A harmful flag vs true harm (GBT)

harmful        False  True 
truly harmful              
False             11      2
True              29     16

## Adult sample rows

   age     education         occupation  hours-per-week   sex  label
0   25          11th  Machine-op-inspct              40  Male  <=50K
1   38       HS-grad    Farming-fishing              50  Male  <=50K
2   28    Assoc-acdm    Protective-serv              40  Male   >50K
3   44  Some-college  Machine-op-inspct              40  Male   >50K
