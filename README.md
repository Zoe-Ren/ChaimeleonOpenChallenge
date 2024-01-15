# ChaimeleonOpenChallenge

## Prostate Classification 

For the binary classification of cancer levels, I employ an initial feature selection method, the 3D wavelet transform, to extract the volume of approximation coefficients. Given an imbalanced distribution with 210 cases at the low level and 85 cases at the high level, I randomly select 90 cases from the low level and all 85 cases from the high level to train a 3D CNN model. This process is repeated three times, and the prediction risk is derived by averaging the output predictions from these three runs.

The details can be found in the methode_description_prostate.pdf

## Lung Classification 

To predict survival time using CT images and clinical data, I initially utilised a Proportional Hazards model to select significant features from the clinical data, setting the threshold at p < 0.05. In the case of 3D image volumes, I also employ the 3D wavelet transform combined with a 3D CNN model to extract significant volumetric features. The selection process involves the Proportional Hazards model with a threshold set at p < 0.05. Finally, a combined dataset comprising clinical features and CT image volumetric features will be used to train a Proportional Hazards Model.

The details can be found in the methode_explication_Lung.pdf
