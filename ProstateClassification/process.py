# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 19:56:28 2024

@author: Ren
"""

import SimpleITK as sitk
from pathlib import Path
import json
#import random
#import matplotlib.pyplot as plt
# basic import
#import os
import tensorflow as tf
#import nibabel as nib
#import pandas as pd
import numpy as np
#from pre_processing import read_nifti_file
from pre_processing import normalize
from pre_processing import resize_volume
from pre_processing import process_scan
from pre_processing import read_json_file
from lele_model import get_model
from tensorflow.keras.models import load_model 
import pywt


from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)


class Prostatecancerriskprediction(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # path to image file
    
        self.image_input_dir = "/input/images/axial-t2-prostate-mri/"
        self.image_input_path = list(Path(self.image_input_dir).glob("*.mha"))[0]

        # load clinical information
        # dictionary with patient_age and psa information
        with open("/input/psa-and-age.json") as fp:
             self.clinical_info = json.load(fp)

        # # path to output files
        self.risk_score_output_file = Path("/output/prostate-cancer-risk-score.json")
        self.risk_score_likelihood_output_file = Path("/output/prostate-cancer-risk-score-likelihood.json")
    
    def predict(self):
        """
        Your algorithm goes here
        """        
        
        # read image
        image = sitk.ReadImage(str(self.image_input_path))
        clinical_info = self.clinical_info
        print('Clinical info: ')
        print(clinical_info)
       
        # TODO: Add your inference code here
        # Pre_processing image 

        volume = process_scan(str(self.image_input_path))
        #plt.imshow(volume[1,:,:])
       # Resize width, height and depth
       # Get_model 
        #model = get_model(depth=16, width=128, height=128)
        model1 = load_model('Wavelet_3d_cnn_version1.h5')
        model2 = load_model('Wavelet_3d_cnn_version2.h5')
        model3 = load_model('Wavelet_3d_cnn_version3.h5')

        
        # model prediction 
        prediction1 = model1.predict(np.expand_dims(volume, axis=0))[0]
        print(prediction1)
        prediction2 = model2.predict(np.expand_dims(volume, axis=0))[0]
        print(prediction2)
        prediction3 = model3.predict(np.expand_dims(volume, axis=0))[0]
        print(prediction3)
        
        
        # our code generates a random probability
        risk_score_likelihood = (prediction1+prediction2+prediction3)/3
        risk_score_likelihood = risk_score_likelihood[0]
        if risk_score_likelihood > 0.5:
            risk_score = 'High'
        else:
            risk_score = 'Low'
        print('Risk score: ', risk_score)
        print('Risk score likelihood: ', risk_score_likelihood)

        # save case-level class
        with open(str(self.risk_score_output_file), 'w') as f:
              json.dump(risk_score, f)

        # # # save case-level likelihood
        with open(str(self.risk_score_likelihood_output_file), 'w') as f:
             json.dump(float(risk_score_likelihood), f)


if __name__ == "__main__":
    Prostatecancerriskprediction().predict()