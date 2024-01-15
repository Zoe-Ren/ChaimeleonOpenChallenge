import SimpleITK as sitk
from pathlib import Path
import json
#import random

import pandas as pd

from tensorflow import keras
#Sfrom tensorflow.keras import layers
from survival_model import survival_model
from pre_processing import read_json_file
from pre_processing import  read_volume
from pre_processing import resize_volume
from pre_processing import normalize
from pre_processing import wavelet_approximation
from pre_processing import process_scan
from tensorflow.keras.models import load_model
import numpy as np

from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)


class Lungcancerosprediction(ClassificationAlgorithm):
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
        self.image_input_dir = "/input/images/chest-ct/"
        #self.image_input_dir = r"G:\CHAIMELEON Open Challenges\train_lung\case_0412"
        self.image_input_path = list(Path(self.image_input_dir).glob("*.mha"))[0]
        #self.image_input_path = list(Path(self.image_input_dir).glob("*.nii.gz"))[0]
        # load clinical information
        # dictionary with patient_age and psa information
        with open("/input/clinical-information-lung-ct.json") as fp:
        #with open(r"G:\CHAIMELEON Open Challenges\train_lung\case_0412\case_0412.json") as fp: 
            self.clinical_info = json.load(fp)

        # path to output files
        self.os_output_file = Path("/output/overall-survival-months.json")

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
        #feature cnn
        volume =  process_scan(str(self.image_input_path))
        model_ct = load_model('model_feautre_extraction.h5')
        model_benchmark = model_ct 
        benchmark_layers = model_benchmark.layers
        benchmark_input = model_benchmark.input
        
        layer_outputs_benchmark = [layer.output for layer in benchmark_layers]
        features_benchmark = keras.Model(inputs=benchmark_input, outputs=layer_outputs_benchmark)
        one_image_data = np.expand_dims(volume, axis=0)

        extracted_benchmark = features_benchmark(one_image_data)
        f_output_dense = extracted_benchmark[17]
        f_output_dense_array = f_output_dense.numpy()
        feautre_cnn_test = f_output_dense_array[:,106]
        #age
        json_data = clinical_info 
        age_test = json_data['age']
        #gender
        gender_test = json_data['gender']
        if gender_test == 'MALE':
            gender_test = 0
        else:
            gender_test = 1
            
        # clinical category
        clinical_test = json_data['clinical_category']    
        if clinical_test == 'cT4':
              clinical_test = True
        else:
              clinical_test = False

        # metastasis_category
        metastasis_test = json_data['metastasis_category']    
        if metastasis_test  == 'cM1b':
              metastasis_test1  = True
        else:
              metastasis_test1  = False
        if metastasis_test  == 'cM1c':
              metastasis_test2  = True
        else:
              metastasis_test2  = False     

        # regional_nodes_category
        regional_nodes_category_test = json_data['regional_nodes_category']    
        if regional_nodes_category_test  == 'cN3':
              regional_nodes_category_test = True
        else:
              regional_nodes_category_test  = False
        
        x_test_data = {
              'age': age_test,
              'gender': gender_test,
              'feature_cnn':feautre_cnn_test ,
              'clinical_category': clinical_test,
              'metastasis_category_1': metastasis_test1,
              'metastasis_category_2': metastasis_test2,
              'regional_nodes_category':regional_nodes_category_test,
              }
        
        x_test_data = pd.DataFrame(x_test_data)
        # x_test_data.head()
        breslow = survival_model(318)
        preds = breslow.predict_expectation(x_test_data)
        # our code generates a random probability
        # our code generates a random pf
        overall_survival = preds [0]
        print('OS (months): ', overall_survival)

        # save case-level class
        with open(str(self.os_output_file), 'w') as f:
            json.dump(overall_survival, f)


if __name__ == "__main__":
    Lungcancerosprediction().predict()
