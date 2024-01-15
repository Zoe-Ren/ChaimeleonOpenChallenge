# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 19:29:07 2024

@author: Ren
"""

#from tensorflow import keras
#from tensorflow.keras import layers
#import json

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
#from sklearn.decomposition import PCA
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import  OneHotEncoder, StandardScaler
#from sklearn.compose import ColumnTransformer
#from sklearn.impute import SimpleImputer
#from tensorflow.keras.models import load_model

# 
def survival_model(nombre_train):
    list1 = np.load('data_0_253.npy')
    list2 = np.load('data_254_320.npy')
    array_data_squeezed = np.squeeze(list1)
    array_data_squeezed2 = np.squeeze(list2)
    data = np.concatenate((array_data_squeezed, array_data_squeezed2), axis=0)
    clinical_data=pd.read_csv("ct_clinical_data.csv", index_col='patient_id')
    
    # model 
    cph = CoxPHFitter()
    
    # x_train data
    x_train_final = clinical_data[0:nombre_train]
    # y_train data
    survivaltime = x_train_final[['survival_time_months']]
    survivaltime = survivaltime.to_numpy()
    event = x_train_final [['event']]
    event = event.to_numpy()
    survivaltime = np.squeeze(survivaltime)
    event = np.squeeze(event)
    # feautre cnn 
    data = data[0:nombre_train]
    feauture_cnn =data [:,106]
    feauture_cnn = np.squeeze(feauture_cnn)
    # clinique data
    ## clinical data
    clinical_train = clinical_data[['clinical_category','survival_time_months','event']]

    clinical_train.clinical_category.value_counts(sort=False, dropna=False)
    clinical_train['clinical_category']=clinical_train.clinical_category.apply(lambda x: 0 if x == 'cT1'
                                                                                  else 1 if x =='cT2'
                                                                                  else 2 if x =='cT1c'
                                                                                  else 3 if x =='cT3'
                                                                                  else 4 if x == 'cT1a'
                                                                                  else 5 if x == 'cT2b'
                                                                                  else 6 if x == 'cT1b'
                                                                                  else 7 if x == 'cTX'
                                                                                  else 8 if x == 'cT2a'
                                                                                  else 9 if x == 'cT4'
                                                                                  else 10)
    dummies_clinical = pd.get_dummies(clinical_train["clinical_category"], prefix = 'clinical_category')
    dummies_clinical = dummies_clinical[["clinical_category_9"]]
    dummies_clinical = dummies_clinical.to_numpy()
    dummies_clinical = np.squeeze(dummies_clinical)

    ## metastasis_category fequtre selection
    metastasis_train = clinical_data[['metastasis_category','survival_time_months','event']]
    metastasis_train.metastasis_category.value_counts(sort=False, dropna=False)
    metastasis_train['metastasis_category']=metastasis_train.metastasis_category.apply(lambda x: 0 if x == 'cM1'
                                                                                  else 1 if x =='cM1a'
                                                                                  else 2 if x =='cM1b'
                                                                                  else 3 if x =='cM0'
                                                                                  else 4 if x == 'cM1c'
                                                                                  else 5)
    dummies_metastasis = pd.get_dummies(metastasis_train["metastasis_category"], prefix = 'metastasis_category')
    dummies_metastasis1 = dummies_metastasis[["metastasis_category_2"]]
    dummies_metastasis1 = dummies_metastasis1.to_numpy()
    dummies_metastasis1 = np.squeeze(dummies_metastasis1)
    dummies_metastasis2 = dummies_metastasis[["metastasis_category_4"]]
    dummies_metastasis2 = dummies_metastasis2.to_numpy()
    dummies_metastasis2 = np.squeeze(dummies_metastasis2)
    ## regional_ nodes
    regional_nodes_category_train = clinical_data[['regional_nodes_category','survival_time_months','event']]
    regional_nodes_category_train.regional_nodes_category.value_counts(sort=False, dropna=False)
    regional_nodes_category_train['regional_nodes_category']=regional_nodes_category_train.regional_nodes_category.apply(lambda x: 0 if x == 'cN1'
                                                                                  else 1 if x =='cN2'
                                                                                  else 2 if x =='cNX'
                                                                                  else 3 if x =='cN0'
                                                                                  else 4 if x == 'cN3'
                                                                                  else 5)
    regional_nodes_category_train.regional_nodes_category.value_counts(sort=False, dropna=False)
    dummies_regional_nodes = pd.get_dummies(regional_nodes_category_train["regional_nodes_category"], prefix = 'regional_nodes_category')
    dummies_regional_nodes = dummies_regional_nodes[["regional_nodes_category_4"]]
    dummies_regional_nodes = dummies_regional_nodes .to_numpy()
    dummies_regional_nodes  = np.squeeze(dummies_regional_nodes )

    ## age
    age_train = clinical_data[['age']]
    age_train =age_train.to_numpy()
    age_train = np.squeeze(age_train)

    # gender
    gender_train = clinical_data[['gender']]
    gender_train['gender']=gender_train.gender.apply(lambda x: 0 if x == 'MALE'
                                                     else 1)
    gender_train = gender_train.to_numpy()
    gender_train = np.squeeze(gender_train)
    
    # 
    x_train_data = {
        'age': age_train[0:nombre_train],
        'gender': gender_train[0:nombre_train],
        'feature_cnn' :feauture_cnn,
        'clinical_category': dummies_clinical[0:nombre_train],
        'metastasis_category_1': dummies_metastasis1[0:nombre_train],
        'metastasis_category_2': dummies_metastasis2[0:nombre_train],
        'regional_nodes_category': dummies_regional_nodes [0:nombre_train],
        #'feautre_cnn1' :feauture_cnn1,
        'survival_time_months': survivaltime, 
        'event': event}
    x_train = pd.DataFrame(x_train_data)
    breslow = cph.fit(x_train, duration_col='survival_time_months', event_col='event')
    cph.print_summary()
    return breslow

#breslow = survival_model(310)
#breslow.print_summary()
