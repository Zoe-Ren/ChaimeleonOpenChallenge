# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 20:07:42 2024

@author: Ren
"""
import json
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import pywt
def read_json_file(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data
def read_volume(filepath):
    """Read and load volume"""
    # Read file
    #scan = nib.load(filepath)
    scan = sitk.ReadImage(filepath)
    # Get raw data
    #scan = scan.get_fdata()
    scan = sitk.GetArrayFromImage(scan)
    return scan
def normalize(volume):
    """Normalize the volume"""
    volume_mean = np.mean(volume)
    volume_std = np.std(volume)
    volume = (volume - volume_mean) / volume_std
    volume = volume.astype("float32")
    return volume
def resize_volume(img,filepath):
    """Resize across z-axis"""
    scan = sitk.ReadImage(filepath)
    # Set the desired depth
    desired_depth = 32
    desired_width = 256
    desired_height = 256
    # Get current depth
    current_depth = scan.GetDepth()
    current_width = scan.GetWidth()
    current_height = scan.GetHeight()
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 0, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (depth_factor, height_factor,width_factor ), order=1)
    return img

def wavelet_approximation(volume):

    wavename = 'haar'
    coeffs = pywt.dwtn(volume, wavename)
    appro = coeffs['aaa']
    appro_volume = appro.astype("float32")
    return appro_volume
def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_volume(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume,path)
    #wavelet decomposition
    volume = wavelet_approximation(volume)
    return volume