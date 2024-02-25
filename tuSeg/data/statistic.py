import glob
import os
import re
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage import measure
from multiprocessing import Pool

def findbb(volume):
    image_shape = volume.shape
    bb = np.zeros((6,), dtype=np.uint)
    bb_extend = 0

    for i in range(image_shape[0]):
        img_slice_begin = volume[i,:,:]
        if np.sum(img_slice_begin)>0:
            bb[0] = np.max([i-bb_extend, 0])
            break;

    for i in range(image_shape[0]):
        img_slice_end = volume[image_shape[0]-1-i,:,:]
        if np.sum(img_slice_end)>0:
            bb[1] = np.min([image_shape[0]-1-i+bb_extend, image_shape[0]-1])
            break

    for i in range(image_shape[1]):
        img_slice_begin = volume[:,i,:]
        if np.sum(img_slice_begin)>0:
            bb[2] = np.max([i-bb_extend, 0])
            break

    for i in range(image_shape[1]):
        img_slice_end = volume[:, image_shape[1]-1-i,:]
        if np.sum(img_slice_end)>0:
            bb[3] = np.min([image_shape[1]-1-i+bb_extend, image_shape[1]-1])
            break

    for i in range(image_shape[2]):
        img_slice_begin = volume[:,:,i]
        if np.sum(img_slice_begin)>0:
            bb[4] = np.max([i-bb_extend, 0])
            break

    for i in range(image_shape[2]):
        img_slice_end = volume[:,:,image_shape[2]-1-i]
        if np.sum(img_slice_end)>0:
            bb[5] = np.min([image_shape[2]-1-i+bb_extend, image_shape[2]-1])
            break
    
    return bb

def statistic(file):

    #ct = sitk.ReadImage(os.path.join(cts_base, file), sitk.sitkInt16)
    #ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(seg_base, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    #ct_spacing = ct.GetSpacing()
    #seg_spacing = seg.GetSpacing()

    liver = seg_array.copy()
    liver[liver>0]=1
    #count_liver = liver.sum()
    liver_box = findbb(liver)

    # liver_volume = ct_array.copy()
    #liver_volume = liver_volume * liver
    # intense_liver = liver_volume.sum()/count_liver


    # tumor = seg_array.copy()
    #tumor[tumor==1]=0
    #tumor[tumor==2]=1
    #count_tumor = tumor.sum()

    #label_tumor = measure.label(tumor, connectivity=2)
    
    #tumor_volume = ct_array.copy()
    #tumor_volume = tumor_volume * tumor
    #intense_tumor = tumor_volume.sum()/count_tumor

    #classNum = seg_array.max()

    return [file, liver_box, seg_array.shape]

if __name__ == '__main__':
    cts_base = '/mountdisk/experiment/one/data/ct/'
    seg_base = '/mountdisk/experiment/one/data/seg/'

    file_list = []

    files = os.listdir(cts_base)

    p = Pool(12)
    file_list.append(p.map(statistic, files))
    p.close()
    p.join() 

    data = np.array(file_list).squeeze()
    data = data[data[:, 0].argsort()]
    np.savetxt('/mountdisk/experiment/one/data/statistic.csv', data, delimiter=',', fmt='%s')
