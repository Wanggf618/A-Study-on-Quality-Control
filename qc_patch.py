# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np
from skimage import io, color, img_as_ubyte
from scipy import ndimage as ndi
import skimage
from skimage.filters import rank
import scipy.signal
from skimage.morphology import remove_small_objects, disk, binary_opening
from skimage import measure
from skimage.metrics import structural_similarity
from skimage import io, transform
from skimage import io, morphology, img_as_ubyte, measure
from collections import Counter
import pandas as pd
import cv2 as cv2
import glob
from PIL import Image
from skimage.color import convert_colorspace, rgb2gray
from skimage.filters import sobel
import gc
import csv
pd.set_option('display.max_columns',None)
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def blend2Images(img, mask):
    if (img.ndim == 3):
        img = color.rgb2gray(img)
    if (mask.ndim == 3):
        mask = color.rgb2gray(mask)
    img = img[:, :, None] * 1.0  # can't use boolean
    mask = mask[:, :, None] * 1.0
    out = np.concatenate((mask, img, mask), 2)
    return out
    
def proportion_mask(mask_use,data2):
    x = mask_use.sum()
    y = data2.sum()
    proportion = y/x
    return proportion
def remove_large_objects(img, max_size):
    # code taken from morphology.remove_small_holes, except switched < with >
    selem = ndi.generate_binary_structure(img.ndim, 1)
    ccs = np.zeros_like(img, dtype=np.int32)
    ndi.label(img, selem, output=ccs)
    component_sizes = np.bincount(ccs.ravel())
    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    img_out = img.copy()
    img_out[too_big_mask] = 0
    return img_out

def getIntensityThresholdPercent(s, params=1):
   
   
   img = s["image_work_size"]
   img = color.rgb2gray(img)
   min_object_size = 50
   if params==1:
      upper_thresh= .9
      #lower_var = 10
      #map_var = img_var > lower_var
      map = img < upper_thresh
      mask_flat =  map == 0
      mask_flat = remove_small_objects(mask_flat, min_size=min_object_size)

      s["img_mask_bright"] = mask_flat
      prev_mask = s['img_mask_use']>0
      s['img_mask_use'] = (prev_mask) & ~(mask_flat)
   else:
      upper_thresh= .15
      map = img < upper_thresh
      mask_flat =  map>0
      prev_mask = s['img_mask_use']>0
      s['img_mask_use'] = (prev_mask) & (~mask_flat)
      darktissue_tem = np.sum(mask_flat)
      return(darktissue_tem)

def imgfold(s):
   img = s["image_work_size"]
   res1 = img 
   hsv_s = cv2.cvtColor(res1, cv2.COLOR_BGR2HSV)
   H, S, V = cv2.split(hsv_s)
   fold_img = ((S>V)).astype(int)

   fold_img = fold_img.astype("uint8")
   fold_img = cv2.erode(fold_img, (3,3), iterations=3)
   fold_img = fold_img.astype("uint8")
   num_objects, labels = cv2.connectedComponents(fold_img)
   if num_objects >= 5:
       fold_img = remove_small_objects(fold_img > 0, min_size=500)
       fold_img = fold_img.astype("uint8")
   num_objects, labels = cv2.connectedComponents(fold_img)
   if num_objects > 4:
       fold_img = np.zeros((512,512),dtype = int)
   fold_img = remove_small_objects(fold_img>0, min_size=300)
   fold_img = (fold_img>0).astype(int)
   s['img_mask_fold'] = fold_img
   prev_mask = s['img_mask_use']>0
   s['img_mask_use'] = (prev_mask)&(~fold_img)


def identifyBlurryRegions(s):
    blur_radius = 3 #7
    blur_threshold = .019 #0.7
    img =  s["image_work_size"]
    img = color.rgb2gray(img)
    img = img*~s["img_mask_bright"]
    img_laplace = np.abs(skimage.filters.laplace(img))
    mask = skimage.filters.gaussian(img_laplace, sigma=blur_radius) < blur_threshold
    mask = np.bitwise_and(mask,~s["img_mask_bright"])
    s["img_mask_blurry"] = mask
    prev_mask = s['img_mask_use']>0
    s['img_mask_use'] = (prev_mask) & (~mask)
    
def detectSmoothness(s):
    thresh=.01
    kernel_size= 10
    min_object_size= 5
    img = s["image_work_size"]
    img = color.rgb2gray(img)
    avg = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    imf = scipy.signal.convolve2d(img, avg, mode="same")
    mask_flat = abs(imf - img) < thresh

    mask_flat = remove_small_objects(mask_flat, min_size=min_object_size)
    mask_flat = ~remove_small_objects(~mask_flat, min_size=min_object_size)

    mask_flat = mask_flat*(~s['img_mask_bright'])
    prev_mask = s['img_mask_use']>0
    s['img_mask_use'] = (prev_mask) & (~mask_flat)
    mask_flat = np.sum(mask_flat)
    return(mask_flat)
def removeFatlikeTissue(s):
    fat_cell_size = 32
    kernel_size = 5
    max_keep_size = 500
    img_reduced = morphology.remove_small_holes(s["img_mask_use"], area_threshold=fat_cell_size)
    img_small = img_reduced & np.invert(s["img_mask_use"])
    img_small = ~morphology.remove_small_holes(~img_small, area_threshold=9)
    mask_dilate = morphology.dilation(img_small, selem=np.ones((kernel_size, kernel_size)))
    mask_dilate_removed = remove_large_objects(mask_dilate, max_keep_size)
    mask_fat = mask_dilate & ~mask_dilate_removed
    fatlike =np.sum((mask_fat) > 0)
    prev_mask = s["img_mask_use"]>0
    s["img_mask_use"] = prev_mask & ~mask_fat
    return(fatlike)


def fillSmallHoles(s):
    min_size = 15
    prev_mask_tem = s["img_mask_use"] == 0
    img_reduced = remove_small_objects(prev_mask_tem, min_size=min_size)
    smallholes = np.sum(prev_mask_tem)-np.sum(img_reduced)
    img_small = img_reduced & np.invert(s["img_mask_use"])
    s["img_mask_small_removed"] = np.sum((img_small) > 0)
    prev_mask = s["img_mask_use"]
    s["img_mask_use"] =(prev_mask)|(~img_reduced)
    return(smallholes)


def computeHistogram(img):
    img = img_as_ubyte(img)
    result = np.zeros(shape=(20, 3))
    for chan in range(0, 3):
        vals = img[:, :, chan].flatten()
        result[:, chan] = np.histogram(vals, bins=20, density=True, range=[0, 255])[0]
    result =np.float32(result)
    return result

def compareToTemplates(s):
    img = s["image_work_size"]
    imghst = computeHistogram(img)
    for i in range(1, 5):
       templates = mpimg.imread('/your path/' + str(i) + '.png')
       template = computeHistogram(templates)
       val = np.sum(pow(abs(template - imghst), 2))
       s[f'hist_templates_{i}'] = val
       
def getContrast(s):
    img = s["image_work_size"]
    img = rgb2gray(img)
    sobel_img = sobel(img) ** 2
    tenenGrad_contrast = np.sqrt(np.sum(sobel_img)) / img.size
    max_img = img.max()
    min_img = img.min()
    contrast = (max_img - min_img) / (max_img + min_img)
    # RMS contrast
    rms_contrast = np.sqrt(pow(img - img.mean(), 2).sum() / img.size)
    return(tenenGrad_contrast, contrast, rms_contrast)
    
def getBrightnessGray(s):
    img = s["image_work_size"]
    img_yuv = convert_colorspace(img, "RGB", 'YUV')
    for chan in range(0, 3):
        vals = img[:, :, chan]
        s[f'chan_{chan + 1}_maen'] = vals.mean()
        s[f'chan_{chan + 1}_std'] = vals.std()
        vals_yuv = img_yuv[:, :, chan]
        s[f'yuv_{chan + 1}_maen'] = vals_yuv.mean()
        s[f'yuv_{chan + 1}_std'] = vals_yuv.std()

    img_g = rgb2gray(img)
    mean_tem = img_g.mean()
    std_tem = img_g.std()
    return(mean_tem, std_tem)





if __name__ == "__main__":

 for k in range(4:5):
    list_title = ['path','use','bright','fold','blurry','contrast']
    with open(f"patch_qc_{k}.csv","a",newline='') as csvfile: 
             writer = csv.writer(csvfile)
             writer.writerow(list_title)
             csvfile.close()



 for j in range(4:5):
   datapath = f'test_{j}.csv'
   data = pd.read_csv(datapath, index_col=None,header=None)

   he_path = []
   use = []
   bright = []
   fold = []
   blurry = []
   contrast = []
   list_title = ['path','use','bright','fold','blurry','contrast']
   
   with open(f"patch_qc_{j}.csv","a",newline='') as csvfile: 
             writer = csv.writer(csvfile)
             writer.writerow(list_title)
             csvfile.close()
  
   for i in range(len(data)):
       data_qcpath = data.iloc[i,0]

       img = mpimg.imread(data_qcpath)
        
       (filepath, filename) = os.path.split(data_qcpath)
       (name, suffix) = os.path.splitext(filename)
       img = np.array(img)

       x, y, z = img.shape
       s = {'image_work_size': img}
       s['img_mask_use'] = np.ones((x, y))
       s['img_mask_bright'] = np.zeros((x, y))
       s['img_mask_fold'] = np.zeros((x, y))
       s['img_mask_blurry'] = np.zeros((x, y))

       getIntensityThresholdPercent(s, params=1)
       darktissue_tem = getIntensityThresholdPercent(s,params=0)
       identifyBlurryRegions(s)
       imgfold(s)
       tenenGrad_contrast_tem, contrast_tem, rms_contrast_tem = getContrast(s)
       mean_tem, std_tem = getBrightnessGray(s)

       bright_tem = np.sum(s['img_mask_bright'])/(x*y)
       fold_tem = np.sum(s['img_mask_fold'])/(x*y)
       blurry_tem = np.sum(s['img_mask_blurry'])/(x*y)
       use_tem = np.sum(s['img_mask_use'])/(x*y)
   
       list1 = [name,use_tem,bright_tem,fold_tem,blurry_tem,contrast_tem]
       
       with open(f"./patch_qc_{j}.csv","a",newline='') as csvfile: 
             writer = csv.writer(csvfile)
             writer.writerow(list1)
             csvfile.close()

