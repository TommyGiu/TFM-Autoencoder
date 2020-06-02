# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 11:53:36 2020

@author: 
"""
# =============================================================================
#                       IMPORTACIÓN DE LIBRERIAS
# =============================================================================
import os
import glob
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import skimage

# =============================================================================
#                       RUIDO GAUSSIANO EN TEST
# =============================================================================
print(os.getcwd())
test_dir1 = 'datos'
directorio = os.chdir(test_dir1)
test_dir = 'test'

all_slices = os.listdir(test_dir)
listall_slices = list(all_slices)
print(listall_slices)

print("Number of DICOM files:", len(all_slices))


fig, axs = plt.subplots(4, 5, figsize=(20, 10))
for file_name,ax in zip(listall_slices[0:21],axs.flatten()): 
    file_path = os.path.join(test_dir, file_name)
    ds_test = pydicom.dcmread(file_path) # read dicom image
    ax.imshow(ds_test.pixel_array, cmap=plt.cm.bone)
    
pixel = ds_test.pixel_array.shape[0] #320 pixeles


# Ruido 1 var = 0.002, Ruido 2 var = 0.01
def plotnoise(img, mode, r, c, i,var = 0.01):
    plt.subplot(r,c,i)
    if mode is not None:
        gimg = skimage.util.random_noise(img, mode=mode,var=var)
        plt.imshow(gimg,cmap=plt.cm.bone)
    else:
        plt.imshow(img,cmap=plt.cm.bone)
    plt.title(mode)
    plt.axis("off")

imagen_Gauss = []
for file_name in listall_slices:
    plt.figure(figsize=(20,20))
    r=4
    c=5
    file_path = os.path.join(test_dir, file_name)
    ds_test = pydicom.dcmread(file_path) # read dicom image
    img = (ds_test.pixel_array-ds_test.pixel_array.min())/(ds_test.pixel_array.max()-ds_test.pixel_array.min()) #Normalizar pixeles #Normalizar pixeles
    plotnoise(img, "gaussian", r,c,1)
    plotnoise(img, None, r,c,2)
    plt.show()
    gimg = skimage.util.random_noise(img, mode="gaussian",var=0.01) #0.03 valor inicial
    imagen_Gauss.append(gimg)

imagenes = np.array(imagen_Gauss)
np.save('x_test_gauss_2',imagenes)

# Comprobación de la correcta carga del numpy generado
prueba = np.load('test_gauss.npy')


for i in prueba[0:10,:,:]:
    plt.figure(figsize=(2,2))
    plt.imshow(i,cmap=plt.cm.bone)
    plt.show()
  

prueba.max()

# =============================================================================
#                          RUIDO GAUSSIANO EN TRAIN
# =============================================================================
  

train_dir = 'train'

all_slices = os.listdir(train_dir)
listall_slices = list(all_slices)
print(listall_slices)

print("Number of DICOM files:", len(all_slices))


fig, axs = plt.subplots(4, 5, figsize=(20, 10))
for file_name,ax in zip(listall_slices[0:21],axs.flatten()): 
    file_path = os.path.join(train_dir, file_name)
    ds_train = pydicom.dcmread(file_path) # read dicom image
    ax.imshow(ds_train.pixel_array, cmap=plt.cm.bone)
    
pixel = ds_train.pixel_array.shape[0] #320 pixeles

# Ruido 1 var = 0.002, Ruido 2 var = 0.01
def plotnoise(img, mode, r, c, i,var = 0.01):
    plt.subplot(r,c,i)
    if mode is not None:
        gimg = skimage.util.random_noise(img, mode=mode,var=var)
        plt.imshow(gimg,cmap=plt.cm.bone)
    else:
        plt.imshow(img,cmap=plt.cm.bone)
    plt.title(mode)
    plt.axis("off")

imagen_Gauss = []
contador = 0
for file_name in listall_slices:
    plt.figure(figsize=(20,20))
    r=4
    c=5
    file_path = os.path.join(train_dir, file_name)
    ds_train = pydicom.dcmread(file_path) # read dicom image
    img = (ds_train.pixel_array-ds_train.pixel_array.min())/(ds_train.pixel_array.max()-ds_train.pixel_array.min()) #Normalizar pixeles #Normalizar pixeles
    plotnoise(img, "gaussian", r,c,1)
    plotnoise(img, None, r,c,2)
    plt.show()
    gimg = skimage.util.random_noise(img, mode="gaussian",var=0.01)
    imagen_Gauss.append(gimg)
    contador+=1
    print (contador)

imagenes = np.array(imagen_Gauss)
np.save('x_train_gauss_2',imagenes)

# Comprobación de la correcta carga del numpy generado
prueba = np.load('x_train_gauss.npy')

for i in prueba[0:10,:,:]:
    plt.figure(figsize=(2,2))
    plt.imshow(i,cmap=plt.cm.bone)
    plt.show()
  
prueba.max()


