#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 14:33:29 2020

@author: thor
"""
# =============================================================================
#                           CARGA DE LIBRERIAS
# =============================================================================
import os
import glob
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import skimage


# =============================================================================
#                               Y-TEST
# =============================================================================

print(os.getcwd())
test_dir1 = 'datos'
directorio = os.chdir(test_dir1)
test_dir = 'test'

all_slices = os.listdir(test_dir)
listall_slices = list(all_slices)
print(listall_slices)

print("Number of DICOM files:", len(all_slices))

#Visualicación de datos

fig, axs = plt.subplots(4, 5, figsize=(20, 10))
for file_name,ax in zip(listall_slices[0:21],axs.flatten()): 
    file_path = os.path.join(test_dir, file_name)
    ds_test = pydicom.dcmread(file_path) # read dicom image
    ax.imshow(ds_test.pixel_array, cmap=plt.cm.bone)
    

# Creamos el numpy con los píxeles de las imagenes y normalizamos
y_test = []
for file_name in listall_slices:
    
    file_path = os.path.join(test_dir, file_name)
    ds_test = pydicom.dcmread(file_path) # read dicom image
    img = (ds_test.pixel_array-ds_test.pixel_array.min())/(ds_test.pixel_array.max()-ds_test.pixel_array.min()) #Normalizar pixeles #Normalizar pixeles
    y_test.append(img)

imagenes = np.array(y_test)
np.save('y_test',imagenes)

#Comprobamos que el numpy generado se ha gravado correctamente
prueba = np.load('y_test.npy')


for i in prueba[0:10,:,:]:
    plt.figure(figsize=(2,2))
    plt.imshow(i,cmap=plt.cm.bone)
    plt.show()
  

prueba.max()
prueba.min()

# =============================================================================
#                                   Y TRAIN
# =============================================================================
  

train_dir = 'train'

all_slices = os.listdir(train_dir)
listall_slices = list(all_slices)
print(listall_slices)

print("Number of DICOM files:", len(all_slices))

#Visualicación de datos
fig, axs = plt.subplots(4, 5, figsize=(20, 10))
for file_name,ax in zip(listall_slices[0:21],axs.flatten()): 
    file_path = os.path.join(train_dir, file_name)
    ds_train = pydicom.dcmread(file_path) # read dicom image
    ax.imshow(ds_train.pixel_array, cmap=plt.cm.bone)
    

# Creamos el numpy con los píxeles de las imagenes y normalizamos
y_train = []
for file_name in listall_slices:
    
    file_path = os.path.join(train_dir, file_name)
    ds_train = pydicom.dcmread(file_path) # read dicom image
    img = (ds_train.pixel_array-ds_train.pixel_array.min())/(ds_train.pixel_array.max()-ds_train.pixel_array.min()) #Normalizar pixeles #Normalizar pixeles
    y_train.append(img)

imagenes = np.array(y_train)
np.save('y_train',imagenes)

#Comprobamos que el numpy generado se ha gravado correctamente
prueba = np.load('y_train.npy')

for i in prueba[0:10,:,:]:
    plt.figure(figsize=(2,2))
    plt.imshow(i,cmap=plt.cm.bone)
    plt.show()




