#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:49:43 2020

@author: thor
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
#                         RUIDO SALT AND PEPPER EN TEST
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
       
# Ruido 1 amount= 0.08 , Ruido 2 amount= 0.3 
    
def plotnoise(img, mode, r, c, i,amount = 0.3):
    plt.subplot(r,c,i)
    if mode is not None:
        gimg = skimage.util.random_noise(img, mode=mode,amount=amount)
        plt.imshow(gimg,cmap=plt.cm.bone)
    else:
        plt.imshow(img,cmap=plt.cm.bone)
    plt.title(mode)
    plt.axis("off")
    
    
imagen_salt = []
for file_name in listall_slices:
    plt.figure(figsize=(20,20))
    r=4
    c=5
    file_path = os.path.join(test_dir, file_name)
    ds_test = pydicom.dcmread(file_path) # read dicom image
    img = (ds_test.pixel_array-ds_test.pixel_array.min())/(ds_test.pixel_array.max()-ds_test.pixel_array.min()) #Normalizar pixeles
    plotnoise(img, "s&p", r,c,1)
    plotnoise(img, None, r,c,2)
    plt.show()
    gimg = skimage.util.random_noise(img, mode="s&p",amount = 0.3)
    imagen_salt.append(gimg)
    

imagenes = np.array(imagen_salt)
imagenes.max()
imagenes.min()

np.save('x_test_sep_2',imagenes)


# Comprobación de la correcta carga del numpy generado
prueba = np.load('x_test_sep_2.npy')


for i in prueba[0:10,:,:]:
    plt.figure(figsize=(2,2))
    plt.imshow(i,cmap=plt.cm.bone)
    plt.show()
  

prueba.max()
#plot histograma
kk = prueba[1,:,:]
plt.hist(kk.flatten(),bins = 100)

# =============================================================================
#                       RUIDO SALT AND PEPPER EN TRAIN
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

# Ruido 1 amount= 0.08 , Ruido 2 amount= 0.3 
def plotnoise(img, mode, r, c, i,amount = 0.3):
    plt.subplot(r,c,i)
    if mode is not None:
        gimg = skimage.util.random_noise(img, mode=mode,amount=amount)
        plt.imshow(gimg,cmap=plt.cm.bone)
    else:
        plt.imshow(img,cmap=plt.cm.bone)
    plt.title(mode)
    plt.axis("off")

contador = 0
imagen_salt = []
for file_name in listall_slices:
    plt.figure(figsize=(20,20))
    r=4
    c=5
    file_path = os.path.join(train_dir, file_name)
    ds_train = pydicom.dcmread(file_path) # read dicom image
    img = (ds_train.pixel_array-ds_train.pixel_array.min())/(ds_train.pixel_array.max()-ds_train.pixel_array.min()) #Normalizar pixeles #Normalizar pixeles
    plotnoise(img, "s&p", r,c,1)
    plotnoise(img, None, r,c,2)
    plt.show()
    gimg = skimage.util.random_noise(img, mode="s&p",amount=0.3)
    imagen_salt.append(gimg)
    contador +=1
    print (contador)

imagenes = np.array(imagen_salt)
imagenes.max()
imagenes.min()

np.save('x_train_sep_2',imagenes)

# Comprobación de la correcta carga del numpy generado
prueba = np.load('x_train_sep_2.npy')


for i in prueba[0:10,:,:]:
    plt.figure(figsize=(2,2))
    plt.imshow(i,cmap=plt.cm.bone)
    plt.show()
  
