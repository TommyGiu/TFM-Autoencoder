# -*- coding: utf-8 -*-
"""
Created on Sat May  2 11:51:17 2020

@author: Maite (local)
"""

# =============================================================================
#                       CARGA DE LIBRERÍAS
# =============================================================================
import os
import glob
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import skimage
import dipy.sims.voxel as vox
import gc


# =============================================================================
#                       Imágenes originales
# =============================================================================

print(os.getcwd())
train_dir1 = 'datos'
directorio = os.chdir(train_dir1)
train_dir = 'train'

all_slices = os.listdir(train_dir)
listall_slices = list(all_slices)
print(listall_slices)

#Plot imagenes originales
contador = 0
fig, axs = plt.subplots(1, 3, figsize=(10, 10))
for file_name,ax in zip(listall_slices[0:3],axs.flatten()): 
    file_path = os.path.join(train_dir, file_name)
    ds_train = pydicom.dcmread(file_path) # read dicom image
    ax.imshow(ds_train.pixel_array, cmap=plt.cm.bone)
    contador +=1
    ax.set_title('Imagen real sin ruido ' + str(contador))

# Imágenes sin normalizar
y_train = []
for file_name in listall_slices[0:2]:
    
    file_path = os.path.join(train_dir, file_name)
    ds_train = pydicom.dcmread(file_path) # read dicom image
    img = ds_train.pixel_array
    y_train.append(img)

imagenes = np.array(y_train)


#plot histograma dos primeras imagenes sin Normalizar

for i in imagenes:
    plt.hist(i.flatten(),bins = 100)
    plt.xlabel('Intensidad de Pixel')
    plt.title('Intensidad de los píxeles de las imágenes originales sin normalizar')
    plt.show()


#Imágenes normalizadas
y_train = []
for file_name in listall_slices[0:2]:
    
    file_path = os.path.join(train_dir, file_name)
    ds_train = pydicom.dcmread(file_path) # read dicom image
    img = (ds_train.pixel_array-ds_train.pixel_array.min())/(ds_train.pixel_array.max()-ds_train.pixel_array.min()) #Normalizar pixeles #Normalizar pixeles
    y_train.append(img)

imagenes = np.array(y_train)

#plot histograma dos primeras imagenes Normalizadas

for i in imagenes:
    plt.hist(i.flatten(),bins = 100)
    plt.xlabel('Intensidad de Pixel')
    plt.title('Intensidad de los píxeles de las imágenes originales  normalizadas')
    plt.show()


# =============================================================================
#           HISTGRAMAS IMAGENES CON RUIDO: FUNCIONES
# =============================================================================

def histogramas (nombre_numpy,nombre_ruido):
    imagenes = np.load(nombre_numpy)

    for i in imagenes[0:4]:
        plt.hist(i.flatten(),bins = 100)
        plt.xlabel('Intensidad de Pixel')
        plt.title('Intensidad de los píxeles '+ str(nombre_ruido))
        plt.show()
    del (imagenes,i)
    gc.collect()
    

# =============================================================================
#                           RUIDO GAUSSIANO 1
# =============================================================================

imagenes = np.load(r'datos\x_train_gauss.npy')

histogramas (r'datos\x_train_gauss.npy','primer ruido Gauss')

# =============================================================================
#                         RUIDO GAUSSIANO 2
# =============================================================================

histogramas (r'datos\x_train_gauss_2.npy','segundo ruido Gauss')

# =============================================================================
#                    RUIDO SALT AND PEPPER 1
# =============================================================================

histogramas (r'datos\x_train_sep.npy','primer ruido s&p')

# =============================================================================
#                    RUIDO SALT AND PEPPER 2
# =============================================================================

histogramas (r'datos\x_train_sep_2.npy','segundo ruido s&p')

# =============================================================================
#                          RUIDO RICIANO 1
# =============================================================================

histogramas (r'datos\train_riccian.npy','primer ruido riciano')

# =============================================================================
#                          RUIDO  RICIANO 2
# =============================================================================

histogramas (r'datos\train_riccian_2.npy','segundo ruido riciano')














