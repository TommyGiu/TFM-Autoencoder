# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:32:07 2020

@author: TOSHIBA
"""
# =============================================================================
#                       IMOPORTACIÓN DE LIBRERIAS
# =============================================================================
import os

import matplotlib.pyplot as plt
import pydicom
import numpy as np
import dipy.sims.voxel as vox

# =============================================================================
#                       RUIDO RICIANO EN TEST
# =============================================================================
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
    



def add_noise(vol, snr=10, S0=None, noise_type='rician'):
    """ Add noise of specified distribution to a 4D array.
    Parameters
    -----------
    vol : array, shape (X,Y,Z,W)
        Diffusion measurements in `W` directions at each ``(X, Y, Z)`` voxel
        position.
    snr : float, optional
        The desired signal-to-noise ratio.  (See notes below.)
    S0 : float, optional
        Reference signal for specifying `snr` (defaults to 1).
    noise_type : string, optional
        The distribution of noise added. Can be either 'gaussian' for Gaussian
        distributed noise, 'rician' for Rice-distributed noise (default) or
        'rayleigh' for a Rayleigh distribution.
    Returns
    --------
    vol : array, same shape as vol
        Volume with added noise.
    Notes
    -----
    SNR is defined here, following [1]_, as ``S0 / sigma``, where ``sigma`` is
    the standard deviation of the two Gaussian distributions forming the real
    and imaginary components of the Rician noise distribution (see [2]_).
    References
    ----------
    .. [1] Descoteaux, Angelino, Fitzgibbons and Deriche (2007) Regularized,
           fast and robust q-ball imaging. MRM, 58: 497-510
    .. [2] Gudbjartson and Patz (2008). The Rician distribution of noisy MRI
           data. MRM 34: 910-914.
    Examples
    --------
    >>> signal = np.arange(800).reshape(2, 2, 2, 100)
    >>> signal_w_noise = add_noise(signal, snr=10, noise_type='rician')
    """
    orig_shape = vol.shape
    vol_flat = np.reshape(vol.copy(), (-1, vol.shape[-1]))

    if S0 is None:
        S0 = np.max(vol)

    for vox_idx, signal in enumerate(vol_flat):
        vol_flat[vox_idx] = vox.add_noise(signal, snr=snr, S0=S0,
                                          noise_type=noise_type)

    return np.reshape(vol_flat, orig_shape)

# =============================================================================
#               Prueba para ver cuanto ruido añadimos
# =============================================================================
for file_name in listall_slices[0:5]: 
    imagen_riccian = []
    file_path = os.path.join(test_dir, file_name)
    ds_test = pydicom.dcmread(file_path) # read dicom image
    img = (ds_test.pixel_array-ds_test.pixel_array.min())/(ds_test.pixel_array.max()-ds_test.pixel_array.min()) #Normalizar pixeles 
    gimg = add_noise(img, snr=14, S0=2, noise_type='rician')
    imagen_riccian.append(gimg)
   
    plt.figure(figsize=(2,2))
    plt.imshow(gimg,cmap=plt.cm.bone)
    plt.show()
    
# Elegimos snr=18, S0=1.5 (Primer ruido)
# Elegimos snr=14, S0=2 (segundo ruido)
        
# =============================================================================
#                   Añadimos el ruido elegido
# =============================================================================
imagen_riccian = []
contador = 0
for file_name in listall_slices: 
    file_path = os.path.join(test_dir, file_name)
    ds_test = pydicom.dcmread(file_path) # read dicom image
    img = (ds_test.pixel_array-ds_test.pixel_array.min())/(ds_test.pixel_array.max()-ds_test.pixel_array.min()) #Normalizar pixeles 
    gimg = add_noise(img, snr=14, S0=2, noise_type='rician')
    imagen_riccian.append(gimg)
    contador = contador+1
    print (contador)

imagenes = np.array(imagen_riccian)



imagenes = (imagenes-imagenes.min())/(imagenes.max()-imagenes.min())
imagenes.max()

np.save('test_riccian_2',imagenes)

for i in imagenes[0:5]:
    plt.figure(figsize=(2,2))
    plt.imshow(i,cmap=plt.cm.bone)
    plt.show()
    
    
# =============================================================================
#                      RUIDO RICIANO EN TRAIN  
# =============================================================================
  

test_dir1 = 'datos'
directorio = os.chdir(test_dir1)
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
    

imagen_riccian = []
contador = 0
for file_name in listall_slices: 
    file_path = os.path.join(train_dir, file_name)
    ds_train = pydicom.dcmread(file_path) # read dicom image
    img = (ds_train.pixel_array-ds_train.pixel_array.min())/(ds_train.pixel_array.max()-ds_train.pixel_array.min()) #Normalizar pixeles #Normalizar pixeles
    gimg = add_noise(img, snr=14, S0=2, noise_type='rician')
    imagen_riccian.append(gimg)
    contador = contador+1
    print (contador)

imagenes = np.array(imagen_riccian)


imagenes = (imagenes-imagenes.min())/(imagenes.max()-imagenes.min())
imagenes.max()

np.save('train_riccian_2',imagenes)

for i in imagenes[0:5]:
    plt.figure(figsize=(2,2))
    plt.imshow(i,cmap=plt.cm.bone)
    plt.show()




