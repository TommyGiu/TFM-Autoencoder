# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:14:30 2020

@author: Maite (local)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import r2_score
import gc

# =============================================================================
#                               CARGA DE MODELOS
# =============================================================================

modelo_1 = load_model ('modelos\modelo1_gauss.h5')
modelo_2 = load_model ('modelos\modelo2_gauss.h5')
modelo_3 = load_model ('modelos\modelo3_gauss.h5')
modelo_4 = load_model ('modelos\modelo4_gauss.h5')


# =============================================================================
#            CARGA X_TEST_GAUSS (IMAGENES CON RUIDO PARA VALIDACIÓN)
# =============================================================================

x_val = np.load (r'datos\x_test_gauss.npy')

x_val = x_val.reshape(-1, 320,320, 1) #Añadimos la dimension color

x_test_plot = x_val.reshape(-1, 320,320) # Lo usaremos en el plot

# =============================================================================
#           CARGA Y_TEST (IMAGENES SIN RUIDO PARA VALIDACIÓN)
# =============================================================================

y_test = np.load (r'datos\y_test.npy')

y_test = y_test.reshape(-1, 320,320, 1) #Añadimos la dimension color

y_test_1d = y_test.flatten()

y_test_plot = y_test.reshape(-1, 320,320) # Lo usaremos en el plot

# =============================================================================
#                          MODELO 1: PREDICCIONES Y R2
# =============================================================================


predict = modelo_1.predict(x_val)

predict_1d = predict.flatten()

error = r2_score(predict_1d, y_test_1d) # r2 = 0.95

predict_plot = predict.reshape(-1, 320,320) # Lo usaremos en el plot

# =============================================================================
#                PLOTEAMOS LAS IMAGENES
# =============================================================================

for salida,real,inicial in zip (predict_plot[0:5],y_test_plot[0:5], x_test_plot[0:5]):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(14, 14))
    ax1.imshow(inicial,cmap=plt.cm.bone)#imagen perturbada con ruido
    ax1.set_title('Imagen perturbada con ruido')
    ax2.imshow(salida,cmap=plt.cm.bone)#salida autoencoder
    ax2.set_title('Salida del autoencoder')
    ax3.imshow(real,cmap=plt.cm.bone)#imagen real sin ruido
    ax3.set_title('Imagen real sin ruido')

# =============================================================================
#  BORRAMOS DATOS QUE NO NECESITAMOS Y LIBERAMOS ESPACIO
# =============================================================================

del (predict,predict_1d,predict_plot)
del (salida,real,inicial)
del modelo_1
gc.collect()

# =============================================================================
#                          MODELO 2 : PREDICCIONES Y R2
# =============================================================================


predict = modelo_2.predict(x_val)

predict_1d = predict.flatten()

error = r2_score(predict_1d, y_test_1d) # r2 = 0.82

predict_plot = predict.reshape(-1, 320,320) # Lo usaremos en el plot

# =============================================================================
#                PLOTEAMOS LAS IMAGENES
# =============================================================================

for salida,real,inicial in zip (predict_plot[0:5],y_test_plot[0:5], x_test_plot[0:5]):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(14, 14))
    ax1.imshow(inicial,cmap=plt.cm.bone)#imagen perturbada con ruido
    ax1.set_title('Imagen perturbada con ruido')
    ax2.imshow(salida,cmap=plt.cm.bone)#salida autoencoder
    ax2.set_title('Salida del autoencoder')
    ax3.imshow(real,cmap=plt.cm.bone)#imagen real sin ruido
    ax3.set_title('Imagen real sin ruido')

# =============================================================================
#  BORRAMOS DATOS QUE NO NECESITAMOS Y LIBERAMOS ESPACIO
# =============================================================================

del (predict,predict_1d,predict_plot)
del (salida,real,inicial)
del modelo_2
gc.collect()

# =============================================================================
#                          MODELO 3 : PREDICCIONES Y R2
# =============================================================================


predict = modelo_3.predict(x_val)

predict_1d = predict.flatten()

error = r2_score(predict_1d, y_test_1d) # r2 = 0.97

predict_plot = predict.reshape(-1, 320,320) # Lo usaremos en el plot

# =============================================================================
#                PLOTEAMOS LAS IMAGENES
# =============================================================================

for salida,real,inicial in zip (predict_plot[0:5],y_test_plot[0:5], x_test_plot[0:5]):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(14, 14))
    ax1.imshow(inicial,cmap=plt.cm.bone)#imagen perturbada con ruido
    ax1.set_title('Imagen perturbada con ruido')
    ax2.imshow(salida,cmap=plt.cm.bone)#salida autoencoder
    ax2.set_title('Salida del autoencoder')
    ax3.imshow(real,cmap=plt.cm.bone)#imagen real sin ruido
    ax3.set_title('Imagen real sin ruido')

# =============================================================================
#  BORRAMOS DATOS QUE NO NECESITAMOS Y LIBERAMOS ESPACIO
# =============================================================================

del (predict,predict_1d,predict_plot)
del (salida,real,inicial)
del modelo_3
gc.collect()


# =============================================================================
#                          MODELO 4 : PREDICCIONES Y R2
# =============================================================================


predict = modelo_4.predict(x_val)

predict_1d = predict.flatten()

error = r2_score(predict_1d, y_test_1d) # r2 = 0.96

predict_plot = predict.reshape(-1, 320,320) # Lo usaremos en el plot

# =============================================================================
#                PLOTEAMOS LAS IMAGENES
# =============================================================================

for salida,real,inicial in zip (predict_plot[0:5],y_test_plot[0:5], x_test_plot[0:5]):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(14, 14))
    ax1.imshow(inicial,cmap=plt.cm.bone)#imagen perturbada con ruido
    ax1.set_title('Imagen perturbada con ruido')
    ax2.imshow(salida,cmap=plt.cm.bone)#salida autoencoder
    ax2.set_title('Salida del autoencoder')
    ax3.imshow(real,cmap=plt.cm.bone)#imagen real sin ruido
    ax3.set_title('Imagen real sin ruido')










