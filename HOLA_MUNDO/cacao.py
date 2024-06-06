import cv2
import numpy as np

# Cargar la imagen del cacao
imagen = cv2.imread('imagen_cacao.jpg')

# Convertir la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplicar umbralización binaria para separar el cacao del fondo
_, imagen_binaria = cv2.threshold(imagen_gris, 127, 255, cv2.THRESH_BINARY)

# Encontrar contornos en la imagen binaria
contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Recorrer los contornos y dibujarlos en la imagen original
for contorno in contornos:
  cv2.drawContours(imagen, [contorno], -1, (0, 255, 0), 2)

# Mostrar la imagen con los contornos dibujados
cv2.imshow('Imagen con contornos', imagen)
cv2.waitKey(0)
print("esto es lo que acabé de subir ")
# Clasificar los contornos para identificar plagas o enfermedades
# (Esta parte del código se debe implementar utilizando un modelo de aprendizaje automático entrenado para reconocer plagas y enfermedades del cacao)

# Mostrar la imagen con las plagas o enfermedades identificadas
# (Esta parte del código se debe implementar para mostrar la imagen con las plagas o enfermedades identificadas)
