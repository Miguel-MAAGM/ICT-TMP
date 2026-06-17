import cv2
import numpy as np
# Escalado para visualización en 1080p
dimensiones_1080p = (1920, 1080)
# 1. Cargar la imagen
img = cv2.imread('preview_capture.jpg')

# 2. Convertir al espacio de color HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 3. Dividir los canales HSV (Necesitamos 'v' además de 's')
h, s, v = cv2.split(hsv) 

# 4. Máscara de Color (Rojo por Alta Saturación)
baja_rojo1 = np.array([0, 150, 50])    
alta_rojo1 = np.array([10, 255, 255])
baja_rojo2 = np.array([170, 150, 50])  
alta_rojo2 = np.array([180, 255, 255])

mascara1 = cv2.inRange(hsv, baja_rojo1, alta_rojo1)
mascara2 = cv2.inRange(hsv, baja_rojo2, alta_rojo2)
mascara_color = cv2.add(mascara1, mascara2)

# --- NUEVO: DETECTAR EL NÚCLEO QUEMADO (BLANCO) ---
# Buscamos píxeles donde el brillo (V) sea casi máximo (ej. > 240) 
# y la saturación (S) sea muy baja (ej. < 40) debido al exceso de luz.
baja_blanco = np.array([0, 0, 240])     # Matiz da igual, Sat baja, Brillo muy alto
alta_blanco = np.array([180, 40, 255])   # Captura el blanco puro del centro

mascara_nucleo = cv2.inRange(hsv, baja_blanco, alta_blanco)

cv2.imshow('Mascara Color (Rojo)', cv2.resize(mascara_color, dimensiones_1080p, interpolation=cv2.INTER_AREA))
cv2.imshow('Mascara Nucleo Quemado (Blanco)', cv2.resize(mascara_nucleo, dimensiones_1080p, interpolation=cv2.INTER_AREA))
# --- NUEVO: COMBINAR AMBAS MÁSCARAS ---
# Usamos un OR lógico: si el píxel es rojo vivo O es blanco quemado, se incluye.
mascara_completa = cv2.bitwise_or(mascara_color, mascara_nucleo)
# --------------------------------------------------


s_escalada = cv2.resize(s, dimensiones_1080p, interpolation=cv2.INTER_AREA)
mascara_escalada = cv2.resize(mascara_completa, dimensiones_1080p, interpolation=cv2.INTER_AREA)

# Mostrar resultados
cv2.imshow('Canal S (Veras el centro oscuro)', s_escalada)
cv2.imshow('Mascara Laser Con Nucleo Relleno (1080p)', mascara_escalada)
cv2.waitKey(0)
cv2.destroyAllWindows()