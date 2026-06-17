import cv2
import numpy as np
import json

# ==========================================
# 1. CONFIGURACIÓN INICIAL
# ==========================================
ruta_imagen = 'ALL.jpg'   # Reemplaza con el nombre de tu foto
ruta_json = 'test_1.json' # El archivo que me compartiste


# Dimensiones de las esquinas internas (ya corregido para tu foto)
columnas = 10
filas = 7

# IMPORTANTE: Esta es la medida real de un cuadrado en milímetros.
# He puesto 25.0 como ejemplo, DEBES cambiarlo por tu medida real.
tamano_cuadro_mm = 25.0 

# ==========================================
# 2. CARGAR PARÁMETROS INTRÍNSECOS
# ==========================================
with open(ruta_json, 'r') as f:
    datos_calibracion = json.load(f)

# Convertir las listas del JSON a arreglos (arrays) de NumPy que usa OpenCV
camera_matrix = np.array(datos_calibracion['camera_matrix'], dtype=np.float64)
dist_coeffs = np.array(datos_calibracion['distortion_coefficients'], dtype=np.float64)

# ==========================================
# 3. PREPARAR EL MODELO 3D
# ==========================================
# Creamos un sistema de coordenadas virtual donde las esquinas están en (0,0,0), (25,0,0), etc.
objp = np.zeros((filas * columnas, 3), np.float32)
objp[:, :2] = np.mgrid[0:columnas, 0:filas].T.reshape(-1, 2)
objp *= tamano_cuadro_mm

# ==========================================
# 4. DETECCIÓN EN LA IMAGEN
# ==========================================
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print("Error: No se pudo cargar la imagen.")
else:
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(gray, (columnas, filas), None)

    if ret_corners:
        print("¡Tablero detectado! Calculando distancia...")
        
        # Refinar esquinas 2D
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refinados = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # ==========================================
        # 5. CÁLCULO DE LA DISTANCIA (solvePnP)
        # ==========================================
        # Compara el modelo 3D (objp) con las esquinas 2D usando tu cámara
        ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners_refinados, camera_matrix, dist_coeffs)
        
        if ret_pnp:
            # El vector de traslación (tvec) tiene 3 ejes: X, Y, Z.
            # Z representa la distancia recta desde el lente de la cámara.
            distancia_z = tvec[2][0]
            
            print(f"-> Distancia estimada al tablero: {distancia_z:.2f} mm")
            
            # Dibujar resultados en la foto
            cv2.drawChessboardCorners(imagen, (columnas, filas), corners_refinados, ret_corners)
            texto = f"Distancia: {distancia_z:.1f} mm"
            cv2.putText(imagen, texto, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            
            # Redimensionar para poder verla bien en pantalla
            alto, ancho = imagen.shape[:2]
            imagen_mostrar = cv2.resize(imagen, (int(ancho * 0.5), int(alto * 0.5)))
            
            cv2.imshow('Calculo de Distancia 3D', imagen_mostrar)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No se pudo detectar el tablero. Revisa columnas y filas.")