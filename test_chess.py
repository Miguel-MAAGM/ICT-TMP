import cv2

# 1. Configuración inicial
# Reemplaza 'ruta/a/tu/imagen.jpg' con la ubicación real de tu foto
ruta_imagen = 'ALL.jpg' 

# Recuerda: Un tablero de 8x8 cuadrados tiene 7x7 esquinas internas.
columnas = 10
filas = 7

# 2. Cargar la imagena
imagen = cv2.imread(ruta_imagen)

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print(f"Error: No se pudo cargar la imagen desde '{ruta_imagen}'.")
    print("Verifica que la ruta y el nombre del archivo sean correctos.")
else:
    # 3. Procesamiento
    # Convertir a escala de grises para la detección
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Buscar las esquinas del tablero de ajedrez
    ret_corners, corners = cv2.findChessboardCorners(gray, (columnas, filas), None)

    # 4. Resultados
    if ret_corners:
        print("¡Tablero detectado con éxito!")
        
        # Refinar las coordenadas para mayor precisión (sub-píxel)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refinados = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Dibujar las esquinas detectadas sobre la imagen original
        cv2.drawChessboardCorners(imagen, (columnas, filas), corners_refinados, ret_corners)
    else:
        print("No se pudo detectar el patrón del tablero en la imagen.")
        print("Asegúrate de que el tablero esté bien iluminado, sin recortes y que los valores de filas/columnas sean correctos.")

    # 5. Mostrar la imagen final
    # Redimensionar la ventana si la foto es muy grande (opcional, quítalo si no lo necesitas)
    alto, ancho = imagen.shape[:2]
    if ancho > 1280 or alto > 720:
        imagen = cv2.resize(imagen, (int(ancho * 0.5), int(alto * 0.5)))

    cv2.imshow('Deteccion en Foto', imagen)
    
    print("Presiona cualquier tecla sobre la ventana de la imagen para cerrarla.")
    
    # waitKey(0) pausa el programa indefinidamente hasta que presiones una tecla
    cv2.waitKey(0)
    cv2.destroyAllWindows()