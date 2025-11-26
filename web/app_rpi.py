import os
import csv
# Importamos Picamera2 en lugar de cv2.VideoCapture
from picamera2 import Picamera2
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2  # Aún necesario para procesamiento de imágenes (cvtColor, imencode, etc.)
from datetime import datetime
import numpy as np
import json
import glob
import time # Necesario para la función de espera

app = Flask(__name__)

MAPS_FOLDER = os.path.join("static", "modelos")
CALIB_FOLDER = os.path.join("static", "calibraciones")
CAPTURAS_FOLDER = os.path.join("static", "capturas")
COLOR_FOLDER = os.path.join("static", "calibraciones_color")

os.makedirs(CAPTURAS_FOLDER, exist_ok=True)
os.makedirs(CALIB_FOLDER, exist_ok=True)

# ----------------- CONFIGURACIÓN DE RESOLUCIONES -----------------
# Usaremos estas resoluciones para configurar Picamera2
stream_resolution = {"width": 1280, "height": 720}  # Resolución para streaming
capture_resolution = {"width": 1920, "height": 1080}  # Resolución para captura

# ----------------- INICIALIZACIÓN DE LA CÁMARA RASPBERRY PI -----------------
picam2 = Picamera2()

# Configuramos la Picamera2 con la resolución de streaming
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (stream_resolution["width"], stream_resolution["height"])}
))
picam2.start()

color_thresholds = {
    "r_min": 0, "r_max": 255,
    "g_min": 0, "g_max": 255,
    "b_min": 0, "b_max": 255
}

# ----------------- VARIABLES GLOBALES PARA CONTROL -----------------
step_counter = 0
step_size = 1

# ----------------- VARIABLES PARA CALIBRACIÓN DE CÁMARA -----------------
calibration_images = []  # Almacena las imágenes capturadas para calibración
objpoints = []  # Puntos 3D en el espacio del mundo real
imgpoints = []  # Puntos 2D en el plano de la imagen

def gen_frames():
    """Genera frames desde Picamera2 con resolución de streaming"""
    while True:
        # Captura el frame como un array de numpy
        frame = picam2.capture_array()
        if frame is not None:
            # Picamera2 por defecto captura en RGB888, cv2 espera BGR para imencode
            # Aunque capture_array en el preview config con RGB888 devuelve RGB,
            # el código original de rpi no hacía conversión explícita antes de imencode.
            # Lo más seguro es que el imencode funcione bien con el array RGB, pero
            # si el color parece incorrecto en el streaming, se podría añadir:
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_color_frames():
    """Genera frames con filtro de color desde Picamera2"""
    while True:
        # Captura el frame como un array de numpy (RGB)
        frame = picam2.capture_array()
        if frame is None:
            break
        
        # El frame de Picamera2 ya está en RGB888 (3 canales), por lo que no necesitamos cvtColor para RGB
        frame_rgb = frame 
        
        # Máscara: True si está dentro de los umbrales
        mask = (frame_rgb[:,:,0] >= color_thresholds["r_min"]) & (frame_rgb[:,:,0] <= color_thresholds["r_max"]) & \
               (frame_rgb[:,:,1] >= color_thresholds["g_min"]) & (frame_rgb[:,:,1] <= color_thresholds["g_max"]) & \
               (frame_rgb[:,:,2] >= color_thresholds["b_min"]) & (frame_rgb[:,:,2] <= color_thresholds["b_max"])
        
        # Crear imagen binaria (negro/fondo, blanco/umbral)
        filtered = np.zeros_like(frame)
        filtered[mask] = [255, 255, 255]
        
        # Codificamos a JPG
        _, buffer = cv2.imencode('.jpg', filtered)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def capture_high_res_frame():
    """Captura un frame en alta resolución usando Picamera2"""
    
    # Detener la configuración de streaming
    picam2.stop()
    
    # Crear una configuración de captura con la resolución deseada
    capture_config = picam2.create_still_configuration(
        main={"format": "RGB888", "size": (capture_resolution["width"], capture_resolution["height"])}
    )
    
    # Aplicar la configuración de captura
    picam2.configure(capture_config)
    picam2.start()
    
    # Esperar un poco para que la cámara se ajuste
    time.sleep(0.1)
    
    # Capturar frame como un array numpy
    frame = picam2.capture_array()
    
    # Detener y restaurar configuración de streaming
    picam2.stop()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (stream_resolution["width"], stream_resolution["height"])}
    ))
    picam2.start()
    
    return frame is not None, frame

# ----------------- RUTAS DE CONFIGURACIÓN DE CÁMARA -----------------

@app.route('/camera_setup')
def camera_setup():
    """Página de configuración de resoluciones de cámara"""
    return render_template("camera_setup.html")

@app.route('/camera/get_resolutions', methods=['GET'])
def get_resolutions():
    """Obtiene las resoluciones actuales"""
    return jsonify({
        "success": True,
        "stream_width": stream_resolution["width"],
        "stream_height": stream_resolution["height"],
        "capture_width": capture_resolution["width"],
        "capture_height": capture_resolution["height"]
    })

@app.route('/camera/set_stream_resolution', methods=['POST'])
def set_stream_resolution():
    """Cambia la resolución de streaming"""
    global stream_resolution
    
    data = request.get_json()
    width = int(data.get('width', 1280))
    height = int(data.get('height', 720))
    
    print(f"🎥 Cambiando resolución de streaming a {width}x{height}")
    
    # Guardar en variables globales
    stream_resolution["width"] = width
    stream_resolution["height"] = height
    
    # Detener, configurar y reiniciar Picamera2
    picam2.stop()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (width, height)}
    ))
    picam2.start()
    
    # Picamera2 es determinista, la resolución aplicada es la configurada.
    actual_width = width
    actual_height = height
    
    print(f"✅ Resolución aplicada: {actual_width}x{actual_height}")
    
    return jsonify({
        "success": True,
        "message": f"Resolución de streaming cambiada a {actual_width}x{actual_height}",
        "actual_width": actual_width,
        "actual_height": actual_height
    })

@app.route('/camera/set_capture_resolution', methods=['POST'])
def set_capture_resolution():
    """Cambia la resolución de captura"""
    global capture_resolution
    
    data = request.get_json()
    width = int(data.get('width', 1920))
    height = int(data.get('height', 1080))
    
    print(f"📸 Resolución de captura configurada a {width}x{height}")
    
    capture_resolution["width"] = width
    capture_resolution["height"] = height
    
    return jsonify({
        "success": True,
        "message": f"Resolución de captura configurada a {width}x{height}"
    })

@app.route('/camera/test_capture', methods=['POST'])
def test_capture():
    """Realiza una captura de prueba con la resolución de captura"""
    print("📷 Realizando captura de prueba...")
    
    success, frame = capture_high_res_frame()
    
    if not success:
        return jsonify({"success": False, "message": "Error al capturar imagen"})
    
    # Convertir de RGB a BGR para guardar con cv2.imwrite
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    filename = "preview_capture.jpg"
    path = os.path.join(CAPTURAS_FOLDER, filename)
    cv2.imwrite(path, frame_bgr)
    
    actual_height, actual_width = frame.shape[:2]
    
    print(f"✅ Captura de prueba guardada: {actual_width}x{actual_height}")
    
    return jsonify({
        "success": True,
        "url": url_for('static', filename=f"capturas/{filename}"),
        "width": actual_width,
        "height": actual_height
    })

# ----------------- NUEVAS FUNCIONES DE CALIBRACIÓN -----------------

@app.route('/calibration/capture', methods=['POST'])
def capture_calibration_image():
    """Captura una imagen para calibración y detecta el patrón de tablero de ajedrez"""
    global calibration_images, objpoints, imgpoints
    
    data = request.get_json()
    chessboard_size = data.get('chessboard_size', [7, 6])  # Por defecto 7x6 esquinas internas
    
    print(f"📸 Capturando imagen de calibración (patrón {chessboard_size[0]}x{chessboard_size[1]})...")
    
    # Usar la función de captura en alta resolución
    success, frame_rgb = capture_high_res_frame()
    
    if not success:
        return jsonify({"success": False, "message": "Error al capturar imagen"})
    
    # Convertir de RGB a BGR (cv2 lo maneja mejor)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Buscar las esquinas del tablero de ajedrez
    ret, corners = cv2.findChessboardCorners(gray, tuple(chessboard_size), None)
    
    if ret:
        # Preparar puntos del objeto (0,0,0), (1,0,0), (2,0,0) ... (6,5,0)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        # Refinar las esquinas con mayor precisión
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Guardar puntos
        objpoints.append(objp)
        imgpoints.append(corners2)
        
        # Dibujar las esquinas en la imagen
        img_with_corners = frame_bgr.copy()
        cv2.drawChessboardCorners(img_with_corners, tuple(chessboard_size), corners2, ret)
        
        # Guardar imagen con esquinas detectadas
        filename = f"calib_{len(calibration_images)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(CAPTURAS_FOLDER, filename)
        cv2.imwrite(path, img_with_corners)
        
        calibration_images.append(path)
        
        print(f"✅ Patrón detectado correctamente. Total de imágenes: {len(calibration_images)}")
        
        return jsonify({
            "success": True,
            "message": f"Patrón detectado. Imágenes capturadas: {len(calibration_images)}",
            "images_count": len(calibration_images),
            "url": url_for('static', filename=f"capturas/{filename}")
        })
    else:
        print("❌ No se pudo detectar el patrón de tablero de ajedrez")
        return jsonify({
            "success": False,
            "message": "No se detectó el patrón de tablero. Asegúrate de que el tablero esté completamente visible."
        })

@app.route('/calibration/compute', methods=['POST'])
def compute_calibration():
    """Calcula los parámetros de calibración de la cámara"""
    global objpoints, imgpoints, calibration_images
    
    if len(calibration_images) < 10:
        return jsonify({
            "success": False,
            "message": f"Se necesitan al menos 10 imágenes para una buena calibración. Tienes {len(calibration_images)}."
        })
    
    print(f"🔧 Calculando calibración con {len(calibration_images)} imágenes...")
    
    # Obtener dimensiones de la imagen (usar resolución de captura)
    success, frame_rgb = capture_high_res_frame()
    if not success:
        return jsonify({"success": False, "message": "Error al acceder a la cámara"})
    
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]
    
    # Realizar calibración
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    if not ret:
        return jsonify({"success": False, "message": "Error en el cálculo de calibración"})
    
    # Calcular error de reproyección
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error = mean_error / len(objpoints)
    
    print(f"✅ Calibración completada. Error medio de reproyección: {mean_error:.4f}")
    
    # Guardar resultados en formato JSON
    data = request.get_json()
    calibration_name = data.get('name', f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    calibration_data = {
        "name": calibration_name,
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.tolist(),
        "reprojection_error": mean_error,
        "num_images": len(calibration_images),
        "image_size": list(img_shape)
    }
    
    # Guardar en archivo JSON
    filename = f"{calibration_name}.json"
    filepath = os.path.join(CALIB_FOLDER, filename)
    
    with open(filepath, 'w') as f:
        json.dump(calibration_data, f, indent=4)
    
    print(f"💾 Calibración guardada en: {filepath}")
    
    return jsonify({
        "success": True,
        "message": f"Calibración completada exitosamente. Error: {mean_error:.4f}",
        "calibration_name": calibration_name,
        "reprojection_error": mean_error,
        "num_images": len(calibration_images)
    })

@app.route('/calibration/reset', methods=['POST'])
def reset_calibration():
    """Reinicia el proceso de calibración"""
    global calibration_images, objpoints, imgpoints
    
    print("🔄 Reiniciando proceso de calibración...")
    
    calibration_images = []
    objpoints = []
    imgpoints = []
    
    return jsonify({
        "success": True,
        "message": "Proceso de calibración reiniciado"
    })

@app.route('/calibration/status', methods=['GET'])
def calibration_status():
    """Devuelve el estado actual de la calibración"""
    return jsonify({
        "images_captured": len(calibration_images),
        "ready_to_calibrate": len(calibration_images) >= 10
    })

# ----------------- RUTAS DE CONTROL -----------------

@app.route('/action/<cmd>', methods=['POST'])
def control_action(cmd):
    global step_counter
    
    if cmd == "left":
        print(f"🟢 Botón izquierda presionado (paso: {step_size})")
        step_counter -= step_size
    
    elif cmd == "right":
        print(f"🟢 Botón derecha presionado (paso: {step_size})")
        step_counter += step_size
    
    elif cmd == "capture":
        print("🟢 Botón capture presionado")
        # Capturar con resolución de captura
        success, frame_rgb = capture_high_res_frame()
        if success:
            # Convertir a BGR para cv2.imwrite
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = os.path.join(CAPTURAS_FOLDER, filename)
            cv2.imwrite(path, frame_bgr)
            
            return jsonify({
                "success": True,
                "step": step_counter,
                "url": url_for('static', filename=f"capturas/{filename}")
            })
        return jsonify({"success": False, "step": step_counter})
    
    elif cmd == "start_loop":
        print("🟢 Botón start loop presionado")
    
    return jsonify({"success": True, "step": step_counter})

@app.route('/set_step_size', methods=['POST'])
def set_step_size_route():
    global step_size
    data = request.get_json()
    step_size = int(data.get('step_size', 1))
    print(f"🔧 Tamaño de paso configurado: {step_size}")
    return jsonify({'status': 'success', 'step_size': step_size})

# ----------------- RUTAS EXISTENTES -----------------

@app.route('/video_feed_color')
def video_feed_color():
    return Response(generate_color_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_thresholds', methods=['POST'])
def set_thresholds():
    data = request.json
    color_thresholds["r_min"] = int(data.get("r_min", 0))
    color_thresholds["r_max"] = int(data.get("r_max", 255))
    color_thresholds["g_min"] = int(data.get("g_min", 0))
    color_thresholds["g_max"] = int(data.get("g_max", 255))
    color_thresholds["b_min"] = int(data.get("b_min", 0))
    color_thresholds["b_max"] = int(data.get("b_max", 255))
    return jsonify(success=True, thresholds=color_thresholds)

@app.route("/ajustes/color")
def ajustes_color_list():
    if not os.path.exists(COLOR_FOLDER):
        os.makedirs(COLOR_FOLDER)
    archivos = [f for f in os.listdir(COLOR_FOLDER) if f.endswith(".json")]
    return render_template("ajustes_color.html", archivos=archivos)

@app.route("/ajustes/color/nueva")
def nueva_calibracion_color():
    return render_template("nueva_calibracion_color.html")

@app.route('/capture', methods=['POST'])
def capture():
    print("Capturando imagen...")
    # Usar la captura de alta resolución
    success, frame_rgb = capture_high_res_frame()
    if not success:
        return jsonify({"success": False})
    
    # Convertir a BGR para cv2.imwrite
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(CAPTURAS_FOLDER, filename)
    cv2.imwrite(path, frame_bgr)
    
    return jsonify({
        "success": True,
        "url": url_for('static', filename=f"capturas/{filename}")
    })

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/ajustes/camara/cancelar")
def cancelar_calibracion():
    folder = CAPTURAS_FOLDER
    if os.path.exists(folder):
        for file in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, file))
            except Exception as e:
                print(f"Error eliminando {file}: {e}")
    return redirect(url_for("ajustes"))

@app.route('/control')
def control():
    return render_template("control.html")

@app.route('/view')
def view_list():
    archivos = []
    if os.path.exists(MAPS_FOLDER):
        archivos = [f for f in os.listdir(MAPS_FOLDER) if f.endswith(".csv")]
    return render_template("view_list.html", archivos=archivos)

@app.route('/ajustes')
def ajustes():
    return render_template("ajustes.html")

@app.route('/ajustes/camara/nueva')
def nueva_calibracion():
    # Reiniciar la lista de imágenes de calibración al entrar a esta ruta
    global calibration_images, objpoints, imgpoints
    calibration_images = []
    objpoints = []
    imgpoints = []
    return render_template("nueva_calibracion.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ajustes/camara')
def ajustes_camara():
    archivos = []
    if os.path.exists(CALIB_FOLDER):
        archivos = [f for f in os.listdir(CALIB_FOLDER) if f.endswith(".json")]
    return render_template("ajustes_camara.html", archivos=archivos)

@app.route('/view/<nombre>')
def view_plot(nombre):
    ruta = os.path.join(MAPS_FOLDER, nombre)
    x, y, z = [], [], []
    
    if os.path.exists(ruta):
        with open(ruta) as f:
            reader = csv.DictReader(f)
            for row in reader:
                x.append(float(row["x"]))
                y.append(float(row["y"]))
                z.append(float(row["z"]))
    
    return render_template("view_plot.html", nombre=nombre, x=x, y=y, z=z)

if __name__ == '__main__':
    try:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    except Exception as e:
        print(f"Error al iniciar la aplicación: {e}")
    finally:
        # Detener la cámara al cerrar
        if picam2.started:
            picam2.stop()
        cv2.destroyAllWindows()