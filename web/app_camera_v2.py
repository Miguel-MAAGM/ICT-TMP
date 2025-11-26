import os
import csv
from picamera2 import Picamera2
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
from datetime import datetime
import numpy as np
import json
import glob
import time 
import serial 

# ======================== CONFIGURACIÓN GENERAL ==========================
# Reemplaza '/dev/ttyACM0' por el puerto correcto de tu Pico si es diferente.
SERIAL_PORT = '/dev/ttyACM0' 
BAUDRATE = 115200 
ser = None # Inicializado a None globalmente

app = Flask(__name__)

MAPS_FOLDER = os.path.join("static", "modelos")
CALIB_FOLDER = os.path.join("static", "calibraciones")
CAPTURAS_FOLDER = os.path.join("static", "capturas")
COLOR_FOLDER = os.path.join("static", "calibraciones_color")

os.makedirs(CAPTURAS_FOLDER, exist_ok=True)
os.makedirs(CALIB_FOLDER, exist_ok=True)

# ----------------- CONFIGURACIÓN DE RESOLUCIONES -----------------
stream_resolution = {"width": 1280, "height": 720} 
capture_resolution = {"width": 1920, "height": 1080} 

# ----------------- INICIALIZACIÓN DE LA CÁMARA RASPBERRY PI -----------------
# Es importante inicializar picam2 aquí para que las funciones lo usen
picam2 = Picamera2()
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
step_size = 20 

# ----------------- FUNCIONES DE COMUNICACIÓN SERIAL -----------------

def send_serial_command(command, wait_for_angle=False, wait_for_ok=False, timeout=3.0):
    global ser

    """Envía un comando al Pico y, opcionalmente, espera:
       - una línea 'ANGULO: xx.xx' (wait_for_angle=True)
       - una línea 'OK' o 'ERROR_...' (wait_for_ok=True)
    """
    if ser is None:
        return "ERROR_SERIAL_OFFLINE"

    try:
        # Limpia el buffer de entrada para no leer basura vieja
        ser.reset_input_buffer()

        # Envía el comando
        ser.write((command + "\n").encode("utf-8"))
        print(f"<- Comando enviado: {command}")

        start_time = time.time()
        last_line = ""

        while True:
            line_bytes = ser.readline()
            if not line_bytes:
                # Timeout parcial, revisamos si ya pasó el tiempo máximo
                if time.time() - start_time > timeout:
                    print("⚠️ Timeout esperando respuesta del Pico")
                    break
                continue

            line = line_bytes.decode("utf-8", errors="ignore").strip()
            if line == "":
                continue

            print(f"-> Respuesta recibida: {line}")
            last_line = line

            # Si estamos esperando "OK" o un error específico
            if wait_for_ok and (line == "OK" or line.startswith("ERROR_")):
                return line

            # Si estamos esperando un ángulo
            if wait_for_angle and line.startswith("ANGULO:"):
                return line

            # Si no esperamos nada especial, con la primera línea nos basta
            if not wait_for_ok and not wait_for_angle:
                return line

            # Si seguimos esperando algo concreto, continúa leyendo hasta timeout

        # Si salimos por timeout, devolvemos lo último que vimos (si hay algo)
        return last_line or "NO_RESPONSE"

    except Exception as e:
        print(f"⚠️ Error en comunicación serial: {e}")
        try:
            ser.close()
        except:
            pass
        ser = None
        return f"ERROR: {e}"



# ----------------- FUNCIONES DE CÁMARA (sin cambios) -----------------

def gen_frames():
    while True:
        frame = picam2.capture_array()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_color_frames():
    while True:
        frame = picam2.capture_array()
        if frame is None:
            break
        frame_rgb = frame 
        mask = (frame_rgb[:,:,0] >= color_thresholds["r_min"]) & (frame_rgb[:,:,0] <= color_thresholds["r_max"]) & \
               (frame_rgb[:,:,1] >= color_thresholds["g_min"]) & (frame_rgb[:,:,1] <= color_thresholds["g_max"]) & \
               (frame_rgb[:,:,2] >= color_thresholds["b_min"]) & (frame_rgb[:,:,2] <= color_thresholds["b_max"])
        filtered = np.zeros_like(frame)
        filtered[mask] = [255, 255, 255]
        _, buffer = cv2.imencode('.jpg', filtered)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def capture_high_res_frame():
    picam2.stop()
    capture_config = picam2.create_still_configuration(
        main={"format": "RGB888", "size": (capture_resolution["width"], capture_resolution["height"])}
    )
    picam2.configure(capture_config)
    picam2.start()
    time.sleep(0.1)
    frame = picam2.capture_array()
    picam2.stop()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (stream_resolution["width"], stream_resolution["height"])}
    ))
    picam2.start()
    return frame is not None, frame

# ----------------- RUTAS DE FLASK -----------------

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/camera_setup')
def camera_setup():
    return render_template("camera_setup.html")

@app.route('/camera/get_resolutions', methods=['GET'])
def get_resolutions():
    return jsonify({
        "success": True,
        "stream_width": stream_resolution["width"],
        "stream_height": stream_resolution["height"],
        "capture_width": capture_resolution["width"],
        "capture_height": capture_resolution["height"]
    })

@app.route('/camera/set_stream_resolution', methods=['POST'])
def set_stream_resolution():
    global stream_resolution
    data = request.get_json()
    width = int(data.get('width', 1280))
    height = int(data.get('height', 720))
    print(f"🎥 Cambiando resolución de streaming a {width}x{height}")
    stream_resolution["width"] = width
    stream_resolution["height"] = height
    picam2.stop()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (width, height)}
    ))
    picam2.start()
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
    print("📷 Realizando captura de prueba...")
    success, frame = capture_high_res_frame()
    if not success:
        return jsonify({"success": False, "message": "Error al capturar imagen"})
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

@app.route('/calibration/capture', methods=['POST'])
def capture_calibration_image():
    global calibration_images, objpoints, imgpoints
    data = request.get_json()
    chessboard_size = data.get('chessboard_size', [7, 6])
    print(f"📸 Capturando imagen de calibración (patrón {chessboard_size[0]}x{chessboard_size[1]})...")
    success, frame_rgb = capture_high_res_frame()
    if not success:
        return jsonify({"success": False, "message": "Error al capturar imagen"})
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, tuple(chessboard_size), None)
    if ret:
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        img_with_corners = frame_bgr.copy()
        cv2.drawChessboardCorners(img_with_corners, tuple(chessboard_size), corners2, ret)
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
    global objpoints, imgpoints, calibration_images
    if len(calibration_images) < 10:
        return jsonify({
            "success": False,
            "message": f"Se necesitan al menos 10 imágenes para una buena calibración. Tienes {len(calibration_images)}."
        })
    print(f"🔧 Calculando calibración con {len(calibration_images)} imágenes...")
    success, frame_rgb = capture_high_res_frame()
    if not success:
        return jsonify({"success": False, "message": "Error al acceder a la cámara"})
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    if not ret:
        return jsonify({"success": False, "message": "Error en el cálculo de calibración"})
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error = mean_error / len(objpoints)
    print(f"✅ Calibración completada. Error medio de reproyección: {mean_error:.4f}")
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
    return jsonify({
        "images_captured": len(calibration_images),
        "ready_to_calibrate": len(calibration_images) >= 10
    })

@app.route('/action/<cmd>', methods=['POST'])
def control_action(cmd):
    global step_counter
    
    if cmd == "left":
        print(f"🟢 Botón izquierda presionado (paso: {step_size} grados)")
        # AHORA esperamos directamente el ANGULO final que manda mover_grados()
        angle_response = send_serial_command("LEFT", wait_for_angle=True)
        response = angle_response

        if angle_response.startswith("ANGULO:"):
            step_counter = float(angle_response.split(":")[1].strip())
        else:
            step_counter = -1

        return jsonify({"success": True, "step": step_counter, "message": response})
        
    elif cmd == "right":
        print(f"🟢 Botón derecha presionado (paso: {step_size} grados)")
        angle_response = send_serial_command("RIGHT", wait_for_angle=True)
        response = angle_response

        if angle_response.startswith("ANGULO:"):
            step_counter = float(angle_response.split(":")[1].strip())
        else:
            step_counter = -1

        return jsonify({"success": True, "step": step_counter, "message": response})

    elif cmd == "capture":
        print("🟢 Botón capture presionado")
        # Pedimos el ángulo actual usando READ y esperamos ANGULO:
        angle_response = send_serial_command("READ", wait_for_angle=True)
        current_step = step_counter

        if angle_response.startswith("ANGULO:"):
            current_step = float(angle_response.split(":")[1].strip())

        success, frame_rgb = capture_high_res_frame()
        if success:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            filename = f"capture_{current_step:.2f}deg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = os.path.join(CAPTURAS_FOLDER, filename)
            cv2.imwrite(path, frame_bgr)
            
            return jsonify({
                "success": True,
                "step": current_step,
                "url": url_for('static', filename=f"capturas/{filename}")
            })
        return jsonify({"success": False, "step": current_step})
    
    elif cmd == "start_loop":
        print("🟢 Botón start loop presionado")
        # Para LOOP no esperamos ANGULO, solo un primer mensaje
        response = send_serial_command("LOOP")
        return jsonify({"success": True, "step": step_counter, "message": response})
        
    elif cmd == "stop":
        print("🔴 Botón STOP presionado")
        # STOP en la Pico termina imprimiendo ANGULO: xx.xx al final
        angle_response = send_serial_command("STOP", wait_for_angle=True)
        response = angle_response

        if angle_response.startswith("ANGULO:"):
            step_counter = float(angle_response.split(":")[1].strip())

        return jsonify({"success": True, "step": step_counter, "message": response})
        
    return jsonify({"success": True, "step": step_counter})



@app.route('/set_step_size', methods=['POST'])
def set_step_size_route():
    global step_size
    data = request.get_json()
    step_size = int(data.get('step_size', 20))
    print(f"🔧 Tamaño de paso configurado (grados): {step_size}")
    response = send_serial_command(f"SET_ANGLE {step_size}", wait_for_ok=True)
    return jsonify({'status': 'success', 'step_size': step_size, 'message': response})


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
    success, frame_rgb = capture_high_res_frame()
    if not success:
        return jsonify({"success": False})
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(CAPTURAS_FOLDER, filename)
    cv2.imwrite(path, frame_bgr)
    return jsonify({
        "success": True,
        "url": url_for('static', filename=f"capturas/{filename}")
    })

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
    # ======================== INICIALIZACIÓN SERIAL SEGURO (MOVIMIENTO) ==========================
    # La conexión serial se realiza justo antes de ejecutar Flask. 
    # Si falla, ser es None, pero Flask se inicia.
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1) 
        print(f"✅ Conexión serial establecida en {SERIAL_PORT} a {BAUDRATE} baudios.")
        time.sleep(2) # Esperar a que la Pico se reinicie
    except serial.SerialException as e:
        print(f"❌ Error al conectar al puerto serial {SERIAL_PORT}: {e}")
        print("La aplicación web se iniciará, pero las funciones de motor no funcionarán.")
        ser = None
    # =================================================================================
    
    try:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    except Exception as e:
        print(f"Error al iniciar la aplicación: {e}")
    finally:
        # Detener la cámara al cerrar
        if picam2.started:
            picam2.stop()
        if ser is not None and ser.is_open:
            ser.close()
            print("Puerto serial cerrado.")
        cv2.destroyAllWindows()