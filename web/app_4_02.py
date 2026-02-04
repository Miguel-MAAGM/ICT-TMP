import os
import csv
from picamera2 import Picamera2
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
from datetime import datetime
import numpy as np
import json
import time
import serial
import sys
import threading

# ======================== CONFIGURACIÓN GENERAL ==========================
SERIAL_PORT = '/dev/ttyACM0'
BAUDRATE = 115200
ser = None

app = Flask(__name__)

# Carpetas del proyecto
MAPS_FOLDER = os.path.join("static", "modelos")
CALIB_FOLDER = os.path.join("static", "calibraciones")
CAPTURAS_FOLDER = os.path.join("static", "capturas")
COLOR_FOLDER = os.path.join("static", "calibraciones_color")
FOTOS_LOOP_FOLDER = os.path.join("static", "fotos_loop")

# Crear carpetas si no existen
os.makedirs(CAPTURAS_FOLDER, exist_ok=True)
os.makedirs(CALIB_FOLDER, exist_ok=True)
os.makedirs(FOTOS_LOOP_FOLDER, exist_ok=True)
os.makedirs(COLOR_FOLDER, exist_ok=True)
os.makedirs(MAPS_FOLDER, exist_ok=True)

# ----------------- CONFIGURACIÓN DE RESOLUCIONES -----------------
stream_resolution = {"width": 1280, "height": 720}
capture_resolution = {"width": 1920, "height": 1080}

# ----------------- INICIALIZACIÓN DE LA CÁMARA -----------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (stream_resolution["width"], stream_resolution["height"])}
))
picam2.start()

# ----------------- VARIABLES GLOBALES -----------------
streaming_active = True  # Control de streaming para optimizar durante escaneo
roi_zoom_level = 1.5

# Variables para Calibración de Cámara
calibration_images = []
objpoints = []
imgpoints = []
is_auto_calibrating = False
auto_calib_config = {"rows": 6, "cols": 7}
last_auto_capture_time = 0
MIN_TIME_BETWEEN_CAPTURES = 2.0

# Variables para Calibración de Color
color_thresholds = {
    "r_min": 0, "r_max": 255,
    "g_min": 0, "g_max": 255,
    "b_min": 0, "b_max": 255
}

# ======================== SISTEMA DE LOGS ==========================
SERIAL_LOG = []
LOG_BUFFER = []
MAX_LOG_LINES = 300

def add_serial_log(msg: str):
    SERIAL_LOG.append(msg)
    if len(SERIAL_LOG) > MAX_LOG_LINES:
        SERIAL_LOG.pop(0)

class LogCapture:
    """Captura logs de consola para mostrarlos en web"""
    def __init__(self, original_stream):
        self.original_stream = original_stream

    def write(self, message):
        self.original_stream.write(message)
        self.original_stream.flush()

        if message.strip():
            LOG_BUFFER.append(message.strip())
            if len(LOG_BUFFER) > MAX_LOG_LINES:
                LOG_BUFFER.pop(0)

    def flush(self):
        self.original_stream.flush()

# Redirigir stdout/stderr
sys.stdout = LogCapture(sys.stdout)
sys.stderr = LogCapture(sys.stderr)

# ======================== CONTROL DE ACCESO SERIAL (EVITA COLISIONES) ==========================
serial_lock = threading.Lock()

def safe_serial_write(line: str):
    """Escritura segura: no se mezcla con otras escrituras."""
    global ser
    if ser is None or not ser.is_open:
        return False
    try:
        with serial_lock:
            ser.write((line.strip() + "\n").encode("utf-8"))
            ser.flush()
        return True
    except Exception as e:
        print(f"⚠️ Error escribiendo en serial: {e}")
        return False

# ======================== SERIAL LISTENER ==========================
def serial_listener():
    """Thread que escucha mensajes de la Pico continuamente"""
    global ser
    print("🔵 Listener serial iniciado...")

    while True:
        try:
            if ser is None or not ser.is_open:
                time.sleep(0.1)
                continue

            # Lectura segura (evita interferencia con escrituras)
            with serial_lock:
                raw = ser.readline()

            if not raw:
                continue

            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except:
                continue

            if line == "" or len(line) < 2:
                continue

            msg = f"[SERIAL] {line}"
            print(msg)
            add_serial_log(msg)

            # ===================== LÓGICA DE FOTO AUTOMÁTICA =====================
            # La Pico imprime: "FOTO <angulo>"
            if line.startswith("FOTO"):
                try:
                    parts = line.split()
                    valor = float(parts[1]) if len(parts) > 1 else 0.0
                except:
                    valor = 0.0

                # 1) Capturar foto (alta resolución)
                success, frame_rgb = capture_high_res_frame()
                if success and frame_rgb is not None:
                    # OJO: Picamera2 puede venir en RGB; aquí guardamos tal cual (como tu original)
                    filename = f"foto_{valor:.2f}.jpg"
                    path = os.path.join(FOTOS_LOOP_FOLDER, filename)
                    cv2.imwrite(path, frame_rgb)

                    log = f"📸 FOTO GUARDADA: {filename}"
                    print(log)
                    add_serial_log(log)
                else:
                    print("⚠️ Falló la captura")
                    add_serial_log("⚠️ Falló la captura")

                # 2) Handshake: decirle a la Pico que continúe
                ok = safe_serial_write("SIGUIENTE")
                if ok:
                    print("🚀 -> COMANDO 'SIGUIENTE' ENVIADO")
                    add_serial_log("🚀 -> COMANDO 'SIGUIENTE' ENVIADO")

        except Exception as e:
            print(f"[ERROR SERIAL LISTENER] {e}")
            add_serial_log(f"[ERROR SERIAL LISTENER] {e}")
            time.sleep(0.1)

# ----------------- COMUNICACIÓN SERIAL -----------------
def send_serial_command(command: str):
    """
    Envía comando al Pico.
    Importante: NO intentamos leer respuesta aquí, porque el listener ya está leyendo.
    (Esto evita que se peleen por el puerto serial.)
    """
    global ser
    if ser is None or not ser.is_open:
        return "ERROR_SERIAL_OFFLINE"

    ok = safe_serial_write(command)
    if not ok:
        return "ERROR_SERIAL_WRITE"

    print(f"<- Comando enviado: {command}")
    add_serial_log(f"[APP] <- {command}")
    return "SENT"

# ----------------- FUNCIONES DE CÁMARA -----------------
def gen_frames():
    """Generador de frames para streaming de video"""
    global is_auto_calibrating, last_auto_capture_time, calibration_images, streaming_active

    # Placeholder cuando el streaming está pausado
    placeholder_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(placeholder_img, "ESCANEO EN PROCESO...", (350, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(placeholder_img, "VIDEO PAUSADO POR RENDIMIENTO", (280, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    _, buffer_placeholder = cv2.imencode('.jpg', placeholder_img)
    bytes_placeholder = buffer_placeholder.tobytes()

    while True:
        if not streaming_active:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytes_placeholder + b'\r\n')
            time.sleep(1.0)
            continue

        frame = picam2.capture_array()
        if frame is None:
            time.sleep(0.01)
            continue

        # Lógica calibración automática (mantengo tu lógica)
        if is_auto_calibrating:
            try:
                frame_bgr = frame
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                rows = auto_calib_config.get("rows", 6)
                cols = auto_calib_config.get("cols", 7)

                flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
                found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)

                if found:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(frame_bgr, (cols, rows), corners2, found)

                    if time.time() - last_auto_capture_time > MIN_TIME_BETWEEN_CAPTURES:
                        img_with_corners = frame_bgr.copy()
                        filename = f"calib_auto_{len(calibration_images)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        path = os.path.join(CAPTURAS_FOLDER, filename)
                        cv2.imwrite(path, img_with_corners)
                        calibration_images.append(path)
                        last_auto_capture_time = time.time()
                        print(f"✅ [AUTO] Captura guardada: {filename}")

                frame = frame_bgr
            except Exception as e:
                print(f"Error auto-calib: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_color_frames():
    """Generador para vista de calibración de color"""
    while True:
        frame = picam2.capture_array()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_rgb = frame
        mask = (frame_rgb[:, :, 0] >= color_thresholds["r_min"]) & (frame_rgb[:, :, 0] <= color_thresholds["r_max"]) & \
               (frame_rgb[:, :, 1] >= color_thresholds["g_min"]) & (frame_rgb[:, :, 1] <= color_thresholds["g_max"]) & \
               (frame_rgb[:, :, 2] >= color_thresholds["b_min"]) & (frame_rgb[:, :, 2] <= color_thresholds["b_max"])

        filtered = np.zeros_like(frame)
        filtered[mask] = [255, 255, 255]

        _, buffer = cv2.imencode('.jpg', filtered)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def capture_high_res_frame():
    """Captura una imagen en alta resolución"""
    try:
        picam2.stop()

        capture_config = picam2.create_still_configuration(
            main={"format": "RGB888", "size": (capture_resolution["width"], capture_resolution["height"])}
        )
        picam2.configure(capture_config)
        picam2.start()
        time.sleep(0.1)

        frame = picam2.capture_array()

    finally:
        # Volver a preview
        try:
            picam2.stop()
        except:
            pass
        picam2.configure(picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (stream_resolution["width"], stream_resolution["height"])}
        ))
        picam2.start()

    return frame is not None, frame

def gen_frames_roi():
    """Generador ROI con zoom digital"""
    global roi_zoom_level
    OUTPUT_SIZE = (1920, 1080)

    while True:
        frame = picam2.capture_array()
        if frame is None:
            time.sleep(0.01)
            continue

        try:
            h_img, w_img, _ = frame.shape
            center_x, center_y = w_img // 2, h_img // 2

            crop_w = int(w_img / roi_zoom_level)
            crop_h = int(h_img / roi_zoom_level)

            crop_w = max(1, min(crop_w, w_img))
            crop_h = max(1, min(crop_h, h_img))

            x1 = max(0, center_x - (crop_w // 2))
            y1 = max(0, center_y - (crop_h // 2))
            x2 = min(w_img, center_x + (crop_w // 2))
            y2 = min(h_img, center_y + (crop_h // 2))

            roi_frame = frame[y1:y2, x1:x2]
            if roi_frame.size > 0:
                roi_frame = cv2.resize(roi_frame, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)

            ret, buffer = cv2.imencode('.jpg', roi_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except:
            time.sleep(0.01)

# ======================== RUTAS PRINCIPALES ==========================
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/control')
def control():
    return render_template("control.html")

@app.route('/ajustes')
def ajustes():
    return render_template("ajustes.html")

# ======================== RUTAS DE VIDEO ==========================
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_color')
def video_feed_color():
    return Response(generate_color_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_roi')
def video_feed_roi():
    return Response(gen_frames_roi(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ======================== RUTAS DE CONTROL MOTOR ==========================
# Tu HTML hoy llama: /action/autohome, /action/startloopcapture, /action/capturefast, etc.
# Tu app original esperaba: auto_home, start_loop_capture, capture_fast, etc.
# => hacemos un normalizador para aceptar ambas.
def normalize_action(cmd: str) -> str:
    c = (cmd or "").strip().lower()
    c = c.replace("-", "_")
    return c

ACTION_MAP = {
    # variantes del HTML
    "autohome": "auto_home",
    "auto_home": "auto_home",
    "auto": "auto_home",

    "captureslow": "capture_slow",
    "capture_slow": "capture_slow",

    "capturemid": "capture_mid",
    "capture_mid": "capture_mid",

    "capturefast": "capture_fast",
    "capture_fast": "capture_fast",

    "startloopcapture": "start_loop_capture",
    "start_loop_capture": "start_loop_capture",
    "loop_capture": "start_loop_capture",

    # === NUEVO COMANDO AÑADIDO ===
    "sololoop": "solo_loop",
    "solo_loop": "solo_loop",

    "stop": "stop",
}

@app.route('/action/<cmd>', methods=['POST'])
def control_action(cmd):
    global streaming_active

    key = normalize_action(cmd)
    key = ACTION_MAP.get(key, key)

    if key == "auto_home":
        print("🏠 AUTO HOME")
        response = send_serial_command("AUTO_HOME")
        return jsonify({"success": True, "message": response})

    if key == "capture_slow":
        print("🐢 Velocidad LENTA")
        response = send_serial_command("CAPTURE_SLOW")
        return jsonify({"success": True, "message": response})

    if key == "capture_mid":
        print("🚶 Velocidad MEDIA")
        response = send_serial_command("CAPTURE_MID")
        return jsonify({"success": True, "message": response})

    if key == "capture_fast":
        print("🏃 Velocidad RÁPIDA")
        response = send_serial_command("CAPTURE_FAST")
        return jsonify({"success": True, "message": response})

    if key == "start_loop_capture":
        print("📸 LOOP CAPTURE (MANUAL) INICIADO")
        streaming_active = False
        print("🚫 Streaming pausado por rendimiento")
        response = send_serial_command("LOOP_CAPTURE")
        return jsonify({"success": True, "message": response})

    # === NUEVA LÓGICA PARA SOLO LOOP ===
    if key == "solo_loop":
        print("🔄 SOLO LOOP (AUTO) INICIADO")
        # También pausamos el streaming para dar prioridad al guardado de fotos
        streaming_active = False 
        print("🚫 Streaming pausado por rendimiento")
        response = send_serial_command("SOLO_LOOP")
        return jsonify({"success": True, "message": response})

    if key == "stop":
        print("⛔ STOP")
        streaming_active = True
        print("✅ Streaming reactivado")
        response = send_serial_command("STOP")
        return jsonify({"success": True, "message": response})

    return jsonify({"success": False, "message": f"Comando desconocido: {cmd}"}), 400

# ======================== RUTAS DE LOGS ==========================
@app.route('/serial_log')
def serial_log_page():
    return "<br>".join(SERIAL_LOG)

@app.route('/logs')
def view_logs():
    return render_template('logs.html')

@app.route('/api/get_logs')
def get_logs_api():
    return jsonify({"logs": LOG_BUFFER})

# Alias para tu control.html (ahora está pidiendo /api/getlogs)
@app.route('/api/getlogs')
def get_logs_api_alias():
    return jsonify({"logs": LOG_BUFFER})

# ======================== RUTAS DE CALIBRACIÓN DE CÁMARA ==========================
@app.route('/camera_setup')
def camera_setup():
    return render_template("camera_setup.html")

@app.route('/ajustes/camara')
def ajustes_camara():
    archivos = []
    if os.path.exists(CALIB_FOLDER):
        archivos = [f for f in os.listdir(CALIB_FOLDER) if f.endswith(".json")]
    return render_template("ajustes_camara.html", archivos=archivos)

@app.route('/ajustes/camara/nueva')
def nueva_calibracion():
    global calibration_images, objpoints, imgpoints
    calibration_images = []
    objpoints = []
    imgpoints = []
    return render_template("nueva_calibracion.html")

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
    data = request.get_json(force=True)
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

    print(f"✅ Resolución aplicada: {width}x{height}")
    return jsonify({
        "success": True,
        "message": f"Resolución de streaming cambiada a {width}x{height}",
        "actual_width": width,
        "actual_height": height
    })

@app.route('/camera/set_capture_resolution', methods=['POST'])
def set_capture_resolution():
    global capture_resolution
    data = request.get_json(force=True)
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

    filename = "preview_capture.jpg"
    path = os.path.join(CAPTURAS_FOLDER, filename)
    cv2.imwrite(path, frame)

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
    data = request.get_json(force=True)
    input_size = data.get('chessboard_size', [7, 6])

    print(f"📸 Capturando imagen de calibración. Tamaño: {input_size}")
    success, frame_rgb = capture_high_res_frame()

    if not success:
        return jsonify({"success": False, "message": "Error al capturar imagen"})

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    sizes_to_try = [
        tuple(input_size),
        (input_size[0] - 1, input_size[1] - 1)
    ]

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret = False
    corners = None
    final_size = None

    for size in sizes_to_try:
        if size[0] < 3 or size[1] < 3:
            continue
        print(f"🔍 Intentando detectar patrón de {size[0]}x{size[1]}...")
        ret, corners = cv2.findChessboardCorners(gray, size, flags)
        if ret:
            final_size = size
            print(f"✅ Patrón detectado con tamaño: {final_size}")
            break

    if ret:
        objp = np.zeros((final_size[0] * final_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:final_size[0], 0:final_size[1]].T.reshape(-1, 2)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)

        img_with_corners = frame_bgr.copy()
        cv2.drawChessboardCorners(img_with_corners, final_size, corners2, ret)

        filename = f"calib_{len(calibration_images)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(CAPTURAS_FOLDER, filename)
        cv2.imwrite(path, img_with_corners)
        calibration_images.append(path)

        return jsonify({
            "success": True,
            "message": f"Patrón detectado ({final_size[0]}x{final_size[1]}). Imágenes: {len(calibration_images)}",
            "images_count": len(calibration_images),
            "url": url_for('static', filename=f"capturas/{filename}")
        })

    print("❌ No se detectó el patrón")
    return jsonify({
        "success": False,
        "message": f"No se detectó el patrón. Probado con {sizes_to_try[0]} y {sizes_to_try[1]}"
    })

@app.route('/calibration/compute', methods=['POST'])
def compute_calibration():
    global objpoints, imgpoints, calibration_images

    if len(calibration_images) < 10:
        return jsonify({
            "success": False,
            "message": f"Se necesitan al menos 10 imágenes. Tienes {len(calibration_images)}."
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

    print(f"✅ Calibración completada. Error: {mean_error:.4f}")

    data = request.get_json(force=True)
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
    return jsonify({"success": True, "message": "Proceso de calibración reiniciado"})

@app.route('/calibration/status', methods=['GET'])
def calibration_status():
    image_urls = []
    for path in calibration_images:
        filename = os.path.basename(path)
        image_urls.append(url_for('static', filename=f"capturas/{filename}"))

    return jsonify({
        "images_captured": len(calibration_images),
        "ready_to_calibrate": len(calibration_images) >= 10,
        "is_auto_calibrating": is_auto_calibrating,
        "image_urls": image_urls
    })

@app.route('/calibration/autostart', methods=['POST'])
def start_auto_calibration():
    global is_auto_calibrating, auto_calib_config, last_auto_capture_time
    data = request.get_json(force=True)

    cols = int(data.get('cols', 7))
    rows = int(data.get('rows', 6))
    auto_calib_config = {"rows": rows, "cols": cols}
    is_auto_calibrating = True
    last_auto_capture_time = time.time()

    print(f"🚀 Iniciando calibración automática. Patrón: {cols}x{rows}")
    return jsonify({"success": True, "message": "Calibración automática iniciada"})

@app.route('/calibration/autostop', methods=['POST'])
def stop_auto_calibration():
    global is_auto_calibrating
    is_auto_calibrating = False
    print("🛑 Calibración automática detenida")
    return jsonify({"success": True, "message": "Calibración automática detenida"})

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

# ======================== RUTAS DE CALIBRACIÓN DE COLOR ==========================
@app.route("/ajustes/color")
def ajustes_color_list():
    if not os.path.exists(COLOR_FOLDER):
        os.makedirs(COLOR_FOLDER)
    archivos = [f for f in os.listdir(COLOR_FOLDER) if f.endswith(".json")]
    return render_template("ajustes_color.html", archivos=archivos)

@app.route("/ajustes/color/nueva")
def nueva_calibracion_color():
    return render_template("nueva_calibracion_color.html")

@app.route('/set_thresholds', methods=['POST'])
def set_thresholds():
    data = request.get_json(force=True)
    color_thresholds["r_min"] = int(data.get("r_min", 0))
    color_thresholds["r_max"] = int(data.get("r_max", 255))
    color_thresholds["g_min"] = int(data.get("g_min", 0))
    color_thresholds["g_max"] = int(data.get("g_max", 255))
    color_thresholds["b_min"] = int(data.get("b_min", 0))
    color_thresholds["b_max"] = int(data.get("b_max", 255))
    return jsonify(success=True, thresholds=color_thresholds)

@app.route('/capture', methods=['POST'])
def capture():
    print("Capturando imagen...")
    success, frame_rgb = capture_high_res_frame()

    if not success:
        return jsonify({"success": False})

    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(CAPTURAS_FOLDER, filename)
    cv2.imwrite(path, frame_rgb)

    return jsonify({
        "success": True,
        "url": url_for('static', filename=f"capturas/{filename}")
    })

# ======================== RUTAS DE VISUALIZACIÓN 3D ==========================
@app.route('/view')
def view_list():
    archivos = []
    if os.path.exists(MAPS_FOLDER):
        archivos = [f for f in os.listdir(MAPS_FOLDER) if f.endswith(".csv")]
    return render_template("view_list.html", archivos=archivos)

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

# ======================== RUTAS DE ROI/MONITOR ==========================
@app.route('/monitor_roi')
def monitor_roi():
    return render_template("roi_monitor.html")

@app.route('/set_roi_zoom', methods=['POST'])
def set_roi_zoom():
    global roi_zoom_level
    data = request.get_json(force=True)

    try:
        new_zoom = float(data.get('zoom', 1.0))
        roi_zoom_level = max(1.0, min(5.0, new_zoom))
        print(f"🔍 Zoom ROI cambiado a: {roi_zoom_level}x")
        return jsonify({"success": True, "zoom": roi_zoom_level})
    except ValueError:
        return jsonify({"success": False})

# ======================== INICIALIZACIÓN ==========================
if __name__ == '__main__':
    # Conectar al puerto serial
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        print(f"✅ Serial conectado en {SERIAL_PORT}")
        time.sleep(2)
    except Exception as e:
        print(f"❌ Error serial: {e}")
        ser = None

    # Iniciar thread listener
    listener = threading.Thread(target=serial_listener, daemon=True)
    listener.start()
    print("🔵 Hilo SERIAL LISTENER iniciado")

    # Ejecutar Flask
    try:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    except Exception as e:
        print(f"Error al iniciar la aplicación: {e}")
    finally:
        try:
            if picam2.started:
                picam2.stop()
        except:
            pass
        try:
            if ser is not None and ser.is_open:
                ser.close()
                print("Puerto serial cerrado.")
        except:
            pass
        cv2.destroyAllWindows()
