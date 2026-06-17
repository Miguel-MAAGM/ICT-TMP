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
from laser_detector import detect_laser, build_preview_jpeg, build_heatmap_jpeg, laser_score
from influx_db import InfluxReader, InfluxWriter

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
PLANO_FOLDER = os.path.join("static", "calibraciones_plano")
EJE_FOLDER = os.path.join("static", "calibraciones_eje")

# Crear carpetas si no existen
os.makedirs(CAPTURAS_FOLDER, exist_ok=True)
os.makedirs(CALIB_FOLDER, exist_ok=True)
os.makedirs(FOTOS_LOOP_FOLDER, exist_ok=True)
os.makedirs(COLOR_FOLDER, exist_ok=True)
os.makedirs(MAPS_FOLDER, exist_ok=True)
os.makedirs(PLANO_FOLDER, exist_ok=True)
os.makedirs(EJE_FOLDER, exist_ok=True)

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
perfil_activo = {
    "camara": None,   # nombre del archivo JSON de calibración de cámara seleccionado
    "laser": None,    # nombre del archivo JSON de calibración de detección del laser
    "plano": None,    # nombre del archivo JSON del plano del laser seleccionado
    "eje": None       # nombre del archivo JSON del eje de giro seleccionado
}

# Variables para Calibración del Plano del Láser (triangulación con chessboard)
plane_calib = {
    "points": [],   # lista de arrays Nx3 con puntos 3D sobre el plano del láser
    "images": [],   # rutas de los overlays guardados por pose
    "counts": [],   # nº de píxeles de láser detectados por pose
}
plane_config = {
    "cols": 7, "rows": 6, "square_mm": 25.0,
}
MIN_PLANE_CAPTURES = 5

# Variables para Calibración del EJE DE GIRO (cámara móvil).
# En cada pose guardamos la rotación/traslación tablero→cámara (solvePnP). Como la
# cámara gira rígidamente alrededor del eje, las rotaciones relativas entre poses
# comparten el mismo eje, del que recuperamos punto y dirección en frame cámara.
eje_calib = {
    "R": [],        # lista de matrices 3x3 (tablero→cámara) por pose
    "t": [],        # lista de vectores 3 (tablero→cámara) por pose
    "images": [],   # overlays guardados
}
eje_config = {
    "cols": 7, "rows": 6, "square_mm": 25.0,
}
MIN_EJE_CAPTURES = 4

# Variables para el TEST DE BARRIDO (nube de puntos rotacional)
# Geometría del eje: pasos del motor → grados del plato (a través de la reductora).
barrido_config = {
    "pasos_motor_rev": 200,   # pasos completos por vuelta del motor (1.8°)
    "microstep": 32,          # micropasos (DRV8825 MICROSTEP_1_32)
    "reductor": 60,           # relación de reducción 1:60
    # Eje de rotación en frame cámara (mm). Por defecto vertical (eje Y de OpenCV).
    "axis_dir": [0.0, 1.0, 0.0],
    "axis_point": None,       # se fija al centroide del primer perfil si es None
    "eje_modo": "auto",       # "auto" (centroide 1er perfil) o "manual" (axis_point fijo)
    "sentido": 1,             # +1 o -1: signo de la rotación al ensamblar
}
barrido_state = {
    "points": [],   # lista de arrays Nx3 ya rotados al frame objeto
    "n_perfiles": 0,
    "angulo_acum_deg": 0.0,
}

# Progreso del barrido en curso (lo consulta la UI por polling).
barrido_progreso = {
    "activo": False,
    "actual": 0,        # perfiles procesados
    "total": 0,         # perfiles previstos
    "angulo_deg": 0.0,  # ángulo acumulado
    "num_puntos": 0,    # puntos acumulados
    "fase": "",         # "home" | "offset" | "barrido" (para la UI)
}

# Evento para sincronizar con la respuesta MOVE_DONE del Pico
move_done_event = threading.Event()
last_move_info = {"pasos": 0, "dir": 0}

# Evento para sincronizar con el fin del AUTO_HOME (el Pico imprime
# "[AUTO_HOME] HOMING COMPLETADO" al terminar; no hay token dedicado).
home_done_event = threading.Event()

# Evento para abortar un barrido en curso (lo activa /test_barrido/stop)
barrido_stop_event = threading.Event()

# ======================== CALIBRACIÓN MECÁNICA ==========================
# Rango de barrido medido EN PASOS desde HOME: offset (inicio) y máximo (fin).
# Un barrido "real" (no test) va a HOME, avanza el offset y barre hasta el máximo.
MECANICA_FILE = os.path.join("static", "mecanica.json")
mecanica_config = {"offset_pasos": 0, "max_pasos": 0}

# Posición actual del eje en pasos desde HOME (dir 1 aleja de home → suma;
# dir 0 acerca a home → resta). Se pone en 0 al completar un AUTO_HOME.
posicion_desde_home = 0
home_referenciado = False     # True tras un homing exitoso

def _cargar_mecanica():
    global mecanica_config
    try:
        if os.path.isfile(MECANICA_FILE):
            with open(MECANICA_FILE, encoding="utf-8") as f:
                d = json.load(f)
            mecanica_config["offset_pasos"] = int(d.get("offset_pasos", 0))
            mecanica_config["max_pasos"]    = int(d.get("max_pasos", 0))
            print(f"📐 Mecánica cargada: offset={mecanica_config['offset_pasos']}, "
                  f"max={mecanica_config['max_pasos']}")
    except Exception as e:
        print(f"⚠️ No se pudo cargar {MECANICA_FILE}: {e}")

_cargar_mecanica()

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

# ======================== CONTROL DE ACCESO A LA CÁMARA ==========================
# Protege TODAS las llamadas a picam2 (capture/configure/start/stop) para evitar
# condiciones de carrera entre los hilos de streaming, captura HR y reconfiguración.
camera_lock = threading.Lock()

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
    global ser, posicion_desde_home, home_referenciado
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

            # ===================== SINCRONIZACIÓN DE MOVIMIENTO EXACTO =====================
            # La Pico responde "MOVE_DONE <pasos> <dir>" al terminar un MOVE.
            if line.startswith("MOVE_DONE"):
                try:
                    parts = line.split()
                    last_move_info["pasos"] = int(parts[1]) if len(parts) > 1 else 0
                    last_move_info["dir"] = int(parts[2]) if len(parts) > 2 else 0
                except Exception:
                    pass
                move_done_event.set()
                continue

            # ===================== FIN DE AUTO_HOME =====================
            if "HOMING COMPLETADO" in line:
                posicion_desde_home = 0
                home_referenciado = True
                home_done_event.set()
                continue

            # ===================== LÓGICA DE FOTO AUTOMÁTICA =====================
            # La Pico imprime: "FOTO <angulo>"
            if line.startswith("FOTO"):
                try:
                    parts = line.split()
                    valor = float(parts[1]) if len(parts) > 1 else 0.0
                except:
                    valor = 0.0

                # 1) Capturar foto (alta resolución, ya en BGR)
                success, frame_bgr = capture_high_res_frame()
                if success and frame_bgr is not None:
                    filename = f"foto_{valor:.2f}.jpg"
                    path = os.path.join(FOTOS_LOOP_FOLDER, filename)
                    cv2.imwrite(path, frame_bgr)

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

        with camera_lock:
            frame_bgr = picam2.capture_array()
        if frame_bgr is None:
            time.sleep(0.01)
            continue

        # Nota: picamera2 con "RGB888" devuelve realmente BGR en el array numpy,
        # que es justo lo que esperan cv2.imwrite/imencode. No convertir.

        # Lógica calibración automática (mantengo tu lógica)
        if is_auto_calibrating:
            try:
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
            except Exception as e:
                print(f"Error auto-calib: {e}")

        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_color_frames():
    """Generador para vista de calibración de color.

    Nota: picamera2 "RGB888" entrega el array en orden BGR. Los nombres r_min/g_min/b_min
    del UI son etiquetas históricas; aquí se aplican sobre los canales del array tal cual.
    """
    while True:
        with camera_lock:
            frame = picam2.capture_array()
        if frame is None:
            time.sleep(0.01)
            continue

        mask = (frame[:, :, 0] >= color_thresholds["r_min"]) & (frame[:, :, 0] <= color_thresholds["r_max"]) & \
               (frame[:, :, 1] >= color_thresholds["g_min"]) & (frame[:, :, 1] <= color_thresholds["g_max"]) & \
               (frame[:, :, 2] >= color_thresholds["b_min"]) & (frame[:, :, 2] <= color_thresholds["b_max"])

        filtered = np.zeros_like(frame)
        filtered[mask] = [255, 255, 255]

        _, buffer = cv2.imencode('.jpg', filtered)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def capture_high_res_frame():
    """Captura una imagen en alta resolución y la devuelve en BGR (lista para OpenCV).

    Usa switch_mode_and_capture_array, que es atómico y mucho más rápido que el
    ciclo stop/configure/start/stop/configure/start. Toda la operación se hace
    bajo camera_lock para evitar colisiones con los generadores de streaming.

    Nota: picamera2 con "RGB888" entrega el array directamente en orden BGR
    (los nombres de formato hacen referencia al orden de bytes, no al orden
    numpy). Por eso NO se hace cvtColor aquí.
    """
    frame = None
    try:
        with camera_lock:
            still_config = picam2.create_still_configuration(
                main={"format": "RGB888",
                      "size": (capture_resolution["width"], capture_resolution["height"])}
            )
            frame = picam2.switch_mode_and_capture_array(still_config, "main")
    except Exception as e:
        print(f"⚠️ Error en capture_high_res_frame: {e}")
        return False, None

    if frame is None:
        return False, None

    return True, frame

def capture_frame_fast():
    """Captura un frame con la cámara YA en modo captura (sin switch de modo).

    Pensada para el barrido con streaming pausado: el sensor se deja fijo en el
    modo de captura una sola vez y cada perfil se toma con un simple
    capture_array(), evitando las dos reconfiguraciones por foto que hace
    switch_mode_and_capture_array. Devuelve (ok, frame_bgr en BGR).
    """
    try:
        with camera_lock:
            frame = picam2.capture_array("main")
    except Exception as e:
        print(f"⚠️ Error en capture_frame_fast: {e}")
        return False, None
    if frame is None:
        return False, None
    return True, frame

def gen_frames_roi():
    """Generador ROI con zoom digital"""
    global roi_zoom_level
    OUTPUT_SIZE = (1920, 1080)

    while True:
        with camera_lock:
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
        except Exception:
            # NO capturar GeneratorExit (cliente desconectado): debe propagar
            # para que el generador cierre limpio.
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
    return render_template("ajustes.html",
        perfil_camara=perfil_activo["camara"],
        perfil_laser=perfil_activo["laser"],
        perfil_plano=perfil_activo["plano"],
        perfil_eje=perfil_activo["eje"]
    )

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
        streaming_active = True 
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

def _calib_resolucion_str(folder, nombre):
    """Devuelve 'W×H' de una calibración, leyendo image_size directo o, si no lo
    tiene, vía la cámara referenciada (camera_calibration). None si no se puede."""
    try:
        with open(os.path.join(folder, nombre), encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    size = data.get("image_size")
    if not size:
        cam = data.get("camera_calibration")
        if cam:
            try:
                with open(os.path.join(CALIB_FOLDER, cam), encoding="utf-8") as f:
                    size = json.load(f).get("image_size")
            except Exception:
                size = None
    if size and len(size) == 2:
        return f"{int(size[0])}×{int(size[1])}"
    return None

def _resoluciones_de(folder, archivos):
    return {a: _calib_resolucion_str(folder, a) for a in archivos}

@app.route('/ajustes/camara')
def ajustes_camara():
    archivos = []
    if os.path.exists(CALIB_FOLDER):
        archivos = [f for f in os.listdir(CALIB_FOLDER) if f.endswith(".json")]
    return render_template("ajustes_camara.html", archivos=archivos,
                           perfil_activo=perfil_activo["camara"],
                           resoluciones=_resoluciones_de(CALIB_FOLDER, archivos))

# ======================== RUTAS DE CALIBRACIÓN DE LASER ==========================

LASER_FOLDER = os.path.join("static", "calibraciones_laser")
os.makedirs(LASER_FOLDER, exist_ok=True)

# Último frame capturado para detección de laser (permite recalcular sin re-capturar)
_last_laser_frame = None

@app.route('/ajustes/laser')
def ajustes_laser():
    archivos = []
    if os.path.exists(LASER_FOLDER):
        archivos = [f for f in os.listdir(LASER_FOLDER) if f.endswith(".json")]
    return render_template("ajustes_laser.html", archivos=archivos,
                           perfil_activo=perfil_activo["laser"],
                           resoluciones=_resoluciones_de(LASER_FOLDER, archivos))


@app.route('/ajustes/laser/nueva')
def nueva_calibracion_laser():
    """Página de calibración interactiva. Si se pasa ?cargar=nombre.json
    pre-carga los parámetros de ese perfil en los sliders."""
    perfil_inicial = None
    cargar = request.args.get("cargar", "").strip()
    if cargar and cargar.endswith(".json"):
        ruta = os.path.join(LASER_FOLDER, cargar)
        if os.path.isfile(ruta):
            try:
                with open(ruta, "r", encoding="utf-8") as f:
                    perfil_inicial = json.load(f)
            except (json.JSONDecodeError, ValueError, OSError) as e:
                # Perfil vacío/corrupto: no romper, abrir con valores por defecto
                print(f"⚠️ Perfil de láser inválido o vacío '{cargar}': {e}")
                perfil_inicial = None
    return render_template("nueva_calibracion_laser.html",
                           perfil_inicial=perfil_inicial)


@app.route('/ajustes/laser/previsualizar', methods=['POST'])
def laser_previsualizar():
    """Captura una imagen, corre el detector con los parámetros recibidos
    y devuelve JSON con estadísticas + imagen JPEG + heatmap en base64.

    Campo extra en el body:
      resolucion: "stream"   → usa resolución de streaming (más rápido)
                  "capture"  → usa resolución de captura completa (más preciso)
    """
    import base64
    data = request.get_json(force=True)
    params = {
        "sat_min":        int(data.get("sat_min",        40)),
        "val_min":        int(data.get("val_min",        30)),
        "val_alto":       int(data.get("val_alto",       220)),
        "score_min":      float(data.get("score_min",    18)),
        "kernel_suave_x": int(data.get("kernel_suave_x", 9)),
        "ancho_min":      int(data.get("ancho_min",      2)),
        "ancho_max":      int(data.get("ancho_max",      60)),
        "kernel_2d":      int(data.get("kernel_2d",      3)),
        "cierre_x":       int(data.get("cierre_x",       0)),
        "peso_nucleo":    float(data.get("peso_nucleo",  60)),
    }
    resolucion = data.get("resolucion", "capture")

    # Captura según resolución elegida
    if resolucion == "stream":
        try:
            with camera_lock:
                frame_bgr = picam2.capture_array()
            success = frame_bgr is not None
        except Exception as e:
            print(f"⚠️ Error capturando frame de stream: {e}")
            success, frame_bgr = False, None
        res_label = f"{stream_resolution['width']}×{stream_resolution['height']}"
    else:
        success, frame_bgr = capture_high_res_frame()
        res_label = f"{capture_resolution['width']}×{capture_resolution['height']}"

    if not success or frame_bgr is None:
        return jsonify({"success": False, "message": "Error al capturar imagen"}), 500

    # Cachear frame para permitir recalcular sin re-capturar
    global _last_laser_frame
    _last_laser_frame = frame_bgr

    roi = bool(data.get("roi", False))
    vectorizado = bool(data.get("vectorizado", True))
    return _calcular_y_devolver(frame_bgr, params, res_label, roi=roi, vectorizado=vectorizado)


def _calcular_y_devolver(frame_bgr, params, res_label, roi=False, vectorizado=True):
    """Calcula detección + heatmap sobre frame_bgr y devuelve JSON.

    Si roi=True usa el mismo modo banda que el barrido (más rápido). El heatmap
    se recoloca a ancho completo para que quede alineado con la imagen.
    vectorizado elige la detección vectorizada (rápida) o la clásica (continuidad).
    """
    import base64
    result = detect_laser(frame_bgr, roi=roi, vectorizado=vectorizado, **params)
    scale = 0.85

    # Heatmap: con ROI el score_s es solo la banda → lo embebemos en un lienzo
    # del ancho completo (fuera de la banda queda negro = "no procesado").
    score_s = result["score_s"]
    band = result.get("band")
    if band is not None:
        full = np.zeros((result["H"], result["W"]), dtype=score_s.dtype)
        full[:, band[0]:band[1]] = score_s
        score_s = full
        res_label = f"{res_label} · banda {band[1] - band[0]}px"

    overlay_bytes = build_preview_jpeg(frame_bgr, result["xs"], scale=scale, quality=88)
    heatmap_bytes = build_heatmap_jpeg(score_s, scale=scale, quality=88)
    overlay_b64   = base64.b64encode(overlay_bytes).decode("utf-8")
    heatmap_b64   = base64.b64encode(heatmap_bytes).decode("utf-8")
    print(f"🔴 Laser: {result['pct']}% válidas en {result['ms']} ms  [{res_label}]")
    # Guardar xs en el endpoint /fila para que pueda marcar la detección
    laser_fila._last_xs = result["xs"]
    return jsonify({
        "success":    True,
        "pct":        result["pct"],
        "ms":         result["ms"],
        "H":          result["H"],
        "W":          result["W"],
        "resolucion": res_label,
        "imagen":     overlay_b64,
        "heatmap":    heatmap_b64,
        "xs":         result["xs"],
        "band":       band,
    })


@app.route('/ajustes/laser/recalcular', methods=['POST'])
def laser_recalcular():
    """Recalcula la detección sobre el último frame capturado sin tomar foto nueva."""
    global _last_laser_frame
    if _last_laser_frame is None:
        return jsonify({"success": False, "message": "No hay frame capturado. Usá Capturar primero."}), 400

    data = request.get_json(force=True)
    params = {
        "sat_min":        int(data.get("sat_min",        40)),
        "val_min":        int(data.get("val_min",        30)),
        "val_alto":       int(data.get("val_alto",       220)),
        "score_min":      float(data.get("score_min",    18)),
        "kernel_suave_x": int(data.get("kernel_suave_x", 9)),
        "ancho_min":      int(data.get("ancho_min",      2)),
        "ancho_max":      int(data.get("ancho_max",      60)),
        "kernel_2d":      int(data.get("kernel_2d",      3)),
        "cierre_x":       int(data.get("cierre_x",       0)),
        "peso_nucleo":    float(data.get("peso_nucleo",  60)),
    }
    H, W = _last_laser_frame.shape[:2]
    res_label = f"{W}×{H} (cached)"
    roi = bool(data.get("roi", False))
    vectorizado = bool(data.get("vectorizado", True))
    return _calcular_y_devolver(_last_laser_frame, params, res_label, roi=roi, vectorizado=vectorizado)


@app.route('/ajustes/laser/fila', methods=['POST'])
def laser_fila():
    """Devuelve el vector de score (post-blur) de una fila específica.

    Body: { row: int, sat_min, val_min, val_alto, score_min, kernel_suave_x }
    Retorna: { success, score: [float...], score_min: float, x_detectado: float|null, row: int, W: int }
    """
    global _last_laser_frame
    if _last_laser_frame is None:
        return jsonify({"success": False, "message": "No hay frame. Capturá primero."}), 400

    data = request.get_json(force=True)
    row  = int(data.get("row", 0))
    sat_min        = int(data.get("sat_min",        40))
    val_min        = int(data.get("val_min",        30))
    val_alto       = int(data.get("val_alto",       220))
    score_min      = float(data.get("score_min",    18))
    kernel_suave_x = int(data.get("kernel_suave_x", 9))
    peso_nucleo    = float(data.get("peso_nucleo",  60))
    cierre_x       = int(data.get("cierre_x",       0))

    H, W = _last_laser_frame.shape[:2]
    row  = max(0, min(row, H - 1))

    # Calcular score + blur + cierre igual que en detect_laser
    score = laser_score(_last_laser_frame,
                        sat_min=sat_min, val_min=val_min, val_alto=val_alto,
                        peso_nucleo=peso_nucleo)
    k2 = int(data.get("kernel_2d", 3))
    if k2 >= 3:
        if k2 % 2 == 0: k2 += 1
        score = cv2.GaussianBlur(score, (k2, k2), 0)
    k = kernel_suave_x if kernel_suave_x % 2 == 1 else kernel_suave_x + 1
    score_s = score if k <= 1 else cv2.GaussianBlur(score, (k, 1), 0)
    if cierre_x >= 2:
        score_s = cv2.morphologyEx(score_s, cv2.MORPH_CLOSE,
                                   np.ones((1, cierre_x), dtype=np.uint8))

    fila = score_s[row].tolist()

    # Posición X detectada en esa fila (del último resultado guardado)
    x_det = None
    xs_cache = getattr(laser_fila, '_last_xs', None)
    if xs_cache and row < len(xs_cache):
        x_det = xs_cache[row]

    return jsonify({
        "success":      True,
        "score":        [round(v, 1) for v in fila],
        "score_min":    score_min,
        "x_detectado":  x_det,
        "row":          row,
        "W":            W,
    })


@app.route('/ajustes/laser/guardar', methods=['POST'])
def laser_guardar():
    """Guarda los parámetros de detección como perfil JSON en LASER_FOLDER."""
    data   = request.get_json(force=True)
    nombre = data.get("nombre", "").strip()
    if not nombre:
        return jsonify({"success": False, "message": "Nombre vacío"}), 400
    if not nombre.endswith(".json"):
        nombre += ".json"

    # Solo guardamos los campos conocidos
    campos = ["sat_min", "val_min", "val_alto", "score_min",
              "kernel_suave_x", "ancho_min", "ancho_max", "kernel_2d",
              "cierre_x", "peso_nucleo"]
    perfil = {c: data[c] for c in campos if c in data}
    perfil["nombre"] = nombre
    # Registrar la resolución del frame con que se ajustó (si hay uno cacheado)
    if _last_laser_frame is not None:
        h, w = _last_laser_frame.shape[:2]
        perfil["image_size"] = [int(w), int(h)]

    ruta = os.path.join(LASER_FOLDER, nombre)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(perfil, f, indent=2)

    print(f"💾 Perfil laser guardado: {nombre}")
    return jsonify({"success": True, "nombre": nombre})

@app.route('/ajustes/plano')
def ajustes_plano():
    archivos = []
    if os.path.exists(PLANO_FOLDER):
        archivos = [f for f in os.listdir(PLANO_FOLDER) if f.endswith(".json")]
    return render_template("ajustes_plano.html", archivos=archivos,
                           perfil_activo=perfil_activo["plano"],
                           resoluciones=_resoluciones_de(PLANO_FOLDER, archivos))

# ----------------- FUNCIONES DE TRIANGULACIÓN DEL PLANO DEL LÁSER -----------------
# Adaptadas de calibrate_laser_plane.py. Idea: en cada foto el láser cruza un
# chessboard; con solvePnP conocemos el plano del tablero en coordenadas de cámara,
# y retroproyectando los píxeles del láser que caen DENTRO del tablero obtenemos
# puntos 3D del plano del láser. Acumulando varias poses ajustamos el plano por SVD.

def _plano_load_camera_json(path):
    """Carga intrínsecos (K, dist) desde un JSON de calibración de cámara."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["distortion_coefficients"], dtype=np.float64).reshape(-1)
    image_size = tuple(data.get("image_size", []))
    return K, dist, image_size

def _plano_detect_chessboard(img_bgr, cols, rows):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             + cv2.CALIB_CB_NORMALIZE_IMAGE
             + cv2.CALIB_CB_FAST_CHECK)
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners

def _plano_object_points(cols, rows, square_mm):
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_mm)
    return objp

def _plano_solve_pose(object_points, image_points, K, dist):
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP no convergió")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)

def _plano_board_plane_camframe(R, t):
    """Plano del tablero en frame cámara como (n, d) con n·X = d, n unitario."""
    n = R[:, 2].astype(np.float64)
    n = n / np.linalg.norm(n)
    d = float(n @ t)
    return n, d

def _plano_load_laser_params(path):
    """Carga los parámetros de detección de un perfil de láser JSON (LASER_FOLDER).

    Solo se quedan los campos que entiende detect_laser; el resto se ignora.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    campos = ["sat_min", "val_min", "val_alto", "score_min",
              "kernel_suave_x", "ancho_min", "ancho_max", "kernel_2d",
              "cierre_x", "peso_nucleo"]
    return {c: data[c] for c in campos if c in data}

def _plano_detect_laser_points(img_bgr, laser_params, mask=None):
    """Detecta la línea láser con el detector calibrado (detect_laser) y devuelve
    array Nx2 de píxeles (u, v), filtrando los que caen dentro de `mask`.

    Reutiliza el mismo algoritmo y perfil de parámetros que la calibración de
    láser, en lugar de una detección ad-hoc.
    """
    result = detect_laser(img_bgr, **laser_params)
    xs = result["xs"]  # list[float|None] de longitud H (X subpíxel por fila)

    puntos = []
    h = len(xs)
    w = img_bgr.shape[1]
    for y in range(h):
        x = xs[y]
        if x is None:
            continue
        xi = int(round(x))
        if xi < 0 or xi >= w:
            continue
        if mask is not None and not mask[y, xi]:
            continue
        puntos.append((float(x), float(y)))
    return np.asarray(puntos, dtype=np.float64).reshape(-1, 2)

def _plano_board_mask(img_shape, corners, cols, rows, margin_frac=0.06):
    """Máscara booleana del interior del tablero (convex hull encogido)."""
    h, w = img_shape[:2]
    pts = corners.reshape(-1, 2)
    centroid = pts.mean(axis=0)
    shrunk = centroid + (pts - centroid) * (1.0 - margin_frac)
    hull = cv2.convexHull(shrunk.astype(np.float32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 1)
    return mask.astype(bool), hull

def _plano_pixels_to_plane(pixels_uv, K, dist, n, d):
    """Intersecta el rayo de cada píxel con el plano del tablero (n·X = d)."""
    if pixels_uv.shape[0] == 0:
        return np.empty((0, 3))
    undist = cv2.undistortPoints(pixels_uv.reshape(-1, 1, 2), K, dist)
    undist = undist.reshape(-1, 2)
    rays = np.hstack([undist, np.ones((undist.shape[0], 1))])  # Nx3
    denom = rays @ n
    valid = np.abs(denom) > 1e-9
    t = np.zeros_like(denom)
    t[valid] = d / denom[valid]
    pts = rays * t[:, None]
    return pts[valid]

def _plano_fit_plane(points):
    """Ajusta n·X = d (n unitario) por SVD. Devuelve (n, d, rms_mm)."""
    if points.shape[0] < 3:
        raise ValueError("Se necesitan al menos 3 puntos para ajustar un plano")
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    d = float(n @ centroid)
    residuals = points @ n - d
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return n, d, rms

# ----------------- RUTAS DE CALIBRACIÓN DEL PLANO DEL LÁSER -----------------
@app.route('/ajustes/plano/nueva')
def nueva_calibracion_plano():
    global plane_calib
    plane_calib = {"points": [], "images": [], "counts": []}
    return render_template("nueva_calibracion_plano.html",
                           perfil_camara=perfil_activo.get("camara"),
                           perfil_laser=perfil_activo.get("laser"))

@app.route('/calibracion_plano/capture', methods=['POST'])
def capture_plane_calibration():
    global plane_calib, plane_config
    data = request.get_json(force=True)
    cols      = int(data.get('cols',      plane_config['cols']))
    rows      = int(data.get('rows',      plane_config['rows']))
    square_mm = float(data.get('square_mm', plane_config['square_mm']))
    plane_config.update(cols=cols, rows=rows, square_mm=square_mm)

    # La triangulación necesita los intrínsecos: exigimos perfil de cámara activo
    cam_name = perfil_activo.get("camara")
    if not cam_name:
        return jsonify({"success": False,
                        "message": "Selecciona primero un perfil de cámara activo en Ajustes › Cámara."}), 400
    cam_path = os.path.join(CALIB_FOLDER, cam_name)
    if not os.path.isfile(cam_path):
        return jsonify({"success": False,
                        "message": f"No se encontró el JSON de cámara: {cam_name}"}), 400
    try:
        K, dist, _ = _plano_load_camera_json(cam_path)
    except Exception as e:
        return jsonify({"success": False,
                        "message": f"Error leyendo calibración de cámara: {e}"}), 400

    # La detección del láser reutiliza el detector calibrado + perfil de láser activo
    laser_name = perfil_activo.get("laser")
    if not laser_name:
        return jsonify({"success": False,
                        "message": "Selecciona primero un perfil de láser activo en Ajustes › Láser."}), 400
    laser_path = os.path.join(LASER_FOLDER, laser_name)
    if not os.path.isfile(laser_path):
        return jsonify({"success": False,
                        "message": f"No se encontró el JSON de láser: {laser_name}"}), 400
    try:
        laser_params = _plano_load_laser_params(laser_path)
    except Exception as e:
        return jsonify({"success": False,
                        "message": f"Error leyendo perfil de láser: {e}"}), 400

    print(f"📸 Captura plano. Patrón {cols}x{rows}, cuadro {square_mm}mm, "
          f"cámara '{cam_name}', láser '{laser_name}'")
    success, frame_bgr = capture_high_res_frame()
    if not success:
        return jsonify({"success": False, "message": "Error al capturar imagen"})

    ok, corners = _plano_detect_chessboard(frame_bgr, cols, rows)
    if not ok:
        return jsonify({"success": False,
                        "message": f"Chessboard {cols}x{rows} no detectado en la imagen."})

    objp = _plano_object_points(cols, rows, square_mm)
    try:
        R, t = _plano_solve_pose(objp, corners, K, dist)
    except Exception as e:
        return jsonify({"success": False, "message": f"Error de pose (solvePnP): {e}"})
    n_board, d_board = _plano_board_plane_camframe(R, t)

    mask, hull = _plano_board_mask(frame_bgr.shape, corners, cols, rows)
    laser_uv = _plano_detect_laser_points(frame_bgr, laser_params, mask=mask)

    if laser_uv.shape[0] < 5:
        return jsonify({"success": False,
                        "message": f"Láser sobre el tablero insuficiente ({laser_uv.shape[0]} px). "
                                   "¿El láser cruza el chessboard?"})

    pts3d = _plano_pixels_to_plane(laser_uv, K, dist, n_board, d_board)
    plane_calib["points"].append(pts3d)
    plane_calib["counts"].append(int(laser_uv.shape[0]))

    # Overlay para la galería: esquinas + contorno del tablero + puntos del láser
    overlay = frame_bgr.copy()
    cv2.drawChessboardCorners(overlay, (cols, rows), corners, True)
    cv2.polylines(overlay, [hull.astype(np.int32)], True, (0, 255, 255), 2)
    for u, v in laser_uv:
        cv2.circle(overlay, (int(round(u)), int(round(v))), 2, (0, 255, 0), -1)

    filename = f"plano_{len(plane_calib['images'])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(CAPTURAS_FOLDER, filename)
    cv2.imwrite(path, overlay)
    plane_calib["images"].append(path)

    print(f"✅ Pose añadida: {laser_uv.shape[0]} px láser → {pts3d.shape[0]} pts 3D "
          f"(total poses: {len(plane_calib['images'])})")

    return jsonify({
        "success": True,
        "message": f"Pose añadida: {laser_uv.shape[0]} px de láser → {pts3d.shape[0]} puntos 3D.",
        "images_count": len(plane_calib['images']),
        "url": url_for('static', filename=f"capturas/{filename}")
    })

@app.route('/calibracion_plano/compute', methods=['POST'])
def compute_plane_calibration():
    global plane_calib, plane_config

    if len(plane_calib["images"]) < MIN_PLANE_CAPTURES:
        return jsonify({
            "success": False,
            "message": f"Se necesitan al menos {MIN_PLANE_CAPTURES} poses. "
                       f"Tienes {len(plane_calib['images'])}."
        })

    data = request.get_json(force=True)
    nombre = (data.get('name') or f"plano_{datetime.now().strftime('%Y%m%d_%H%M%S')}").strip()

    points = np.vstack(plane_calib["points"])
    print(f"🔧 Ajustando plano con {points.shape[0]} puntos de "
          f"{len(plane_calib['images'])} poses...")
    try:
        n, d, rms = _plano_fit_plane(points)
    except Exception as e:
        return jsonify({"success": False, "message": f"Error ajustando plano: {e}"})

    # Normalizamos el signo para que d sea positivo (plano delante de la cámara)
    if d < 0:
        n, d = -n, -d

    # Aviso de poca diversidad de poses (puntos casi colineales)
    warning = None
    try:
        _, sv, _ = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
        if sv[1] < 1e-3 * sv[0]:
            warning = ("Los puntos son casi colineales (poca diversidad de poses). "
                       "Añade fotos con el tablero más inclinado y a distintas distancias.")
    except Exception:
        pass

    if not nombre.endswith(".json"):
        nombre += ".json"

    payload = {
        "name": nombre,
        "plane_normal": n.tolist(),
        "plane_d_mm": d,
        "convention": "n . X = d ; X en mm, frame cámara OpenCV",
        "rms_residual_mm": rms,
        "image_size": [capture_resolution["width"], capture_resolution["height"]],
        "num_points": int(points.shape[0]),
        "num_images_used": len(plane_calib["images"]),
        "camera_calibration": perfil_activo.get("camara"),
        "laser_profile": perfil_activo.get("laser"),
        "chessboard": {"cols": plane_config["cols"], "rows": plane_config["rows"],
                       "square_mm": plane_config["square_mm"]},
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    filepath = os.path.join(PLANO_FOLDER, nombre)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"💾 Plano del láser guardado: {filepath}  (RMS {rms:.3f} mm)")

    return jsonify({
        "success": True,
        "message": f"Plano calibrado. Error RMS: {rms:.3f} mm",
        "calibration_name": nombre,
        "rms_residual_mm": rms,
        "num_images": len(plane_calib['images']),
        "warning": warning,
    })

@app.route('/calibracion_plano/reset', methods=['POST'])
def reset_plane_calibration():
    global plane_calib
    print("🔄 Reiniciando calibración de plano...")
    plane_calib = {"points": [], "images": [], "counts": []}
    return jsonify({"success": True, "message": "Calibración de plano reiniciada"})

@app.route('/calibracion_plano/status', methods=['GET'])
def plane_calibration_status():
    image_urls = [url_for('static', filename=f"capturas/{os.path.basename(p)}")
                  for p in plane_calib["images"]]
    return jsonify({
        "images_captured": len(plane_calib["images"]),
        "ready_to_calibrate": len(plane_calib["images"]) >= MIN_PLANE_CAPTURES,
        "image_urls": image_urls,
        "perfil_camara": perfil_activo.get("camara"),
        "perfil_laser": perfil_activo.get("laser"),
    })

@app.route('/ajustes/plano/cancelar')
def cancelar_calibracion_plano():
    global plane_calib
    plane_calib = {"points": [], "images": [], "counts": []}
    if os.path.exists(CAPTURAS_FOLDER):
        for file in os.listdir(CAPTURAS_FOLDER):
            if file.startswith("plano_"):
                try:
                    os.remove(os.path.join(CAPTURAS_FOLDER, file))
                except Exception as e:
                    print(f"Error eliminando {file}: {e}")
    return redirect(url_for("ajustes"))

# ======================== CALIBRACIÓN DEL EJE DE GIRO ==========================
# Setup de cámara móvil: la cámara + láser cuelgan de una barra que bascula
# alrededor de un eje fijo. Para ensamblar la nube del barrido necesitamos ese
# eje (un punto y una dirección) EN EL FRAME DE LA CÁMARA.
#
# Procedimiento: chessboard fijo en el suelo; se captura en varios ángulos del
# barrido. En cada uno, solvePnP da la pose tablero→cámara (R_k, t_k). La cámara
# es rígida respecto al eje, así que la transformación cámara_0→cámara_k es una
# rotación pura alrededor del eje:
#     Q_k = R_k · R_0ᵀ      (rotación relativa)
#     c_k = t_k - Q_k · t_0 (traslación relativa)
# La dirección del eje es el eje de rotación de cada Q_k (promediado), y un punto
# del eje sale de resolver (I - Q_k)·a = c_k por mínimos cuadrados sobre todas
# las poses. Todo queda en coordenadas de la cámara (mm), igual que los puntos 3D
# reconstruidos en el barrido.

def _eje_fit_axis(Rs, ts):
    """Ajusta el eje de giro (punto a, dirección d unitaria) en frame cámara a
    partir de poses tablero→cámara. Devuelve (a, d, angulos_deg, rms_punto_mm)."""
    R0, t0 = Rs[0], ts[0]
    ejes      = []     # dirección del eje de cada rotación relativa
    angulos   = []     # magnitud de la rotación relativa (grados) — sanity check
    A_rows    = []     # filas de (I - Q_k)
    c_rows    = []     # términos c_k
    for k in range(1, len(Rs)):
        Q = Rs[k] @ R0.T
        c = ts[k] - Q @ t0
        ang = np.arccos(np.clip((np.trace(Q) - 1.0) / 2.0, -1.0, 1.0))
        # vector eje = parte antisimétrica de Q  (= 2·sin(ang)·d)
        ax = np.array([Q[2, 1] - Q[1, 2],
                       Q[0, 2] - Q[2, 0],
                       Q[1, 0] - Q[0, 1]], dtype=np.float64)
        nrm = np.linalg.norm(ax)
        if nrm < 1e-9:
            continue   # poses casi idénticas: no aportan eje
        ejes.append(ax / nrm)
        angulos.append(float(np.degrees(ang)))
        A_rows.append(np.eye(3) - Q)
        c_rows.append(c)

    if not ejes:
        raise ValueError("No hubo rotación apreciable entre las poses. "
                         "Mové la barra varios grados entre capturas.")

    # Dirección promedio (alineando signos respecto a la primera)
    ref = ejes[0]
    d = np.mean([e if e @ ref >= 0 else -e for e in ejes], axis=0)
    d = d / np.linalg.norm(d)

    # Punto del eje por mínimos cuadrados (I-Q es rank 2; varias poses lo fijan)
    A = np.vstack(A_rows)
    c = np.concatenate(c_rows)
    a, _, _, _ = np.linalg.lstsq(A, c, rcond=None)
    rms = float(np.sqrt(np.mean((A @ a - c) ** 2)))
    # La posición del punto A LO LARGO del eje NO es observable (la recta es
    # infinita; rotar da igual qué punto del eje se use). La lstsq puede inflar
    # esa componente con ruido (→ valores absurdos). Nos quedamos con el punto
    # del eje MÁS CERCANO al origen de la cámara: la componente perpendicular.
    a = a - float(a @ d) * d
    return a, d, angulos, rms


@app.route('/ajustes/eje')
def ajustes_eje():
    archivos = []
    if os.path.exists(EJE_FOLDER):
        archivos = [f for f in os.listdir(EJE_FOLDER) if f.endswith(".json")]
    return render_template("ajustes_eje.html", archivos=archivos,
                           perfil_activo=perfil_activo["eje"],
                           resoluciones=_resoluciones_de(EJE_FOLDER, archivos))

@app.route('/ajustes/eje/nueva')
def nueva_calibracion_eje():
    global eje_calib
    eje_calib = {"R": [], "t": [], "images": []}
    return render_template("nueva_calibracion_eje.html",
                           perfil_camara=perfil_activo.get("camara"))

@app.route('/calibracion_eje/capture', methods=['POST'])
def capture_eje_calibration():
    global eje_calib, eje_config
    data = request.get_json(force=True)
    cols      = int(data.get('cols',      eje_config['cols']))
    rows      = int(data.get('rows',      eje_config['rows']))
    square_mm = float(data.get('square_mm', eje_config['square_mm']))
    eje_config.update(cols=cols, rows=rows, square_mm=square_mm)

    cam_name = perfil_activo.get("camara")
    if not cam_name:
        return jsonify({"success": False,
                        "message": "Selecciona primero un perfil de cámara activo en Ajustes › Cámara."}), 400
    cam_path = os.path.join(CALIB_FOLDER, cam_name)
    if not os.path.isfile(cam_path):
        return jsonify({"success": False,
                        "message": f"No se encontró el JSON de cámara: {cam_name}"}), 400
    try:
        K, dist, _ = _plano_load_camera_json(cam_path)
    except Exception as e:
        return jsonify({"success": False,
                        "message": f"Error leyendo calibración de cámara: {e}"}), 400

    success, frame_bgr = capture_high_res_frame()
    if not success:
        return jsonify({"success": False, "message": "Error al capturar imagen"})

    ok, corners = _plano_detect_chessboard(frame_bgr, cols, rows)
    if not ok:
        return jsonify({"success": False,
                        "message": f"Chessboard {cols}x{rows} no detectado en la imagen."})

    objp = _plano_object_points(cols, rows, square_mm)
    try:
        R, t = _plano_solve_pose(objp, corners, K, dist)
    except Exception as e:
        return jsonify({"success": False, "message": f"Error de pose (solvePnP): {e}"})

    eje_calib["R"].append(R)
    eje_calib["t"].append(t)

    overlay = frame_bgr.copy()
    cv2.drawChessboardCorners(overlay, (cols, rows), corners, True)
    filename = f"eje_{len(eje_calib['images'])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(CAPTURAS_FOLDER, filename)
    cv2.imwrite(path, overlay)
    eje_calib["images"].append(path)

    print(f"✅ Pose de eje añadida (total: {len(eje_calib['images'])})")
    return jsonify({
        "success": True,
        "message": f"Pose añadida. Total: {len(eje_calib['images'])}.",
        "images_count": len(eje_calib['images']),
        "url": url_for('static', filename=f"capturas/{filename}")
    })

@app.route('/calibracion_eje/compute', methods=['POST'])
def compute_eje_calibration():
    global eje_calib, eje_config

    if len(eje_calib["images"]) < MIN_EJE_CAPTURES:
        return jsonify({
            "success": False,
            "message": f"Se necesitan al menos {MIN_EJE_CAPTURES} poses. "
                       f"Tienes {len(eje_calib['images'])}."
        })

    data = request.get_json(force=True)
    nombre = (data.get('name') or f"eje_{datetime.now().strftime('%Y%m%d_%H%M%S')}").strip()

    try:
        a, d, angulos, rms = _eje_fit_axis(eje_calib["R"], eje_calib["t"])
    except Exception as e:
        return jsonify({"success": False, "message": f"Error ajustando eje: {e}"})

    # Distancia del eje al origen de la cámara (en tu montaje el eje pasa CERCA
    # de la cámara → deberían ser pocos cm). Si es grande, la toma fue ruidosa.
    offset_mm = float(np.linalg.norm(a))

    # Avisos de calidad de la calibración
    warning = None
    if angulos and max(angulos) < 15.0:
        warning = (f"Barriste poco (ángulo máx {max(angulos):.1f}°). El punto del eje "
                   "queda mal condicionado: capturá poses cubriendo todo el rango (~110°).")
    elif offset_mm > 400.0:
        warning = (f"El eje pasa a {offset_mm:.0f} mm del origen de la cámara, pero en tu "
                   "montaje debería pasar cerca (pocos cm). Probablemente faltan poses con "
                   "mayor separación angular o el tablero se detectó con ruido.")
    elif rms > 8.0:
        warning = (f"Residual del ajuste alto ({rms:.1f} mm): poses poco consistentes. "
                   "Asegurá que el tablero llene el encuadre y esté bien enfocado.")

    if not nombre.endswith(".json"):
        nombre += ".json"

    payload = {
        "name": nombre,
        "axis_point": a.tolist(),     # mm, frame cámara OpenCV
        "axis_dir": d.tolist(),       # unitario, frame cámara OpenCV
        "convention": "cámara móvil; rotar puntos 3D del frame cámara alrededor de (axis_point, axis_dir)",
        "rms_point_mm": rms,
        "axis_offset_mm": round(offset_mm, 2),
        "image_size": [capture_resolution["width"], capture_resolution["height"]],
        "rotation_angles_deg": [round(x, 3) for x in angulos],
        "num_poses": len(eje_calib["images"]),
        "camera_calibration": perfil_activo.get("camara"),
        "chessboard": {"cols": eje_config["cols"], "rows": eje_config["rows"],
                       "square_mm": eje_config["square_mm"]},
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    filepath = os.path.join(EJE_FOLDER, nombre)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"💾 Eje de giro guardado: {filepath}  (RMS punto {rms:.3f} mm, "
          f"ángulos {[round(x,1) for x in angulos]})")

    return jsonify({
        "success": True,
        "message": f"Eje calibrado. Residual del punto: {rms:.3f} mm",
        "calibration_name": nombre,
        "rms_point_mm": rms,
        "axis_point": a.tolist(),
        "axis_dir": d.tolist(),
        "warning": warning,
    })

@app.route('/calibracion_eje/reset', methods=['POST'])
def reset_eje_calibration():
    global eje_calib
    print("🔄 Reiniciando calibración de eje...")
    eje_calib = {"R": [], "t": [], "images": []}
    return jsonify({"success": True, "message": "Calibración de eje reiniciada"})

@app.route('/calibracion_eje/status', methods=['GET'])
def eje_calibration_status():
    image_urls = [url_for('static', filename=f"capturas/{os.path.basename(p)}")
                  for p in eje_calib["images"]]
    return jsonify({
        "images_captured": len(eje_calib["images"]),
        "ready_to_calibrate": len(eje_calib["images"]) >= MIN_EJE_CAPTURES,
        "image_urls": image_urls,
        "perfil_camara": perfil_activo.get("camara"),
    })

@app.route('/ajustes/eje/cancelar')
def cancelar_calibracion_eje():
    global eje_calib
    eje_calib = {"R": [], "t": [], "images": []}
    if os.path.exists(CAPTURAS_FOLDER):
        for file in os.listdir(CAPTURAS_FOLDER):
            if file.startswith("eje_"):
                try:
                    os.remove(os.path.join(CAPTURAS_FOLDER, file))
                except Exception as e:
                    print(f"Error eliminando {file}: {e}")
    return redirect(url_for("ajustes"))

# ======================== TEST DE MEDICIÓN (RECONSTRUCCIÓN 3D) ==========================
# Con el plano del láser calibrado (n·X = d) podemos recuperar el punto 3D de
# CADA píxel donde se detecta el láser, sin necesidad del chessboard:
#   1) píxel (u,v) -> rayo normalizado r=(x,y,1) con undistortPoints
#   2) intersección rayo-plano:  t = d / (r·n) ;  X = t·r   (en mm, frame cámara)
# La "distancia" es Z = X[2] (profundidad) o ‖X‖ (distancia euclidiana al lente).

def _plano_load_plane_json(path):
    """Carga el plano del láser (n unitario, d en mm) desde un JSON de PLANO_FOLDER."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = np.asarray(data["plane_normal"], dtype=np.float64).reshape(3)
    n = n / np.linalg.norm(n)
    d = float(data["plane_d_mm"])
    return n, d

def _test_cargar_perfiles_activos():
    """Carga (K, dist, laser_params, n, d) desde los perfiles activos.

    Devuelve (datos, None) si todo está OK, o (None, mensaje_error) si falta algo.
    """
    cam = perfil_activo.get("camara")
    las = perfil_activo.get("laser")
    pla = perfil_activo.get("plano")
    faltan = [nombre for nombre, v in
              [("cámara", cam), ("láser", las), ("plano", pla)] if not v]
    if faltan:
        return None, f"Faltan perfiles activos: {', '.join(faltan)}."

    rutas = [(os.path.join(CALIB_FOLDER, cam), cam),
             (os.path.join(LASER_FOLDER, las), las),
             (os.path.join(PLANO_FOLDER, pla), pla)]
    for ruta, nombre in rutas:
        if not os.path.isfile(ruta):
            return None, f"No se encontró el archivo de perfil: {nombre}"

    # Garantizar que se capture a la MISMA resolución que la calibración de cámara
    # (si no, K no coincide con los píxeles y la triangulación falla → Z<0 → 0 pts).
    _aplicar_resolucion_camara(cam)

    try:
        K, dist, _ = _plano_load_camera_json(rutas[0][0])
        laser_params = _plano_load_laser_params(rutas[1][0])
        n, d = _plano_load_plane_json(rutas[2][0])
    except Exception as e:
        return None, f"Error leyendo perfiles: {e}"

    return (K, dist, laser_params, n, d), None

def _test_reconstruir(frame_bgr, laser_params, K, dist, n, d):
    """Detecta el láser y reconstruye sus puntos 3D sobre el plano.

    Devuelve (pts3d Nx3 mm con Z>0, result_detect dict). El detector es el mismo
    que usa la calibración de láser (detect_laser), con el perfil activo.
    """
    result = detect_laser(frame_bgr, roi=True, **laser_params)
    xs = result["xs"]  # list[float|None], X subpíxel por fila
    w = frame_bgr.shape[1]

    uv = []
    for y, x in enumerate(xs):
        if x is None or x < 0 or x >= w:
            continue
        uv.append((float(x), float(y)))
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)

    pts3d = _plano_pixels_to_plane(uv, K, dist, n, d)
    # Conservar solo puntos delante de la cámara (Z>0)
    if pts3d.shape[0]:
        pts3d = pts3d[pts3d[:, 2] > 0]
    return pts3d, result

def _test_overlay_b64(frame_bgr, xs_list):
    """Genera el JPEG (base64) con la línea detectada dibujada sobre la imagen."""
    import base64
    overlay_bytes = build_preview_jpeg(frame_bgr, xs_list, scale=0.85, quality=85)
    return base64.b64encode(overlay_bytes).decode("utf-8")

def _stats_distancia(pts3d):
    """Calcula estadísticas de profundidad Z y distancia euclidiana ‖X‖ (mm)."""
    Z = pts3d[:, 2]
    dist_eucl = np.linalg.norm(pts3d, axis=1)
    return {
        "num_puntos": int(pts3d.shape[0]),
        "z_min":   round(float(Z.min()), 1),
        "z_media": round(float(Z.mean()), 1),
        "z_max":   round(float(Z.max()), 1),
        "dist_min":   round(float(dist_eucl.min()), 1),
        "dist_media": round(float(dist_eucl.mean()), 1),
        "dist_max":   round(float(dist_eucl.max()), 1),
    }

@app.route('/ajustes/test')
def test_medicion():
    return render_template("test_medicion.html",
        perfil_camara=perfil_activo.get("camara"),
        perfil_laser=perfil_activo.get("laser"),
        perfil_plano=perfil_activo.get("plano"))

@app.route('/test_medicion/medir', methods=['POST'])
def test_medir():
    """Modo puntual: captura, detecta el láser, reconstruye 3D y devuelve
    estadísticas de distancia + imagen con la línea detectada."""
    ctx, err = _test_cargar_perfiles_activos()
    if err:
        return jsonify({"success": False, "message": err}), 400
    K, dist, laser_params, n, d = ctx

    success, frame_bgr = capture_high_res_frame()
    if not success:
        return jsonify({"success": False, "message": "Error al capturar imagen"}), 500

    pts3d, result = _test_reconstruir(frame_bgr, laser_params, K, dist, n, d)
    if pts3d.shape[0] == 0:
        return jsonify({"success": False,
                        "message": "No se detectó láser válido sobre el plano."}), 200

    stats = _stats_distancia(pts3d)
    print(f"📏 Medición: {stats['num_puntos']} pts · Z medio {stats['z_media']} mm "
          f"(min {stats['z_min']} / max {stats['z_max']})")

    return jsonify({
        "success":   True,
        "stats":     stats,
        "pct":       result["pct"],
        "ms":        result["ms"],
        "imagen":    _test_overlay_b64(frame_bgr, result["xs"]),
    })

@app.route('/test_medicion/escanear', methods=['POST'])
def test_escanear():
    """Modo perfil completo: devuelve la nube de puntos 3D del perfil actual.
    Si `guardar` es true, la guarda como CSV (x,y,z) en static/modelos."""
    data = request.get_json(silent=True) or {}
    guardar = bool(data.get("guardar", False))
    nombre = (data.get("nombre") or "").strip()

    ctx, err = _test_cargar_perfiles_activos()
    if err:
        return jsonify({"success": False, "message": err}), 400
    K, dist, laser_params, n, d = ctx

    success, frame_bgr = capture_high_res_frame()
    if not success:
        return jsonify({"success": False, "message": "Error al capturar imagen"}), 500

    pts3d, result = _test_reconstruir(frame_bgr, laser_params, K, dist, n, d)
    if pts3d.shape[0] == 0:
        return jsonify({"success": False,
                        "message": "No se detectó láser válido sobre el plano."}), 200

    stats = _stats_distancia(pts3d)

    csv_url = None
    if guardar:
        if not nombre:
            nombre = f"perfil_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not nombre.endswith(".csv"):
            nombre += ".csv"
        ruta = os.path.join(MAPS_FOLDER, nombre)
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z"])
            for X, Y, Z in pts3d:
                writer.writerow([round(float(X), 3), round(float(Y), 3), round(float(Z), 3)])
        csv_url = url_for('view_plot', nombre=nombre)
        print(f"💾 Perfil guardado: {ruta} ({pts3d.shape[0]} puntos)")

    # Submuestreo para no enviar demasiados puntos al navegador
    paso = max(1, pts3d.shape[0] // 2000)
    muestra = pts3d[::paso]

    return jsonify({
        "success": True,
        "stats":   stats,
        "pct":     result["pct"],
        "ms":      result["ms"],
        "imagen":  _test_overlay_b64(frame_bgr, result["xs"]),
        "puntos":  [[round(float(x), 2), round(float(y), 2), round(float(z), 2)]
                    for x, y, z in muestra],
        "csv_url": csv_url,
        "nombre":  nombre if guardar else None,
    })


# ======================== TEST DE BARRIDO (NUBE DE PUNTOS ROTACIONAL) ==========================
# Con el plano calibrado reconstruimos el perfil 3D en cada paso del eje. Como el
# eje GIRA (plato), cada perfil se rota por el ángulo acumulado del plato para
# ensamblar la nube completa en un frame "objeto" común.
#   grados_por_paso = 360 / (pasos_motor_rev * microstep * reductor)
# El ángulo se obtiene SOLO del conteo de pasos (sin usar el encoder).

def _barrido_grados_por_paso():
    c = barrido_config
    pasos_rev_plato = float(c["pasos_motor_rev"]) * float(c["microstep"]) * float(c["reductor"])
    if pasos_rev_plato <= 0:
        return 0.0
    return 360.0 / pasos_rev_plato

def _rotar_puntos(pts, axis_point, axis_dir, ang_rad):
    """Rota pts (Nx3) un ángulo ang_rad alrededor del eje que pasa por axis_point
    con dirección axis_dir (Rodrigues). Devuelve Nx3."""
    if pts.shape[0] == 0:
        return pts
    a = np.asarray(axis_dir, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    p0 = np.asarray(axis_point, dtype=np.float64).reshape(3)
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]], dtype=np.float64)
    R = np.eye(3) + np.sin(ang_rad) * K + (1.0 - np.cos(ang_rad)) * (K @ K)
    return (R @ (pts - p0).T).T + p0

def mover_eje_pasos(pasos, direction, timeout=60.0):
    """Envía MOVE al Pico y espera la confirmación MOVE_DONE.

    Devuelve (ok: bool, pasos_realizados: int)."""
    global posicion_desde_home
    if pasos <= 0:
        return True, 0
    move_done_event.clear()
    resp = send_serial_command(f"MOVE {int(pasos)} {int(direction)}")
    if resp != "SENT":
        return False, 0
    ok = move_done_event.wait(timeout)
    realizados = last_move_info.get("pasos", 0)
    if ok:
        # dir 1 aleja de home (suma) ; dir 0 acerca a home (resta)
        posicion_desde_home += realizados if int(direction) == 1 else -realizados
    return ok, realizados

def ir_a_home(timeout=180.0):
    """Envía AUTO_HOME y espera a que el Pico confirme el fin del homing
    (línea 'HOMING COMPLETADO'). Devuelve True si homeó dentro del timeout."""
    home_done_event.clear()
    resp = send_serial_command("AUTO_HOME")
    if resp != "SENT":
        return False
    return home_done_event.wait(timeout)

def _resolucion_captura_efectiva():
    """Resolución que usará la captura: la de la calibración de cámara activa
    (image_size) si existe, o capture_resolution como fallback."""
    cap_w, cap_h = capture_resolution["width"], capture_resolution["height"]
    cam = perfil_activo.get("camara")
    if cam:
        ruta = os.path.join(CALIB_FOLDER, cam)
        if os.path.isfile(ruta):
            try:
                with open(ruta, encoding="utf-8") as f:
                    size = json.load(f).get("image_size")
                if size and len(size) == 2:
                    cap_w, cap_h = int(size[0]), int(size[1])
            except Exception:
                pass
    return cap_w, cap_h

@app.route('/ajustes/barrido')
def test_barrido():
    cap_w, cap_h = _resolucion_captura_efectiva()
    return render_template("test_barrido.html",
        perfil_camara=perfil_activo.get("camara"),
        perfil_laser=perfil_activo.get("laser"),
        perfil_plano=perfil_activo.get("plano"),
        perfil_eje=perfil_activo.get("eje"),
        config=barrido_config,
        grados_por_paso=_barrido_grados_por_paso(),
        captura_w=cap_w, captura_h=cap_h)

@app.route('/test_barrido/config', methods=['POST'])
def test_barrido_config():
    """Actualiza la geometría del eje (pasos/rev, microstep, reductor, sentido, eje)."""
    data = request.get_json(force=True)
    for k in ("pasos_motor_rev", "microstep", "reductor"):
        if k in data:
            barrido_config[k] = int(data[k])
    if "sentido" in data:
        barrido_config["sentido"] = 1 if int(data["sentido"]) >= 0 else -1
    if "axis_dir" in data and isinstance(data["axis_dir"], list) and len(data["axis_dir"]) == 3:
        barrido_config["axis_dir"] = [float(v) for v in data["axis_dir"]]
    # Modo del punto del eje: "auto" (centroide del 1er perfil) o "manual" (x,y,z)
    if "eje_modo" in data:
        barrido_config["eje_modo"] = "manual" if data["eje_modo"] == "manual" else "auto"
    if "axis_point" in data:
        ap = data["axis_point"]
        if isinstance(ap, list) and len(ap) == 3:
            barrido_config["axis_point"] = [float(v) for v in ap]
        elif ap is None:
            barrido_config["axis_point"] = None
    return jsonify({"success": True,
                    "config": barrido_config,
                    "grados_por_paso": _barrido_grados_por_paso()})

@app.route('/test_barrido/jog', methods=['POST'])
def test_barrido_jog():
    """Posiciona el eje moviendo N pasos en una dirección, SIN capturar ni
    acumular ángulo (solo para llevar el eje al punto de inicio)."""
    data = request.get_json(force=True)
    pasos = int(data.get("pasos", 0))
    direction = int(data.get("dir", 1))
    if pasos <= 0:
        return jsonify({"success": False, "message": "Pasos debe ser > 0"}), 400
    ok, realizados = mover_eje_pasos(pasos, direction)
    if not ok:
        return jsonify({"success": False,
                        "message": "El Pico no confirmó el movimiento (timeout)."}), 500
    return jsonify({"success": True, "pasos_realizados": realizados})

@app.route('/test_barrido/reset', methods=['POST'])
def test_barrido_reset():
    """Fija el origen del barrido: limpia la nube y pone el ángulo acumulado en 0."""
    global barrido_state
    barrido_state = {"points": [], "n_perfiles": 0, "angulo_acum_deg": 0.0}
    print("🔄 Barrido reiniciado (origen fijado).")
    return jsonify({"success": True})

@app.route('/test_barrido/barrer', methods=['POST'])
def test_barrido_barrer():
    """Ejecuta un barrido automático: captura un perfil inicial y luego, por cada
    incremento, mueve N pasos, captura, reconstruye 3D, rota por el ángulo
    acumulado del plato y acumula la nube.

    Body: { pasos_incremento, n_incrementos, dir, capturar_inicial }
    """
    global barrido_state, streaming_active
    data = request.get_json(force=True)
    pasos_inc = int(data.get("pasos_incremento", 0))
    n_inc = int(data.get("n_incrementos", 0))
    direction = int(data.get("dir", 1))
    capturar_inicial = bool(data.get("capturar_inicial", True))
    # streaming=True mantiene el preview vivo; False lo pausa para más fluidez.
    mantener_streaming = bool(data.get("streaming", True))
    # Referencia reproducible: ir a HOME y avanzar un offset fijo antes de barrer.
    hacer_home   = bool(data.get("ir_a_home", False))
    offset_pasos = int(data.get("offset_pasos", 0))
    offset_dir   = int(data.get("offset_dir", 1))

    # Barrido "real" usando el rango mecánico calibrado (HOME → offset → máximo).
    if bool(data.get("usar_mecanica", False)):
        recorrido = int(mecanica_config["max_pasos"]) - int(mecanica_config["offset_pasos"])
        if recorrido <= 0:
            return jsonify({"success": False,
                            "message": "Calibrá la mecánica (offset/máximo) primero en Ajustes › Mecánica."}), 400
        hacer_home   = True
        offset_pasos = int(mecanica_config["offset_pasos"])
        offset_dir   = 1
        direction    = 1                       # barre alejándose de home
        if pasos_inc > 0:
            n_inc = max(1, recorrido // pasos_inc)   # cubre el rango hasta el máximo

    if pasos_inc <= 0 or n_inc <= 0:
        return jsonify({"success": False,
                        "message": "pasos_incremento y n_incrementos deben ser > 0."}), 400

    ctx, err = _test_cargar_perfiles_activos()
    if err:
        return jsonify({"success": False, "message": err}), 400
    K, dist, laser_params, n, d = ctx

    # Si hay un perfil de eje activo, lo reaplicamos aquí para que el barrido use
    # SIEMPRE el eje calibrado, sin depender de los campos del formulario.
    if perfil_activo.get("eje"):
        _aplicar_eje_a_barrido(perfil_activo["eje"])

    # Pausar el streaming durante el barrido (si se pidió): libera la cámara y
    # evita los cambios de modo preview↔captura → barrido más fluido. Se
    # restaura SIEMPRE al terminar (try/finally), aunque haya error o timeout.
    streaming_prev = streaming_active
    if not mantener_streaming:
        streaming_active = False
        print("🚫 Streaming pausado durante el barrido")

    try:
        # Con streaming pausado dejamos el sensor FIJO en modo captura durante
        # todo el barrido (1 reconfiguración), y capturamos cada perfil sin
        # switch de modo. Con streaming activo, captura normal (switch por foto).
        if not mantener_streaming:
            with camera_lock:
                _still_cfg = picam2.create_still_configuration(
                    main={"format": "RGB888",
                          "size": (capture_resolution["width"], capture_resolution["height"])})
                picam2.switch_mode(_still_cfg)
            capturar = capture_frame_fast
            print(f"📷 Sensor fijo en modo captura "
                  f"{capture_resolution['width']}x{capture_resolution['height']} para el barrido")
        else:
            capturar = capture_high_res_frame

        barrido_stop_event.clear()
        grados_por_paso = _barrido_grados_por_paso()
        sentido = barrido_config["sentido"]
        perfiles_ok = 0
        errores = []

        # Inicializa el progreso (total = incrementos + perfil inicial opcional)
        total_perfiles = n_inc + (1 if capturar_inicial else 0)
        barrido_progreso.update(activo=True, actual=0, total=total_perfiles,
                                angulo_deg=0.0, num_puntos=0, fase="barrido")

        # ===================== REFERENCIA: HOME + OFFSET =====================
        # Para que cada barrido arranque desde la MISMA posición física (y los
        # planos sean comparables), opcionalmente vamos a HOME y avanzamos un
        # offset fijo. El origen del ángulo queda en home+offset.
        if hacer_home:
            barrido_state = {"points": [], "n_perfiles": 0, "angulo_acum_deg": 0.0}
            barrido_progreso.update(fase="home", actual=0, num_puntos=0, angulo_deg=0.0)
            print("🏠 Barrido: yendo a HOME...")
            if not ir_a_home(timeout=180.0):
                return jsonify({"success": False,
                                "message": "El Pico no confirmó el HOME (timeout)."}), 500
            if barrido_stop_event.is_set():
                return jsonify({"success": False,
                                "message": "Barrido detenido durante el HOME."}), 200
            if offset_pasos > 0:
                barrido_progreso["fase"] = "offset"
                print(f"➡️  Barrido: avanzando offset {offset_pasos} pasos (dir {offset_dir})...")
                ok_off, _ = mover_eje_pasos(offset_pasos, offset_dir)
                if not ok_off:
                    return jsonify({"success": False,
                                    "message": "El Pico no confirmó el offset (timeout)."}), 500
            barrido_progreso["fase"] = "barrido"

        def _actualizar_progreso():
            npts = sum(p.shape[0] for p in barrido_state["points"])
            barrido_progreso.update(actual=perfiles_ok,
                                    angulo_deg=round(barrido_state["angulo_acum_deg"], 3),
                                    num_puntos=int(npts))

        def _procesar_perfil_actual():
            """Captura, reconstruye y acumula el perfil en el ángulo acumulado."""
            nonlocal perfiles_ok
            success, frame_bgr = capturar()
            if not success:
                errores.append("captura fallida")
                return
            pts3d, _ = _test_reconstruir(frame_bgr, laser_params, K, dist, n, d)
            if pts3d.shape[0] == 0:
                errores.append(f"sin láser @ {barrido_state['angulo_acum_deg']:.2f}°")
                return
            # Punto del eje:
            #  - modo "manual": usa el axis_point fijado por el usuario (no se toca).
            #  - modo "auto": si aún no hay punto, lo fija al centroide del 1er perfil.
            if barrido_config.get("eje_modo") != "manual" and barrido_config["axis_point"] is None:
                barrido_config["axis_point"] = pts3d.mean(axis=0).tolist()
            if barrido_config["axis_point"] is None:  # fallback de seguridad
                barrido_config["axis_point"] = pts3d.mean(axis=0).tolist()
            ang_rad = np.deg2rad(sentido * barrido_state["angulo_acum_deg"])
            pts_rot = _rotar_puntos(pts3d, barrido_config["axis_point"],
                                    barrido_config["axis_dir"], ang_rad)
            barrido_state["points"].append(pts_rot)
            barrido_state["n_perfiles"] += 1
            perfiles_ok += 1

        # Perfil inicial (ángulo actual)
        if capturar_inicial:
            _procesar_perfil_actual()
            _actualizar_progreso()

        # Incrementos
        interrumpido = False
        for _ in range(n_inc):
            if barrido_stop_event.is_set():
                interrumpido = True
                errores.append("barrido detenido por el usuario")
                break
            ok, realizados = mover_eje_pasos(pasos_inc, direction)
            if not ok:
                errores.append("timeout MOVE")
                break
            barrido_state["angulo_acum_deg"] += realizados * grados_por_paso
            _procesar_perfil_actual()
            _actualizar_progreso()
            if barrido_stop_event.is_set():
                interrumpido = True
                errores.append("barrido detenido por el usuario")
                break

        barrido_progreso["activo"] = False

        if not barrido_state["points"]:
            return jsonify({"success": False,
                            "message": "No se acumuló ningún perfil. " + "; ".join(errores)}), 200

        nube = np.vstack(barrido_state["points"])
        print(f"🌀 Barrido: {perfiles_ok} perfiles · {nube.shape[0]} puntos · "
              f"ángulo total {barrido_state['angulo_acum_deg']:.2f}°")

        paso = max(1, nube.shape[0] // 4000)
        muestra = nube[::paso]

        return jsonify({
            "success": True,
            "perfiles": perfiles_ok,
            "num_puntos": int(nube.shape[0]),
            "angulo_total_deg": round(barrido_state["angulo_acum_deg"], 3),
            "grados_por_paso": grados_por_paso,
            "interrumpido": interrumpido,
            "errores": errores,
            "puntos": [[round(float(x), 2), round(float(y), 2), round(float(z), 2)]
                       for x, y, z in muestra],
        })
    finally:
        # Restaurar modo preview y streaming pase lo que pase
        barrido_progreso["activo"] = False
        barrido_progreso["fase"] = ""
        if not mantener_streaming:
            try:
                with camera_lock:
                    _prev_cfg = picam2.create_preview_configuration(
                        main={"format": "RGB888",
                              "size": (stream_resolution["width"], stream_resolution["height"])})
                    picam2.switch_mode(_prev_cfg)
            except Exception as e:
                print(f"⚠️ Error restaurando modo preview: {e}")
            streaming_active = streaming_prev
            print("✅ Modo preview y streaming restaurados tras el barrido")

@app.route('/test_barrido/stop', methods=['POST'])
def test_barrido_stop():
    """Solicita abortar el barrido en curso: activa el flag de parada y manda
    STOP al Pico para interrumpir un movimiento en progreso."""
    barrido_stop_event.set()
    send_serial_command("STOP")
    print("⛔ Barrido: parada solicitada por el usuario.")
    return jsonify({"success": True})

@app.route('/test_barrido/progreso', methods=['GET'])
def test_barrido_progreso():
    """Devuelve el progreso del barrido en curso (para la barra de progreso)."""
    return jsonify(barrido_progreso)

@app.route('/test_barrido/estado', methods=['GET'])
def test_barrido_estado():
    """Snapshot completo del estado del barrido para restaurar la página al
    entrar/recargar: si hay un barrido en curso, la nube acumulada y el ángulo.
    Así la página refleja el estado del servidor (no se pierde al navegar)."""
    pts_list = []
    num = 0
    if barrido_state["points"]:
        nube = np.vstack(barrido_state["points"])
        num = int(nube.shape[0])
        paso = max(1, num // 4000)
        pts_list = [[round(float(x), 2), round(float(y), 2), round(float(z), 2)]
                    for x, y, z in nube[::paso]]
    return jsonify({
        "activo":          barrido_progreso["activo"],
        "actual":          barrido_progreso["actual"],
        "total":           barrido_progreso["total"],
        "fase":            barrido_progreso.get("fase", ""),
        "angulo_deg":      round(barrido_state["angulo_acum_deg"], 3),
        "num_puntos":      num,
        "grados_por_paso": _barrido_grados_por_paso(),
        "puntos":          pts_list,
    })

@app.route('/test_barrido/guardar', methods=['POST'])
def test_barrido_guardar():
    """Guarda la nube acumulada como CSV (x,y,z) en static/modelos."""
    data = request.get_json(force=True)
    nombre = (data.get("nombre") or "").strip()
    if not barrido_state["points"]:
        return jsonify({"success": False, "message": "No hay nube para guardar."}), 400

    if not nombre:
        nombre = f"barrido_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not nombre.endswith(".csv"):
        nombre += ".csv"

    nube = np.vstack(barrido_state["points"])
    ruta = os.path.join(MAPS_FOLDER, nombre)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        for X, Y, Z in nube:
            writer.writerow([round(float(X), 3), round(float(Y), 3), round(float(Z), 3)])

    print(f"💾 Nube de barrido guardada: {ruta} ({nube.shape[0]} puntos)")
    return jsonify({"success": True, "nombre": nombre,
                    "csv_url": url_for('view_plot', nombre=nombre),
                    "num_puntos": int(nube.shape[0])})


@app.route('/api/set_active_profile', methods=['POST'])
def set_active_profile():
    global perfil_activo
    data = request.get_json(force=True)
    tipo = data.get("tipo")   # "camara", "laser", "plano" o "eje"
    nombre = data.get("nombre")  # nombre del archivo
    if tipo not in ("camara", "laser", "plano", "eje"):
        return jsonify({"success": False, "message": "Tipo inválido"}), 400
    perfil_activo[tipo] = nombre
    # Al cambiar un perfil intermedio, invalidar los pasos siguientes
    if tipo == "camara":
        perfil_activo["laser"] = None
        perfil_activo["plano"] = None
        perfil_activo["eje"] = None
        _resetear_eje_barrido()
        # Fijar la resolución de captura a la de calibración → K siempre coincide
        # con los píxeles capturados en plano/eje/medición/barrido.
        _aplicar_resolucion_camara(nombre)
    elif tipo == "laser":
        perfil_activo["plano"] = None
        perfil_activo["eje"] = None
        _resetear_eje_barrido()
    elif tipo == "plano":
        perfil_activo["eje"] = None
        _resetear_eje_barrido()
    elif tipo == "eje":
        # Cargar el eje calibrado en la geometría del barrido (modo manual)
        _aplicar_eje_a_barrido(nombre)
    print(f"✅ Perfil activo '{tipo}' cambiado a: {nombre}")
    return jsonify({"success": True, "perfil_activo": perfil_activo})

def _aplicar_eje_a_barrido(nombre):
    """Carga axis_point/axis_dir de un perfil de eje y los fija en barrido_config
    en modo 'manual', de modo que el ensamblado del barrido use el eje calibrado
    en vez del centroide del primer perfil."""
    if not nombre:
        return
    ruta = os.path.join(EJE_FOLDER, nombre)
    if not os.path.isfile(ruta):
        print(f"⚠️ Perfil de eje no encontrado: {nombre}")
        return
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
        barrido_config["axis_point"] = [float(v) for v in data["axis_point"]]
        barrido_config["axis_dir"]   = [float(v) for v in data["axis_dir"]]
        barrido_config["eje_modo"]   = "manual"
        print(f"🧭 Eje aplicado al barrido: punto={barrido_config['axis_point']}, "
              f"dir={barrido_config['axis_dir']}")
    except Exception as e:
        print(f"⚠️ Error aplicando eje al barrido: {e}")

def _resetear_eje_barrido():
    """Vuelve el barrido a modo automático (eje = centroide del 1er perfil),
    usado cuando se invalida el perfil de eje activo."""
    barrido_config["eje_modo"] = "auto"
    barrido_config["axis_point"] = None
    barrido_config["axis_dir"] = [0.0, 1.0, 0.0]

def _aplicar_resolucion_camara(nombre):
    """Fija la resolución de captura a la usada al calibrar la cámara (image_size
    del JSON). Así K coincide con los píxeles capturados en plano/eje/medición/
    barrido y la triangulación es consistente, sin reescalar K (que fallaría si
    cambia el modo/recorte del sensor)."""
    if not nombre:
        return
    ruta = os.path.join(CALIB_FOLDER, nombre)
    if not os.path.isfile(ruta):
        return
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
        size = data.get("image_size")
        if size and len(size) == 2:
            capture_resolution["width"]  = int(size[0])
            capture_resolution["height"] = int(size[1])
            print(f"📐 Resolución de captura fijada a la de calibración: "
                  f"{int(size[0])}x{int(size[1])}")
    except Exception as e:
        print(f"⚠️ No se pudo fijar la resolución de captura: {e}")

@app.route('/api/eliminar_perfil', methods=['POST'])
def eliminar_perfil():
    """Elimina un archivo de perfil. Body: { tipo, nombre }.
    tipo ∈ {camara, laser, plano, eje}. Si era el perfil activo, lo limpia."""
    data = request.get_json(force=True)
    tipo = data.get("tipo")
    nombre = (data.get("nombre") or "").strip()
    folders = {"camara": CALIB_FOLDER, "laser": LASER_FOLDER,
               "plano": PLANO_FOLDER, "eje": EJE_FOLDER}
    if tipo not in folders:
        return jsonify({"success": False, "message": "Tipo inválido"}), 400
    # Seguridad: solo el nombre base y que termine en .json (evita path traversal)
    nombre = os.path.basename(nombre)
    if not nombre.endswith(".json"):
        return jsonify({"success": False, "message": "Nombre inválido"}), 400
    ruta = os.path.join(folders[tipo], nombre)
    if not os.path.isfile(ruta):
        return jsonify({"success": False, "message": "No existe el archivo"}), 404
    try:
        os.remove(ruta)
    except Exception as e:
        return jsonify({"success": False, "message": f"Error eliminando: {e}"}), 500
    # Si era el perfil activo, limpiarlo (y revertir el eje del barrido si aplica)
    if perfil_activo.get(tipo) == nombre:
        perfil_activo[tipo] = None
        if tipo == "eje":
            _resetear_eje_barrido()
    print(f"🗑️ Perfil {tipo} eliminado: {nombre}")
    return jsonify({"success": True})

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

    with camera_lock:
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

    # Aviso si no coincide con la resolución de la calibración de cámara activa:
    # la triangulación (plano/eje/medición/barrido) la re-fijará a la de calib.
    warning = None
    cam = perfil_activo.get("camara")
    if cam:
        ruta = os.path.join(CALIB_FOLDER, cam)
        if os.path.isfile(ruta):
            try:
                with open(ruta, "r", encoding="utf-8") as f:
                    size = json.load(f).get("image_size")
                if size and (int(size[0]), int(size[1])) != (width, height):
                    warning = (f"Esta resolución ({width}x{height}) no coincide con la de la "
                               f"calibración de cámara activa ({int(size[0])}x{int(size[1])}). "
                               "Para medir/barrer se usará la de la calibración.")
                    print(f"⚠️ {warning}")
            except Exception:
                pass

    return jsonify({
        "success": True,
        "message": f"Resolución de captura configurada a {width}x{height}",
        "warning": warning,
    })

@app.route('/camera/test_capture', methods=['POST'])
def test_capture():
    print("📷 Realizando captura de prueba...")
    success, frame_bgr = capture_high_res_frame()

    if not success:
        return jsonify({"success": False, "message": "Error al capturar imagen"})

    filename = "preview_capture.jpg"
    path = os.path.join(CAPTURAS_FOLDER, filename)
    cv2.imwrite(path, frame_bgr)

    actual_height, actual_width = frame_bgr.shape[:2]
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
    success, frame_bgr = capture_high_res_frame()

    if not success:
        return jsonify({"success": False, "message": "Error al capturar imagen"})

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
    success, frame_bgr = capture_high_res_frame()
    if not success:
        return jsonify({"success": False, "message": "Error al acceder a la cámara"})

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
    success, frame_bgr = capture_high_res_frame()

    if not success:
        return jsonify({"success": False})

    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(CAPTURAS_FOLDER, filename)
    cv2.imwrite(path, frame_bgr)

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

# ======================== CALIBRACIÓN MECÁNICA (RANGO DESDE HOME) ==========================
@app.route('/ajustes/mecanica')
def ajustes_mecanica():
    return render_template('ajustes_mecanica.html', mecanica=mecanica_config)

@app.route('/api/mecanica/estado', methods=['GET'])
def api_mecanica_estado():
    return jsonify({
        "posicion":      posicion_desde_home,
        "referenciado":  home_referenciado,
        "offset_pasos":  mecanica_config["offset_pasos"],
        "max_pasos":     mecanica_config["max_pasos"],
    })

@app.route('/api/mecanica/home', methods=['POST'])
def api_mecanica_home():
    """Va a HOME (referencia) y deja la posición en 0."""
    ok = ir_a_home(timeout=180.0)
    return jsonify({"success": ok,
                    "message": "Home OK" if ok else "El Pico no confirmó el HOME (timeout).",
                    "posicion": posicion_desde_home})

@app.route('/api/mecanica/guardar', methods=['POST'])
def api_mecanica_guardar():
    """Guarda el rango mecánico (offset y máximo, en pasos desde HOME)."""
    global mecanica_config
    data = request.get_json(force=True)
    off = int(data.get("offset_pasos", mecanica_config["offset_pasos"]))
    mx  = int(data.get("max_pasos",    mecanica_config["max_pasos"]))
    if off < 0 or mx <= off:
        return jsonify({"success": False,
                        "message": "El máximo debe ser mayor que el offset (y el offset ≥ 0)."}), 400
    mecanica_config["offset_pasos"] = off
    mecanica_config["max_pasos"]    = mx
    try:
        with open(MECANICA_FILE, "w", encoding="utf-8") as f:
            json.dump(mecanica_config, f, indent=2)
    except Exception as e:
        return jsonify({"success": False, "message": f"Error guardando: {e}"}), 500
    print(f"💾 Mecánica guardada: offset={off}, max={mx}")
    return jsonify({"success": True, "mecanica": mecanica_config})

# ======================== PROGRAMACIÓN DE BARRIDOS ==========================
# Barridos programados (intervalo o fecha puntual), persistidos en JSON. Cuando
# uno está vencido y el usuario está en la web, se le muestra un modal con cuenta
# de 30 s para ejecutar o abortar. Al ejecutar se registra en InfluxDB el archivo
# y la hora. El barrido usa los perfiles activos + el rango mecánico calibrado.
PROG_FILE = os.path.join("static", "programacion.json")
programacion = []

def _cargar_programacion():
    global programacion
    try:
        if os.path.isfile(PROG_FILE):
            with open(PROG_FILE, encoding="utf-8") as f:
                programacion = json.load(f)
    except Exception as e:
        print(f"⚠️ No se pudo cargar {PROG_FILE}: {e}")
        programacion = []

def _guardar_programacion():
    try:
        with open(PROG_FILE, "w", encoding="utf-8") as f:
            json.dump(programacion, f, indent=2)
    except Exception as e:
        print(f"⚠️ No se pudo guardar {PROG_FILE}: {e}")

_cargar_programacion()

def _job_por_id(jid):
    try:
        jid = int(jid)
    except Exception:
        return None
    for j in programacion:
        if j.get("id") == jid:
            return j
    return None

def _proxima_pendiente():
    ahora = time.time()
    for j in programacion:
        if j.get("activo") and j.get("proximo_epoch") and j["proximo_epoch"] <= ahora:
            return j
    return None

def _avanzar_job(job, dt=None):
    ahora = int((dt or datetime.now()).timestamp())
    job["ultimo_epoch"] = ahora
    if job.get("tipo") == "intervalo":
        job["proximo_epoch"] = ahora + max(1, int(job.get("intervalo_min", 60))) * 60
    else:
        job["activo"] = False   # fecha puntual: ya se ejecutó

# Writer InfluxDB perezoso (para registrar los barridos ejecutados)
_influx_writer = None
def _get_influx_writer():
    global _influx_writer
    if _influx_writer is None:
        _influx_writer = InfluxWriter()
    return _influx_writer

@app.route('/programacion')
def programacion_page():
    return render_template('programacion.html')

@app.route('/api/programacion/listar', methods=['GET'])
def programacion_listar():
    return jsonify({"jobs": programacion, "ahora": int(time.time())})

@app.route('/api/programacion/agregar', methods=['POST'])
def programacion_agregar():
    data = request.get_json(force=True)
    nombre = (data.get("nombre") or "Barrido").strip()
    tipo = data.get("tipo")
    pasos_inc = int(data.get("pasos_incremento", 1000))
    nid = max([j.get("id", 0) for j in programacion], default=0) + 1
    job = {"id": nid, "nombre": nombre, "tipo": tipo, "pasos_incremento": pasos_inc,
           "activo": True, "ultimo_epoch": None}
    if tipo == "intervalo":
        mins = int(data.get("intervalo_min", 60))
        if mins <= 0:
            return jsonify({"success": False, "message": "Intervalo inválido"}), 400
        job["intervalo_min"] = mins
        job["proximo_epoch"] = int(time.time()) + mins * 60
    elif tipo == "fecha":
        ep = int(float(data.get("fecha_epoch", 0)))
        if ep <= 0:
            return jsonify({"success": False, "message": "Fecha inválida"}), 400
        job["fecha_epoch"] = ep
        job["proximo_epoch"] = ep
    else:
        return jsonify({"success": False, "message": "Tipo inválido"}), 400
    programacion.append(job)
    _guardar_programacion()
    return jsonify({"success": True, "job": job})

@app.route('/api/programacion/eliminar', methods=['POST'])
def programacion_eliminar():
    global programacion
    job = _job_por_id(request.get_json(force=True).get("id"))
    if job:
        programacion = [j for j in programacion if j is not job]
        _guardar_programacion()
    return jsonify({"success": True})

@app.route('/api/programacion/toggle', methods=['POST'])
def programacion_toggle():
    job = _job_por_id(request.get_json(force=True).get("id"))
    if not job:
        return jsonify({"success": False}), 404
    job["activo"] = not job.get("activo", True)
    if job["activo"] and job.get("tipo") == "intervalo":
        job["proximo_epoch"] = int(time.time()) + int(job.get("intervalo_min", 60)) * 60
    _guardar_programacion()
    return jsonify({"success": True, "activo": job["activo"]})

@app.route('/api/programacion/pendiente', methods=['GET'])
def programacion_pendiente():
    if barrido_progreso.get("activo"):
        return jsonify({"pendiente": None})
    job = _proxima_pendiente()
    if not job:
        return jsonify({"pendiente": None})
    return jsonify({"pendiente": {"id": job["id"], "nombre": job["nombre"],
                                  "pasos_incremento": job.get("pasos_incremento", 1000)}})

@app.route('/api/programacion/ejecutado', methods=['POST'])
def programacion_ejecutado():
    """Registra en InfluxDB un barrido programado ya ejecutado y avanza el job."""
    data = request.get_json(force=True)
    job = _job_por_id(data.get("id"))
    archivo = (data.get("archivo") or "").strip()
    ahora = datetime.now()
    registrado = False
    try:
        _get_influx_writer().write_evento(
            "barridos",
            fields={"archivo": archivo, "num_puntos": int(data.get("num_puntos", 0) or 0)},
            tags={"job": job["nombre"] if job else "?"},
            ts=ahora)
        registrado = True
    except Exception as e:
        print(f"⚠️ No se pudo registrar el barrido en InfluxDB: {e}")
    if job:
        _avanzar_job(job, ahora)
        _guardar_programacion()
    return jsonify({"success": True, "registrado": registrado, "hora": ahora.strftime("%H:%M")})

@app.route('/api/programacion/abortar', methods=['POST'])
def programacion_abortar():
    job = _job_por_id(request.get_json(force=True).get("id"))
    if job:
        _avanzar_job(job, datetime.now())
        _guardar_programacion()
    return jsonify({"success": True})

# ======================== DASHBOARD DE SENSORES (InfluxDB) ==========================
_influx_reader = None

def _get_influx_reader():
    """Lector InfluxDB perezoso (se crea en el primer uso)."""
    global _influx_reader
    if _influx_reader is None:
        _influx_reader = InfluxReader()
    return _influx_reader

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/sensores/ultimos')
def api_sensores_ultimos():
    """Último valor de cada campo (para el Home)."""
    try:
        return jsonify({"success": True, "valores": _get_influx_reader().ultimos_valores()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 200

@app.route('/api/sensores/datos')
def api_sensores_datos():
    """Series recientes para los gráficos. ?minutos=60&cada=30s"""
    minutos = int(request.args.get('minutos', 60))
    cada = request.args.get('cada', '30s')
    try:
        data = _get_influx_reader().datos_recientes(minutos=minutos, cada=cada)
        return jsonify({"success": True, "data": data, "minutos": minutos})
    except Exception as e:
        # 200 para que el dashboard muestre un aviso en vez de tirar error
        return jsonify({"success": False, "message": str(e)}), 200

@app.route('/api/sensores/reporte')
def api_sensores_reporte():
    """Descarga CSV del rango pedido. ?inicio=<epoch>&fin=<epoch>&cada=1m"""
    inicio = request.args.get('inicio')
    fin = request.args.get('fin')
    cada = request.args.get('cada', '1m')
    if not inicio or not fin:
        return jsonify({"success": False, "message": "Indicá inicio y fin."}), 400
    try:
        headers, filas = _get_influx_reader().reporte_rango(inicio, fin, cada=cada)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 200

    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerows(filas)
    nombre = f"reporte_sensores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={nombre}"})

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
            with camera_lock:
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
