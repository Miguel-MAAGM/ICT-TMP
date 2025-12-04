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
import threading


# ======================== CONFIGURACIÓN GENERAL ==========================
SERIAL_PORT = '/dev/ttyACM0' 
BAUDRATE = 115200 
ser = None


app = Flask(__name__)

MAPS_FOLDER = os.path.join("static", "modelos")
CALIB_FOLDER = os.path.join("static", "calibraciones")
CAPTURAS_FOLDER = os.path.join("static", "capturas")
COLOR_FOLDER = os.path.join("static", "calibraciones_color")
FOTOS_LOOP_FOLDER = os.path.join("static", "fotos_loop")

os.makedirs(CAPTURAS_FOLDER, exist_ok=True)
os.makedirs(CALIB_FOLDER, exist_ok=True)
os.makedirs(FOTOS_LOOP_FOLDER, exist_ok=True)

# ----------------- CONFIGURACIÓN DE RESOLUCIONES -----------------
stream_resolution = {"width": 1280, "height": 720} 
capture_resolution = {"width": 1920, "height": 1080} 

# ----------------- INICIALIZACIÓN DE LA CÁMARA RASPBERRY PI -----------------
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
roi_zoom_level = 1.5

# Variables para Calibración Automática
is_auto_calibrating = False
auto_calib_config = {"rows": 6, "cols": 7}
last_auto_capture_time = 0
MIN_TIME_BETWEEN_CAPTURES = 2.0

SERIAL_LOG = []
MAX_LOG_LINES = 150

# ----------------- VARIABLES GLOBALES ADICIONALES -----------------
calibration_images = []
objpoints = []
imgpoints = []
camera_lock = threading.Lock()

# ======================== FUNCIONES AUXILIARES DE MENSAJES ==========================

def format_user_message(raw_response, action_type, angle=None):
    """
    Convierte respuestas técnicas en mensajes amigables para el usuario.
    """
    if raw_response.startswith("ANGULO:"):
        try:
            angle_val = float(raw_response.split(":")[1].strip())
            if action_type == "left":
                return f"Motor girado a la izquierda. Posición actual: {angle_val:.2f}°"
            elif action_type == "right":
                return f"Motor girado a la derecha. Posición actual: {angle_val:.2f}°"
            elif action_type == "stop":
                return f"Motor detenido en posición: {angle_val:.2f}°"
            else:
                return f"Posición actual del motor: {angle_val:.2f}°"
        except:
            return "Movimiento completado"
    
    elif raw_response == "OK":
        if action_type == "loop":
            return "Ciclo continuo iniciado correctamente"
        elif action_type == "loop_capture":
            return "Ciclo con captura automática iniciado"
        elif action_type == "set_angle":
            return f"Tamaño de paso configurado a {angle}°"
        else:
            return "Comando ejecutado correctamente"
    
    elif raw_response.startswith("ERROR_"):
        error_type = raw_response.replace("ERROR_", "")
        return f"Error: {error_type}. Verifica la conexión y vuelve a intentar"
    
    elif raw_response == "NO_RESPONSE":
        return "Sin respuesta del motor. Verifica la conexión serial"
    
    elif raw_response == "ERROR_SERIAL_OFFLINE":
        return "Motor desconectado. Reconecta el dispositivo"
    
    else:
        # Para cualquier otro mensaje, dejarlo pasar
        return raw_response


def add_serial_log(msg):
    SERIAL_LOG.append(msg)
    if len(SERIAL_LOG) > MAX_LOG_LINES:
        SERIAL_LOG.pop(0)


@app.route("/serial_log")
def serial_log_page():
    return "<br>".join(SERIAL_LOG)


# ======================== SERIAL LISTENER ==========================

def serial_listener():
    global ser
    print("🔵 Listener serial iniciado…")

    while True:
        try:
            if ser is None or not ser.is_open:
                time.sleep(0.1)
                continue

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
            if line.startswith("FOTO"):
                try:
                    valor = float(line.split()[1])
                except:
                    valor = 0.0

                success, frame_rgb = capture_high_res_frame()
                
                if success:
                    frame_bgr = frame_rgb 
                    filename = f"foto_{valor:.2f}.jpg"
                    path = os.path.join(FOTOS_LOOP_FOLDER, filename)
                    cv2.imwrite(path, frame_bgr)

                    log = f"📸 Foto capturada automáticamente: {filename} (Ángulo: {valor:.2f}°)"
                    print(log)
                    add_serial_log(log)
                else:
                    print("⚠️ Error en captura automática, continuando...")

                # Handshake con la Pico
                if ser and ser.is_open:
                    ser.write(b"SIGUIENTE\n")
                    ser.flush()
                    print("✅ Señal enviada al motor para continuar")

        except Exception as e:
            print(f"[ERROR SERIAL LISTENER] {e}")
            add_serial_log(f"[ERROR] Listener serial: {e}")
            time.sleep(0.1)


# ----------------- FUNCIONES DE COMUNICACIÓN SERIAL -----------------

def send_serial_command(command, wait_for_angle=False, wait_for_ok=False, timeout=3.0):
    global ser

    if ser is None:
        return "ERROR_SERIAL_OFFLINE"

    try:
        ser.reset_input_buffer()
        ser.write((command + "\n").encode("utf-8"))
        print(f"<- Comando enviado: {command}")

        start_time = time.time()
        last_line = ""

        while True:
            line_bytes = ser.readline()
            if not line_bytes:
                if time.time() - start_time > timeout:
                    print("⚠️ Timeout esperando respuesta del motor")
                    break
                continue

            line = line_bytes.decode("utf-8", errors="ignore").strip()
            if line == "":
                continue

            print(f"-> Respuesta recibida: {line}")
            last_line = line

            if wait_for_ok and (line == "OK" or line.startswith("ERROR_")):
                return line

            if wait_for_angle and line.startswith("ANGULO:"):
                return line

            if not wait_for_ok and not wait_for_angle:
                return line

        return last_line or "NO_RESPONSE"

    except Exception as e:
        print(f"⚠️ Error en comunicación serial: {e}")
        try:
            ser.close()
        except:
            pass
        ser = None
        return f"ERROR: {e}"


# ----------------- FUNCIONES DE CÁMARA -----------------

def gen_frames():
    global is_auto_calibrating, last_auto_capture_time, calibration_images, objpoints, imgpoints
    
    while True:
        frame = picam2.capture_array()
        if frame is not None:
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
                            clean_bgr = frame
                            objp = np.zeros((rows * cols, 3), np.float32)
                            objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
                            
                            objpoints.append(objp)
                            imgpoints.append(corners2)
                            
                            img_with_corners = clean_bgr.copy()
                            cv2.drawChessboardCorners(img_with_corners, (cols, rows), corners2, found)
                            
                            filename = f"calib_auto_{len(calibration_images)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            path = os.path.join(CAPTURAS_FOLDER, filename)
                            cv2.imwrite(path, img_with_corners)
                            
                            calibration_images.append(path)
                            last_auto_capture_time = time.time()
                            print(f"✅ [AUTO] Captura guardada: {filename} ({len(calibration_images)} total)")
                    
                    frame = frame_bgr
                    
                except Exception as e:
                    print(f"Error en auto-calibración: {e}")

            if not is_auto_calibrating:
                 frame = frame

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


def gen_frames_roi():
    global roi_zoom_level
    OUTPUT_SIZE = (1920, 1080)

    while True:
        frame = picam2.capture_array()
        
        if frame is not None:
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
            except Exception as e:
                pass
        else:
            time.sleep(0.01)


# ----------------- RUTAS DE FLASK -----------------

@app.route('/video_feed_roi')
def video_feed_roi():
    return Response(gen_frames_roi(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_roi_zoom', methods=['POST'])
def set_roi_zoom():
    global roi_zoom_level
    data = request.get_json()
    try:
        new_zoom = float(data.get('zoom', 1.0))
        roi_zoom_level = max(1.0, min(5.0, new_zoom))
        print(f"🔍 Zoom ROI cambiado a: {roi_zoom_level}x")
        return jsonify({"success": True, "zoom": roi_zoom_level})
    except ValueError:
        return jsonify({"success": False})


@app.route('/monitor_roi')
def monitor_roi():
    return render_template("roi_monitor.html")


@app.route('/camera/set_mode_4k', methods=['POST'])
def set_mode_4k():
    global stream_resolution
    data = request.get_json()
    enable_4k = data.get('enable', False)
    
    if enable_4k:
        new_res = {"width": 3840, "height": 2160}
        print("🚀 Cambiando a MODO 4K...")
    else:
        new_res = {"width": 1280, "height": 720}
        print("🍃 Cambiando a MODO NORMAL...")

    with camera_lock:
        try:
            if picam2.started:
                picam2.stop()
            
            stream_resolution = new_res
            
            config = picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (stream_resolution["width"], stream_resolution["height"])}
            )
            picam2.configure(config)
            picam2.start()
            print(f"✅ Cámara reiniciada en: {stream_resolution['width']}x{stream_resolution['height']}")
            
            return jsonify({"success": True, "resolution": stream_resolution})
        except Exception as e:
            print(f"❌ Error cambiando resolución: {e}")
            return jsonify({"success": False, "message": str(e)})


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
    frame_bgr = frame
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
    input_size = data.get('chessboard_size', [7, 6])
    
    print(f"📸 Capturando imagen de calibración. Tamaño solicitado: {input_size}")
    
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
    else:
        print("❌ No se pudo detectar el patrón de tablero de ajedrez")
        return jsonify({
            "success": False,
            "message": f"No se detectó el patrón. Probado con {sizes_to_try[0]} y {sizes_to_try[1]}. Asegúrate de que el tablero esté iluminado y visible."
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
    data = request.get_json()
    
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


# ======================== RUTAS DE CONTROL DE MOTOR (MEJORADAS) ==========================

@app.route('/action/<cmd>', methods=['POST'])
def control_action(cmd):
    global step_counter
    
    if cmd == "left":
        print(f"🟢 Rotación izquierda iniciada (paso: {step_size}°)")
        raw_response = send_serial_command("LEFT", wait_for_angle=True)
        
        if raw_response.startswith("ANGULO:"):
            try:
                step_counter = float(raw_response.split(":")[1].strip())
            except:
                step_counter = -1
        
        user_message = format_user_message(raw_response, "left")
        return jsonify({"success": True, "step": step_counter, "message": user_message})
        
    elif cmd == "right":
        print(f"🟢 Rotación derecha iniciada (paso: {step_size}°)")
        raw_response = send_serial_command("RIGHT", wait_for_angle=True)
        
        if raw_response.startswith("ANGULO:"):
            try:
                step_counter = float(raw_response.split(":")[1].strip())
            except:
                step_counter = -1
        
        user_message = format_user_message(raw_response, "right")
        return jsonify({"success": True, "step": step_counter, "message": user_message})

    elif cmd == "capture":
        print("🟢 Captura manual solicitada")
        
        angle_response = send_serial_command("READ", wait_for_angle=True)
        current_step = step_counter
        
        if angle_response.startswith("ANGULO:"):
            try:
                current_step = float(angle_response.split(":")[1].strip())
            except:
                pass
        
        success, frame_rgb = capture_high_res_frame()
        if success:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            filename = f"capture_{current_step:.2f}deg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = os.path.join(CAPTURAS_FOLDER, filename)
            cv2.imwrite(path, frame_bgr)
            
            return jsonify({
                "success": True,
                "step": current_step,
                "message": f"Imagen capturada en ángulo {current_step:.2f}°",
                "url": url_for('static', filename=f"capturas/{filename}")
            })
        
        return jsonify({
            "success": False, 
            "step": current_step,
            "message": "Error al capturar la imagen"
        })
    
    elif cmd == "start_loop":
        print("🟢 Ciclo continuo iniciado")
        raw_response = send_serial_command("LOOP")
        user_message = format_user_message(raw_response, "loop")
        return jsonify({"success": True, "step": step_counter, "message": user_message})
    
    elif cmd == "start_loop_capture":
        print("🟢 Ciclo con captura automática iniciado")
        raw_response = send_serial_command("LOOP_CAPTURE")
        user_message = format_user_message(raw_response, "loop_capture")
        return jsonify({"success": True, "message": user_message})
        
    elif cmd == "stop":
        print("🔴 Detención solicitada")
        raw_response = send_serial_command("STOP", wait_for_angle=True)
        
        if raw_response.startswith("ANGULO:"):
            try:
                step_counter = float(raw_response.split(":")[1].strip())
            except:
                pass
        
        user_message = format_user_message(raw_response, "stop")
        return jsonify({"success": True, "step": step_counter, "message": user_message})
   
    return jsonify({"success": True, "step": step_counter})


@app.route('/set_step_size', methods=['POST'])
def set_step_size_route():
    global step_size
    data = request.get_json()
    step_size = int(data.get('step_size', 20))
    print(f"🔧 Configurando tamaño de paso: {step_size}°")
    
    raw_response = send_serial_command(f"SET_ANGLE {step_size}", wait_for_ok=True)
    user_message = format_user_message(raw_response, "set_angle", angle=step_size)
    
    return jsonify({
        'status': 'success', 
        'step_size': step_size, 
        'message': user_message
    })


# ======================== RUTAS ADICIONALES ==========================

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
    frame_bgr = frame_rgb
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


# ======================== INICIALIZACIÓN ==========================

if __name__ == '__main__':
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        print(f"✅ Conexión serial establecida en {SERIAL_PORT}")
        time.sleep(2)
    except Exception as e:
        print(f"❌ Error de conexión serial: {e}")
        print("⚠️ La aplicación iniciará sin control de motor")
        ser = None

    listener = threading.Thread(target=serial_listener, daemon=True)
    listener.start()
    print("🔵 Sistema de monitoreo serial iniciado")

    try:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    finally:
        if picam2.started:
            picam2.stop()
        if ser is not None and ser.is_open:
            ser.close()
            print("Puerto serial cerrado")
