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
import serial # <<--- AÑADIDO: Para comunicación serial

# ======================== CONFIGURACIÓN SERIAL ==========================
# Reemplaza '/dev/ttyACM0' por el puerto correcto de tu Pico si es diferente.
# Puede ser /dev/ttyS0 si usas pines GPIO TX/RX. Si es USB, suele ser /dev/ttyACM0 o /dev/ttyUSB0
SERIAL_PORT = '/dev/ttyACM0' 
BAUDRATE = 115200 

try:
    # Inicializa la conexión serial
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1) 
    print(f"✅ Conexión serial establecida en {SERIAL_PORT} a {BAUDRATE} baudios.")
    # Esperar un momento para que el Pico se reinicie y esté listo
    time.sleep(2) 
except serial.SerialException as e:
    print(f"❌ Error al conectar al puerto serial {SERIAL_PORT}: {e}")
    ser = None
# ========================================================================


app = Flask(__name__)

MAPS_FOLDER = os.path.join("static", "modelos")
CALIB_FOLDER = os.path.join("static", "calibraciones")
CAPTURAS_FOLDER = os.path.join("static", "capturas")
COLOR_FOLDER = os.path.join("static", "calibraciones_color")

os.makedirs(CAPTURAS_FOLDER, exist_ok=True)
os.makedirs(CALIB_FOLDER, exist_ok=True)

# ----------------- CONFIGURACIÓN DE CÁMARA RASPBERRY PI -----------------
stream_resolution = {"width": 1280, "height": 720}  
capture_resolution = {"width": 1920, "height": 1080} 

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
step_counter = 0 # Ahora representa el ángulo o posición que se usará para el loop
step_size = 20 # Grados a mover en cada paso (LEFT/RIGHT)

# ----------------- FUNCIONES DE COMUNICACIÓN SERIAL -----------------

def send_serial_command(command):
    """Envía un comando al Pico y espera una respuesta (opcional)."""
    global ser
    if ser is None:
        return "ERROR_SERIAL_OFFLINE"

    try:
        # Enviar comando con salto de línea
        ser.write(f"{command}\n".encode('utf-8')) 
        print(f"<- Comando enviado: {command}")
        
        # Esperar respuesta (el Pico debe enviar una línea de respuesta)
        # Bucle para leer la respuesta. La respuesta esperada debe ser simple, ej: "OK" o "ANGULO: 15.5"
        
        # Solo leemos si esperamos una respuesta, como READ o después de un movimiento
        if command in ["LEFT", "RIGHT", "SET_ANGLE", "LOOP", "STOP", "READ"]:
            response = ser.readline().decode('utf-8').strip()
            print(f"-> Respuesta recibida: {response}")
            return response
        
        return "OK"

    except Exception as e:
        print(f"⚠️ Error en comunicación serial: {e}")
        return f"ERROR: {e}"

# ----------------- FUNCIONES DE CÁMARA (sin cambios significativos) -----------------

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
    # ... (código de captura de alta resolución de la versión anterior) ...
    # Se mantiene igual, ya que solo se encarga de la Picamera2
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

# ----------------- RUTAS DE CONTROL MODIFICADAS -----------------

@app.route('/action/<cmd>', methods=['POST'])
def control_action(cmd):
    global step_counter
    
    # Aquí es donde enviamos los comandos al Pico y actualizamos el step_counter 
    # (que ahora es el ángulo, si el Pico lo devuelve)
    
    if cmd == "left":
        print(f"🟢 Botón izquierda presionado (paso: {step_size} grados)")
        # Enviar comando al Pico
        response = send_serial_command("LEFT") 
        
    elif cmd == "right":
        print(f"🟢 Botón derecha presionado (paso: {step_size} grados)")
        # Enviar comando al Pico
        response = send_serial_command("RIGHT")

    # Si la acción fue de movimiento, intenta obtener la posición final del Pico
    if cmd in ["left", "right"]:
        try:
            # Comando extra para obtener el ángulo
            angle_response = send_serial_command("READ")
            if angle_response.startswith("ANGULO:"):
                 # La respuesta esperada del Pico es: "ANGULO: 123.45"
                step_counter = float(angle_response.split(":")[1].strip())
            else:
                step_counter = -1 # Error de lectura

        except Exception as e:
            print(f"Error al leer ángulo: {e}")
            step_counter = -2
        
        return jsonify({"success": True, "step": step_counter, "message": response})
        
    elif cmd == "capture":
        print("🟢 Botón capture presionado")
        # Antes de capturar, leemos la posición actual
        angle_response = send_serial_command("READ")
        current_step = step_counter
        
        if angle_response.startswith("ANGULO:"):
            current_step = float(angle_response.split(":")[1].strip())

        success, frame_rgb = capture_high_res_frame()
        if success:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Usamos el ángulo en el nombre del archivo si está disponible
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
        response = send_serial_command("LOOP")
        return jsonify({"success": True, "step": step_counter, "message": response})
        
    elif cmd == "stop":
        print("🔴 Botón STOP presionado")
        response = send_serial_command("STOP")
        return jsonify({"success": True, "step": step_counter, "message": response})
        
    return jsonify({"success": True, "step": step_counter})


@app.route('/set_step_size', methods=['POST'])
def set_step_size_route():
    global step_size
    data = request.get_json()
    step_size = int(data.get('step_size', 20))
    print(f"🔧 Tamaño de paso configurado (grados): {step_size}")
    
    # Enviar el nuevo tamaño de paso al Pico (SET_ANGLE)
    response = send_serial_command(f"SET_ANGLE {step_size}")
    
    return jsonify({'status': 'success', 'step_size': step_size, 'message': response})

# ----------------- ... (El resto del código de la App Web se mantiene igual) ... -----------------

# (El resto de las rutas de la App Web)

# Asegúrate de incluir el código de liberación de la cámara y el puerto serial
if __name__ == '__main__':
    try:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    except Exception as e:
        print(f"Error al iniciar la aplicación: {e}")
    finally:
        if picam2.started:
            picam2.stop()
        if ser is not None and ser.is_open:
            ser.close()
            print("Serial port closed.")
        cv2.destroyAllWindows()