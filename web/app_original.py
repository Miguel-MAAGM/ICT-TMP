import os
import csv
from picamera2 import Picamera2
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
from datetime import datetime
import numpy as np

app = Flask(__name__)

MAPS_FOLDER = os.path.join("static", "modelos")
CALIB_FOLDER = os.path.join("static", "calibraciones")
CAPTURAS_FOLDER = os.path.join("static", "capturas")
COLOR_FOLDER = os.path.join("static", "calibraciones_color")

os.makedirs(CAPTURAS_FOLDER, exist_ok=True)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (1280, 720)}
))
picam2.start()

color_thresholds = {
    "r_min": 0, "r_max": 255,
    "g_min": 0, "g_max": 255,
    "b_min": 0, "b_max": 255
}

# ----------------- VARIABLES GLOBALES PARA CONTROL -----------------
step_counter = 0
step_size = 1  # Tamaño del paso del motor


def gen_frames():
    while True:
        frame = picam2.capture_array()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_color_frames():
    while True:
        frame = picam2.capture_array()
        if frame is None:
            break
        
        # Convertimos a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Máscara: True si está dentro de los umbrales
        mask = (frame_rgb[:,:,0] >= color_thresholds["r_min"]) & (frame_rgb[:,:,0] <= color_thresholds["r_max"]) & \
               (frame_rgb[:,:,1] >= color_thresholds["g_min"]) & (frame_rgb[:,:,1] <= color_thresholds["g_max"]) & \
               (frame_rgb[:,:,2] >= color_thresholds["b_min"]) & (frame_rgb[:,:,2] <= color_thresholds["b_max"])
        
        # Crear imagen binaria (negro/fondo, blanco/umbral)
        filtered = np.zeros_like(frame)  # todo negro
        filtered[mask] = [255, 255, 255]  # píxeles dentro del rango → blanco
        
        _, buffer = cv2.imencode('.jpg', filtered)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ----------------- NUEVAS RUTAS DE CONTROL -----------------

@app.route('/action/<cmd>', methods=['POST'])
def control_action(cmd):
    global step_counter
    
    if cmd == "left":
        print(f"🟢 Botón izquierda presionado (paso: {step_size})")
        step_counter -= step_size
        # Aquí llamarías tu función para mover el motor a la izquierda
        # Por ejemplo: move_stepper_left(step_size)
        
    elif cmd == "right":
        print(f"🟢 Botón derecha presionado (paso: {step_size})")
        step_counter += step_size
        # Aquí llamarías tu función para mover el motor a la derecha
        # Por ejemplo: move_stepper_right(step_size)
        
    elif cmd == "capture":
        print("🟢 Botón capture presionado")
        # Reutilizamos la función capture existente
        frame = picam2.capture_array()
        if frame is not None:
            filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = os.path.join(CAPTURAS_FOLDER, filename)
            cv2.imwrite(path, frame)
            return jsonify({
                "success": True,
                "step": step_counter,
                "url": url_for('static', filename=f"capturas/{filename}")
            })
        return jsonify({"success": False, "step": step_counter})
        
    elif cmd == "start_loop":
        print("🟢 Botón start loop presionado")
        # Aquí va tu función de loop
        # Por ejemplo: start_capture_loop()
        
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
    frame = picam2.capture_array()
    if frame is None:
        return jsonify({"success": False})
    
    # Nombre único con fecha/hora
    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(CAPTURAS_FOLDER, filename)
    cv2.imwrite(path, frame)
    
    return jsonify({
        "success": True,
        "url": url_for('static', filename=f"capturas/{filename}")
    })


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/ajustes/camara/cancelar")
def cancelar_calibracion():
    # Vaciar carpeta de capturas
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
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
