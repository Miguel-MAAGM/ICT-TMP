from machine import Pin, I2C
from DRV8825 import DRV8825 
import time
import sys
import select
#ESTE CODIGO FUNCIONA BIEN, PERO EL MOVIMIENTO DEL MOTOR NO ES CORRECTO

# ======================== MOTOR DRV8825 ==========================
pins_motor_B = {
    "enable": 8, "m0": 9, "m1": 10, "m2": 11,
    "reset": 12, "sleep": 13, "step": 14, "dir": 15
}
motor = DRV8825(pins_motor_B)
motor.set_microstep(DRV8825.MICROSTEP_1_32)
motor.enable(True)
PASOS_POR_GRADO = 6400 / 360.0    # ≈ 17.7777

# ======================== ENCODER AS5600 ==========================
AS5600_ADDR = 0x36
AS5600_ANGLE_REG = 0x0C 
i2c = I2C(1, scl=Pin(7), sda=Pin(6), freq=400000)

def leer_angulo():
    """Lee el ángulo del encoder AS5600 en grados y el valor RAW (0-4095)."""
    try:
        data = i2c.readfrom_mem(AS5600_ADDR, AS5600_ANGLE_REG, 2)
        raw = ((data[0] << 8) | data[1]) & 0x0FFF
        ang = (raw * 360.0) / 4096.0
        return ang, raw
    except OSError:
        # En caso de error de I2C, devuelve valores que indiquen problema
        return -1.0, 0


# ======================== ENDSTOPS ==========================
DEBOUNCE_MS = 80
last_irq_time1 = 0
last_irq_time2 = 0
stop_loop = False  

def endstop1(pin):
    global last_irq_time1, stop_loop
    now = time.ticks_ms()
    if time.ticks_diff(now, last_irq_time1) < DEBOUNCE_MS:
        return
    last_irq_time1 = now
    print("[ENDSTOP IZQUIERDA ACTIVADO]")
    stop_loop = True

def endstop2(pin):
    global last_irq_time2, stop_loop
    now = time.ticks_ms()
    if time.ticks_diff(now, last_irq_time2) < DEBOUNCE_MS:
        return
    last_irq_time2 = now
    print("[ENDSTOP DERECHA ACTIVADO]")
    stop_loop = True

Pin(4, Pin.IN, Pin.PULL_UP).irq(trigger=Pin.IRQ_FALLING, handler=endstop1)
Pin(5, Pin.IN, Pin.PULL_UP).irq(trigger=Pin.IRQ_FALLING, handler=endstop2)


# ======================== VARIABLES GLOBALES ==========================
angle_to_move = 20.0    # Grados a mover en comando LEFT/RIGHT (Valor inicial)
running_loop = False    
pasos_totales = 0       

# Parámetros de Control Proporcional 
KP = 0.0000005     
MIN_DELAY = 0.0003 
MAX_DELAY = 0.005  
RAW_TOLERANCE = 10 


def mover_grados(dir):
    """
    Mueve angle_to_move grados EXACTOS usando el AS5600 
    e implementa un Control Proporcional de velocidad para movimiento fino.
    """
    global stop_loop, pasos_totales

    direccion_txt = "DERECHA" if dir == 1 else "IZQUIERDA"
    print(f"\n[INFO] Solicitado mover {angle_to_move}° hacia {direccion_txt}")

    # 1) Obtener RAW inicial y calcular objetivo en cuentas RAW (0-4095)
    _, raw_ini = leer_angulo()
    raw_to_move = int((angle_to_move * 4096.0) / 360.0)
    
    if dir == 1:
        target_raw = (raw_ini + raw_to_move) % 4096
    else:
        target_raw = (raw_ini - raw_to_move + 4096) % 4096

    stop_loop = False
    motor._stop_flag = False

    ultima_muestra = time.ticks_ms()

    while True:

        ang, raw = leer_angulo()

        # 2) Error angular en cuentas RAW (12 bits de resolución)
        error_raw = ((target_raw - raw + 6144) % 4096) - 2048 

        # 3) ¿Llegamos?
        if abs(error_raw) < RAW_TOLERANCE:
            # Mandar respuesta al final
            print(f"ANGULO: {ang:.2f}") # <--- RESPUESTA SERIAL AL HOST
            break

        # 4) Dirección según error
        direction = 1 if error_raw > 0 else 0
        dir_txt = "DERECHA" if direction == 1 else "IZQUIERDA"

        # 5) Control Proporcional: Calcular delay (inverso de velocidad)
        error_abs = abs(error_raw)
        delay_motor = MAX_DELAY - error_abs * KP
        delay_motor = max(MIN_DELAY, min(MAX_DELAY, delay_motor))

        # 6) Mover un micro-paso
        motor.step(1, direction=direction, delay=delay_motor)
        pasos_totales += 1

        # 7) Mostrar telemetría (solo en consola local, no es la respuesta que espera el host)
        now = time.ticks_ms()
        if time.ticks_diff(now, ultima_muestra) > 150:
            # print(f"[MOVE] DIR={dir_txt:<8} ANG={ang:6.2f}° ERR_R={error_raw:5d} RAW={raw:4d} PASOS={pasos_totales}")
            ultima_muestra = now

        # 8) Revisar comandos/finales de carrera (para detener por comando 'STOP' o endstop)
        if stop_loop:
            print("[INFO] Movimiento detenido por ENDSTOP o STOP")
            break # Salir del while True (el loop se detiene en el punto actual)

        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
             cmd_local = sys.stdin.readline().strip().upper()
             if cmd_local == "STOP":
                 print("[INFO] STOP recibido → Cancelando movimiento manual")
                 stop_loop = True # Forzar salida del while
                 break
        
        time.sleep(0.0001)

    # Si se detuvo por endstop/stop, enviamos la posición actual
    if stop_loop:
        ang, _ = leer_angulo()
        print(f"ANGULO: {ang:.2f}") # <--- RESPUESTA SERIAL AL HOST


# ===================== LOOP AUTOMÁTICO =======================
def loop_auto():
    global running_loop, stop_loop, pasos_totales
    running_loop = True
    stop_loop = False

    direction = 1  
    dir_txt = "DERECHA"

    print("\n===================== LOOP AUTOMÁTICO INICIADO =====================")
    
    motor._stop_flag = False
    ultima_muestra = time.ticks_ms()
    
    LOOP_DELAY = 0.0015 

    while running_loop:

        # -------- 1) REVISAR COMANDOS SERIAL (Principalmente "STOP") --------
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            entrada = sys.stdin.readline().strip().upper()
            if entrada == "STOP":
                print("\n[INFO] STOP recibido → deteniendo LOOP")
                running_loop = False
                break # Sale del while running_loop

        # -------- 2) SI ACTIVÓ ENDSTOP → CAMBIAR DIRECCIÓN --------
        if stop_loop:
            stop_loop = False
            direction = 0 if direction == 1 else 1
            dir_txt = "DERECHA" if direction == 1 else "IZQUIERDA"
            print(f"[INFO] Cambio de dirección → {dir_txt}")
            motor._stop_flag = False

        # -------- 3) MOVER UN PASO con delay fijo para finura --------
        motor.step(1, direction=direction, delay=LOOP_DELAY) 
        pasos_totales += 1

        # -------- 4) MOSTRAR INFORMACIÓN CADA 150 ms --------
        now = time.ticks_ms()
        if time.ticks_diff(now, ultima_muestra) > 150:
            ang, raw = leer_angulo()
            # print(f"[INFO] DIR={dir_txt}  Ángulo={ang:6.2f}° RAW={raw:4d}  PasosTot={pasos_totales}")
            ultima_muestra = now

        time.sleep(0.0001)

    print("\n===================== LOOP AUTOMÁTICO DETENIDO =====================")
    ang, raw = leer_angulo()
    # Enviamos la respuesta final
    print(f"ANGULO: {ang:.2f}") # <--- RESPUESTA SERIAL AL HOST


# ======================= PROCESAR COMANDOS =======================
def procesar_comando(cmd):
    global angle_to_move, running_loop, stop_loop
    
    cmd = cmd.strip().upper()

    if cmd.startswith("SET_ANGLE"):
        try:
            _, valor = cmd.split()
            angle_to_move = float(valor)
            print(f"[INFO] SET_ANGLE = {angle_to_move}°")
            # Respuesta simple para el host
            print("OK") 
        except:
            print("[ERROR] Formato correcto: SET_ANGLE 20")
            print("ERROR_SET_ANGLE")

    elif cmd == "RIGHT":
        mover_grados(1) # La respuesta se imprime dentro de mover_grados

    elif cmd == "LEFT":
        mover_grados(0) # La respuesta se imprime dentro de mover_grados

    elif cmd == "LOOP":
        if not running_loop:
            # Respuesta inmediata para el host
            print("LOOP_STARTING")
            loop_auto() # loop_auto imprime la respuesta final al terminar

    elif cmd == "STOP":
        print("[INFO] STOP solicitado")
        running_loop = False
        motor.stop()
        stop_loop = True
        # Respuesta inmediata
        ang, _ = leer_angulo()
        print(f"ANGULO: {ang:.2f}")

    elif cmd == "READ":
        ang, raw = leer_angulo()
        # La respuesta clave que el host espera
        print(f"ANGULO: {ang:.2f}") 
        # print(f"[INFO] Ángulo: {ang:.2f}° | RAW: {raw}") # Opcional, solo para debug local
        
    else:
        if cmd.strip() != "":
            print("[ERROR] Comando inválido:", cmd)
            print("ERROR_INVALID_CMD")


# ======================= LOOP PRINCIPAL =======================
# Deshabilitamos la impresión de bienvenida para evitar que el host la interprete como un comando.
# print("Sistema listo. Esperando comandos por serial...") 

while True:
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        linea = sys.stdin.readline()
        procesar_comando(linea)

    time.sleep(0.01)
