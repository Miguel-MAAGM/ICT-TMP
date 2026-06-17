# ==================================================
# SISTEMA DE CAPTURA FOTOGRÁFICA ROTACIONAL
# DRV8825 + AS5600 + ENDSTOPS
# ==================================================

from machine import Pin, I2C
from DRV8825 import DRV8825
import time
import sys
import select

# ======================== MOTOR DRV8825 ==========================
pins_motor = {
    "enable": 8, "m0": 9, "m1": 10, "m2": 11,
    "reset": 12, "sleep": 13, "step": 14, "dir": 15
}

motor = DRV8825(pins_motor)
motor.set_microstep(DRV8825.MICROSTEP_1_32)
motor.enable(True)

AUTO_ADVANCE = True        # True = no espera SIGUIENTE
TIEMPO_ENTRE_CAPTURAS = 0.1  # segundos


STEP_DELAY = 0.001  # delay FIJO del motor (estable)

# ======================== ENCODER AS5600 ==========================
AS5600_ADDR = 0x36
AS5600_ANGLE_REG = 0x0C

i2c = I2C(1, scl=Pin(7), sda=Pin(6), freq=400000)

def leer_angulo():
    try:
        data = i2c.readfrom_mem(AS5600_ADDR, AS5600_ANGLE_REG, 2)
        raw = ((data[0] << 8) | data[1]) & 0x0FFF
        ang = raw * 360.0 / 4096.0
        return ang, raw
    except:
        return -1.0, 0

# ======================== ENDSTOPS ==========================
endstop_1 = Pin(4, Pin.IN, Pin.PULL_UP)
endstop_2 = Pin(5, Pin.IN, Pin.PULL_UP)

DEBOUNCE_MS = 80
last_irq_1 = 0
last_irq_2 = 0
endstop_hit = 0  # 0 = ninguno, 1 = endstop1, 2 = endstop2


def irq_endstop_1(pin):
    global last_irq_1, endstop_hit
    t = time.ticks_ms()
    if time.ticks_diff(t, last_irq_1) > DEBOUNCE_MS:
        last_irq_1 = t
        endstop_hit = 1

def irq_endstop_2(pin):
    global last_irq_2, endstop_hit
    t = time.ticks_ms()
    if time.ticks_diff(t, last_irq_2) > DEBOUNCE_MS:
        last_irq_2 = t
        endstop_hit = 2

endstop_1.irq(trigger=Pin.IRQ_FALLING, handler=irq_endstop_1)
endstop_2.irq(trigger=Pin.IRQ_FALLING, handler=irq_endstop_2)

# ======================== MODOS DE CAPTURA ==========================
CAPTURE_MODES = {
    "SLOW": 1,
    "MID": 50,
    "FAST": 100
}
PASOS_POR_FOTO = CAPTURE_MODES["MID"]

# ======================== UTILIDADES SERIAL ==========================
def limpiar_serial():
    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()

def check_serial():
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip().upper()
    return None

# ======================== AUTO HOME ==========================
def auto_home():
    print("[AUTO_HOME] Iniciando...")
    motor.enable(True)
    time.sleep(0.1)

    SPEED = 0.00001

    en1 = endstop_1.value() == 0
    en2 = endstop_2.value() == 0

    print(f"[AUTO_HOME] Estado inicial → END1={en1}, END2={en2}")

    # Liberar si parte presionado
    if en1:
        while endstop_1.value() == 0:
            motor.step(10, direction=0, delay=SPEED)

    if en2:
        while endstop_2.value() == 0:
            motor.step(10, direction=1, delay=SPEED)

    # Alejarse un poco
    for _ in range(300):
        motor.step(1, direction=0, delay=SPEED)

    # Buscar HOME (ENDSTOP 2)
    print("[AUTO_HOME] Buscando HOME...")
    while endstop_2.value() == 1:
        motor.step(1, direction=0, delay=SPEED)

    # Margen final
    for _ in range(200):
        motor.step(1, direction=1, delay=SPEED)

    print("[AUTO_HOME] HOMING COMPLETADO ✔")

# ======================== LOOP DE CAPTURA ==========================

def loop_captura():
    global endstop_hit

    print("[CAPTURE] Iniciando barrido (PING-PONG) MODO MANUAL")
    print(f"[CAPTURE] Pasos por foto: {PASOS_POR_FOTO}")

    endstop_hit = 0
    direction = 1  # como lo tienes tú

    RETROCESO_DESPEGUE = 200  # ajustable

    while True:
        # ===================== MOVER PASOS =====================
        for _ in range(PASOS_POR_FOTO):
            cmd = check_serial()
            if cmd == "STOP":
                print("[CAPTURE] STOP recibido")
                return

            if endstop_hit != 0:
                break

            motor.step(1, direction=direction, delay=STEP_DELAY)

        # ===================== SI TOCÓ ENDSTOP =====================
        if endstop_hit != 0:
            ang, _ = leer_angulo()
            print(f"FOTO_ENDSTOP {endstop_hit} {ang:.2f}")

            print(f"[CAPTURE] Endstop {endstop_hit} alcanzado → invirtiendo")

            # despegar del endstop
            inv_dir = 1 - direction
            for _ in range(RETROCESO_DESPEGUE):
                cmd = check_serial()
                if cmd == "STOP":
                    print("[CAPTURE] STOP recibido")
                    return
                motor.step(1, direction=inv_dir, delay=STEP_DELAY)

            # cambiar dirección
            endstop_hit = 0
            direction = inv_dir

            # ---- esperar SIGUIENTE (manual) ----
            while True:
                cmd = check_serial()
                if cmd == "SIGUIENTE":
                    break
                if cmd == "STOP":
                    print("[CAPTURE] STOP recibido")
                    return
                time.sleep(0.01)

            continue

        # ===================== FOTO NORMAL =====================
        ang, _ = leer_angulo()
        print(f"FOTO {ang:.2f}")

        # ===================== ESPERAR SIGUIENTE (manual) =====================
        while True:
            cmd = check_serial()
            if cmd == "SIGUIENTE":
                break
            if cmd == "STOP":
                print("[CAPTURE] STOP recibido")
                return
            time.sleep(0.01)


def solo_loop():
    global endstop_hit

    print("[SOLO_LOOP] Iniciando Bucle Continuo (PING-PONG AUTOMÁTICO)")
    print(f"[SOLO_LOOP] Pasos por bloque: {PASOS_POR_FOTO}")

    endstop_hit = 0
    direction = 1  # 1 o 0, según tu lógica inicial
    RETROCESO_DESPEGUE = 200  # Ajustable
    PASOS_POR_AVANCE=1000
    DELAY_AVANCE=0.0001

    while True:
        # ===================== MOVER PASOS =====================
        for _ in range(PASOS_POR_AVANCE):
            # Revisamos STOP en cada paso para que la detención sea inmediata
            cmd = check_serial()
            if cmd == "STOP":
                print("[SOLO_LOOP] STOP recibido")
                return

            if endstop_hit != 0:
                break

            motor.step(1, direction=direction, delay=DELAY_AVANCE)

        # ===================== SI TOCÓ ENDSTOP =====================
        if endstop_hit != 0:
            ang, _ = leer_angulo()
            print(f"ENDSTOP_HIT {endstop_hit} {ang:.2f}")
            print(f"[SOLO_LOOP] Endstop {endstop_hit} alcanzado → invirtiendo")

            # Despegar del endstop
            inv_dir = 1 - direction
            for _ in range(RETROCESO_DESPEGUE):
                cmd = check_serial()
                if cmd == "STOP":
                    print("[SOLO_LOOP] STOP recibido")
                    return
                motor.step(1, direction=inv_dir, delay=DELAY_AVANCE)

            # Cambiar dirección
            endstop_hit = 0
            direction = inv_dir

            # AQUÍ YA NO ESPERAMOS "SIGUIENTE", solo continuamos
            continue

        # ===================== CICLO NORMAL (SIN ENDSTOP) =====================
        ang, _ = leer_angulo()
        # Imprimimos para que sepas por donde va, pero no se detiene
        print(f"PASO_AUTO {ang:.2f}")

        # Pequeña pausa opcional por seguridad (puedes quitarla si quieres velocidad máxima)
        # time.sleep(0.05)

        # El loop (while True) se repite automáticamente aquí sin esperar nada
# ======================== COMANDOS ==========================
def _endstop_bloquea(direction):
    """Devuelve el endstop que BLOQUEA esa dirección si ya está presionado, o 0.

    Mapeo (de auto_home):
      direction = 1 -> avanza hacia endstop_1
      direction = 0 -> avanza hacia endstop_2 (HOME)
    Con PULL_UP, value()==0 significa presionado.
    """
    if direction == 1 and endstop_1.value() == 0:
        return 1
    if direction == 0 and endstop_2.value() == 0:
        return 2
    return 0

def mover_pasos(pasos, direction, delay=0.0005):
    """Mueve un nº exacto de pasos en una dirección (0 o 1).

    Seguridad: NO avanza hacia un endstop que ya está presionado (evita empujar
    contra el tope; la IRQ es por flanco y no detecta un switch ya pisado). Sí
    permite moverse en la dirección opuesta para despegarse. Se interrumpe si
    llega STOP o si se toca un endstop (por flanco o por nivel). Devuelve los
    pasos realmente realizados.
    """
    global endstop_hit
    endstop_hit = 0
    realizados = 0

    # Bloqueo de arranque: el tope de esta dirección ya está presionado
    bloq = _endstop_bloquea(direction)
    if bloq != 0:
        print(f"[MOVE] BLOQUEADO: endstop {bloq} ya presionado en esa direccion")
        return 0

    for _ in range(int(pasos)):
        cmd = check_serial()
        if cmd == "STOP":
            print("[MOVE] STOP recibido")
            break
        # Cortar por flanco (IRQ) o por nivel real del endstop de esta dirección
        if endstop_hit != 0 or _endstop_bloquea(direction) != 0:
            print("[MOVE] Endstop alcanzado")
            break
        motor.step(1, direction=direction, delay=delay)
        realizados += 1
    return realizados

def procesar_comando(cmd):
    global PASOS_POR_FOTO

    if not cmd:
        return

    if cmd == "AUTO_HOME":
        limpiar_serial()
        auto_home()

    elif cmd.startswith("MOVE"):
        # MOVE <pasos> <dir>   (dir: 0 o 1). Mueve un nº exacto de pasos.
        partes = cmd.split()
        try:
            pasos = int(partes[1])
            direction = int(partes[2]) if len(partes) > 2 else 1
        except:
            print("[ERROR] Uso: MOVE <pasos> <dir>")
            return
        realizados = mover_pasos(pasos, direction)
        print(f"MOVE_DONE {realizados} {direction}")

    elif cmd == "CAPTURE_SLOW":
        PASOS_POR_FOTO = CAPTURE_MODES["SLOW"]
        print("[INFO] Modo LENTO (1 paso/foto)")

    elif cmd == "CAPTURE_MID":
        PASOS_POR_FOTO = CAPTURE_MODES["MID"]
        print("[INFO] Modo MEDIO (50 pasos/foto)")

    elif cmd == "CAPTURE_FAST":
        PASOS_POR_FOTO = CAPTURE_MODES["FAST"]
        print("[INFO] Modo RÁPIDO (100 pasos/foto)")

    elif cmd == "LOOP_CAPTURE":
        limpiar_serial()
        loop_captura()
        print("LOOP_CAPTURE_FINISHED")

    elif cmd == "READ":
        ang, _ = leer_angulo()
        print(f"ANGULO {ang:.2f}")

    elif cmd == "SOLO_LOOP":
        limpiar_serial()
        solo_loop()
        print("SOLO_LOOP_FINISHED")

    elif cmd == "STOP":
        print("[INFO] STOP")

    else:
        print("[ERROR] Comando desconocido")

# ======================== MAIN ==========================
print("=" * 50)
print("SISTEMA DE CAPTURA ROTACIONAL")
print("FW endstop-safety v2")
print("Comandos:")
print(" AUTO_HOME")
print(" CAPTURE_SLOW | CAPTURE_MID | CAPTURE_FAST")
print(" LOOP_CAPTURE")
print(" MOVE <pasos> <dir>")
print(" READ")
print(" STOP")
print("=" * 50)

while True:
    cmd = check_serial()
    if cmd:
        procesar_comando(cmd)
    time.sleep(0.005)
