from ads1x15 import ADS1115
from machine import Pin
from DRV8825 import DRV8825
from machine import Pin, I2C
import time
from ads1x15 import ADS1115




pins_logic = {
    "Sw_anemometro": 2,
    "Sw_rain": 3,
    "Sw_end_B": 4,
    "Sw_end_A": 5
    }
pins_motor_A = {
    "enable": 26,
    "m0": 22,
    "m1": 21,
    "m2": 20,
    "reset": 19,
    "sleep": 18,
    "step": 17,
    "dir": 16
}
pins_motor_B = {
    "enable": 8,
    "m0": 9,
    "m1": 10,
    "m2": 11,
    "reset": 12,
    "sleep": 13,
    "step": 14,
    "dir": 15
}
motor = DRV8825(pins_motor_B)
DEBOUNCE_MS = 80   # puedes probar entre 50 y 120 ms

last_irq_time1 = 0
last_irq_time2 = 0
def endstop1(pin):
    global last_irq_time1
    now = time.ticks_ms()

    if time.ticks_diff(now, last_irq_time1) < DEBOUNCE_MS:
        return   # Ignorar rebote

    last_irq_time1 = now
    print("[INT1] Endstop 1 activado")
    motor.stop()

def endstop2(pin):
    global last_irq_time2
    now = time.ticks_ms()

    if time.ticks_diff(now, last_irq_time2) < DEBOUNCE_MS:
        return   # Ignorar rebote

    last_irq_time2 = now
    print("[INT2] Endstop 2 activado")
    motor.stop()

# Pines de endstops (entradas con pull-up interno)
end1 = Pin(4, Pin.IN, Pin.PULL_UP)
end2 = Pin(5, Pin.IN, Pin.PULL_UP)

# Asociar interrupciones a flanco de bajada
end1.irq(trigger=Pin.IRQ_FALLING, handler=endstop1)
end2.irq(trigger=Pin.IRQ_FALLING, handler=endstop2)

# ==== USO DEL MOTOR ====
motor.set_microstep(DRV8825.MICROSTEP_1_32)
motor.enable(True)

# Configurar I2C1 en GP6=SCL, GP7=SDA
i2c = I2C(1, scl=Pin(7), sda=Pin(6), freq=400000)
print("Escaneando I2C...")

AS5600_ADDR = 0x36
REG_ANGLE_H = 0x0E


def leer_angulo(): #encoder
    # Leemos 2 bytes desde ANGLE (0x0E y 0x0F)
    data = i2c.readfrom_mem(AS5600_ADDR, REG_ANGLE_H, 2)

    raw = (data[0] << 8) | data[1]     # 12 bits
    raw &= 0x0FFF                      # aseguramos 12 bits

    angulo = (raw * 360.0) / 4096.0
    return angulo, raw

# Ejemplo
while True: #el main.py
    ang, raw = leer_angulo()
    v1 = end1.value()   # 1 = no presionado (por pull-up), 0 = presionado
    v2 = end2.value()
    print("end1:", v1, "  end2:", v2)
    motor.step(steps=500, direction=1, delay=0.0001)
    print("Ángulo: {:.2f}°  (raw={})".format(ang, raw))
    time.sleep_ms(10)

