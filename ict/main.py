from machine import Pin
from DRV8825 import DRV8825
from machine import Pin, I2C
import time
from ads1x15 import ADS1115



# Configurar I2C1 en GP6=SCL, GP7=SDA
i2c = I2C(1, scl=Pin(7), sda=Pin(6), freq=400000)



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




# ==== INTERRUPCIONES ====
def endstop1(pin):
    print(f"[INT1] Pin {pin.id()} activado -> Stop motor")
    motor.stop()

def endstop2(pin):
    print(f"[INT2] Pin {pin.id()} activado -> Stop motor")
    motor.stop()

# Pines de endstops (entradas con pull-up interno)
end1 = Pin(18, Pin.IN, Pin.PULL_UP)
end2 = Pin(19, Pin.IN, Pin.PULL_UP)

# Asociar interrupciones a flanco de bajada
end1.irq(trigger=Pin.IRQ_FALLING, handler=endstop1)
end2.irq(trigger=Pin.IRQ_FALLING, handler=endstop2)

# ==== USO DEL MOTOR ====
motor.set_microstep(DRV8825.MICROSTEP_1_1)
motor.enable(True)

devices = i2c.scan()
ADS1115_address = 72  # Dirección I2C por defecto del ADS1115
adc = ADS1115(i2c, address=ADS1115_address, gain=0)



switches = {name: Pin(num, Pin.IN, Pin.PULL_UP) for name, num in pins_logic.items()}

while True:
    # Leer todos los pines
    estados = {name: sw.value() for name, sw in switches.items()}
    an3 = adc.read(rate=4,channel1=3)  # Canal 3
    print(f"AN3: {an3}")

    # Imprimir estados
    print("Lecturas:", estados)
    
       
    time.sleep(0.1)