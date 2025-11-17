from machine import Pin
import time

# Clase DRV8825 en MicroPython (resumida para este ejemplo)
class DRV8825:
    MICROSTEP_1_1  = (0, 0, 0)
    MICROSTEP_1_2  = (1, 0, 0)
    MICROSTEP_1_4  = (0, 1, 0)
    MICROSTEP_1_8  = (1, 1, 0)
    MICROSTEP_1_16 = (0, 0, 1)
    MICROSTEP_1_32 = (1, 0, 1)

    def __init__(self, pins):
        self.pins = {name: Pin(num, Pin.OUT) for name, num in pins.items()}
        self._stop_flag = False

        # Estado inicial
        self.pins["enable"].off()   # LOW = habilitado
        self.pins["reset"].on()
        self.pins["sleep"].on()

    def set_microstep(self, mode):
        m0, m1, m2 = mode
        self.pins["m0"].value(m0)
        self.pins["m1"].value(m1)
        self.pins["m2"].value(m2)

    def enable(self, state=True):
        self.pins["enable"].value(0 if state else 1)

    def step(self, steps, direction=1, delay=0.001):
        self._stop_flag = False
        self.pins["dir"].value(1 if direction else 0)

        for _ in range(steps):
            print(f"[DEBUG] Step {_+1}/{steps}, Dir: {direction}")
            if self._stop_flag:
                print("[INFO] Movimiento detenido por stop()")
                break
            self.pins["step"].on()
            time.sleep(delay)
            self.pins["step"].off()
            time.sleep(delay)

    def stop(self):
        self._stop_flag = True