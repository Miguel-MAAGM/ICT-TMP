import minimalmodbus
import serial
import time

# ===================== CONFIG =====================
PORT = "COM3"   # <-- cambia al tuyo (ej: COM4)
SLAVE_ID = 1

BAUDRATE = 9600
TIMEOUT = 1.0
PERIODO = 0.5   # segundos entre lecturas
# ==================================================

sensor = minimalmodbus.Instrument(PORT, SLAVE_ID, mode=minimalmodbus.MODE_RTU)
sensor.serial.baudrate = BAUDRATE
sensor.serial.bytesize = 8
sensor.serial.parity   = serial.PARITY_NONE
sensor.serial.stopbits = 1
sensor.serial.timeout  = TIMEOUT

sensor.clear_buffers_before_each_transaction = True

print("Leyendo sensor WSS-05 (Ctrl+C para detener)\n")

while True:
    try:
        regs = sensor.read_registers(0, 10, functioncode=3)

        temp_c = regs[0] / 10.0
        hum_pct = regs[1] / 10.0
        lux = regs[4]
        press_hpa = regs[8]

        print(f"Temp: {temp_c:5.1f} °C | Hum: {hum_pct:5.1f} % | Luz: {lux:6d} lux | Pres: {press_hpa:4d} hPa")

        time.sleep(PERIODO)

    except KeyboardInterrupt:
        print("\nDetenido por usuario.")
        break

    except Exception as e:
        print("Error leyendo sensor:", e)
        time.sleep(1)
