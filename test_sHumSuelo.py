import time
import minimalmodbus
import serial

PUERTO = "/dev/ttyUSB0"
BAUDRATE = 9600 # Asegúrate de haber cambiado el sensor a 9600 usando el software de fábrica
TIMEOUT = 0.8
MAX_REINTENTOS = 2

# ID del nuevo sensor de suelo (por defecto es 1)
ID_SUELO = 1

def configurar_instrumento(direccion):
    instrumento = minimalmodbus.Instrument(PUERTO, direccion, mode=minimalmodbus.MODE_RTU)
    instrumento.serial.baudrate = BAUDRATE
    instrumento.serial.bytesize = 8
    instrumento.serial.parity = serial.PARITY_NONE
    instrumento.serial.stopbits = 1
    instrumento.serial.timeout = TIMEOUT
    instrumento.clear_buffers_before_each_transaction = True
    instrumento.close_port_after_each_call = False
    return instrumento

def leer_con_reintentos(instrumento, registro_inicial, cantidad):
    ultimo_error = "sin respuesta"
    for _ in range(MAX_REINTENTOS):
        try:
            datos = instrumento.read_registers(
                registeraddress=registro_inicial,
                number_of_registers=cantidad,
                functioncode=3,
            )
            return datos, None
        except (minimalmodbus.NoResponseError, minimalmodbus.InvalidResponseError, serial.SerialException) as error:
            ultimo_error = str(error)
            time.sleep(0.05)
    return None, ultimo_error

try:
    sensor_suelo = configurar_instrumento(ID_SUELO)
    print(f"Iniciando lectura de sensor de suelo en ID {ID_SUELO} a {BAUDRATE} baudios...\n")

    while True:
        # El manual indica que el registro 0x00 es Humedad y el 0x01 es Temperatura.
        # Leemos 2 registros seguidos a partir del 0.
        datos_suelo, error_suelo = leer_con_reintentos(sensor_suelo, registro_inicial=0, cantidad=2)
        
        if datos_suelo is not None:
            # Humedad: Registro 0 (viene multiplicado por 10)
            humedad_cruda = datos_suelo[0]
            humedad = humedad_cruda / 10.0
            
            # Temperatura: Registro 1 (viene multiplicado por 10 y puede ser negativo en complemento a 2)
            temp_cruda = datos_suelo[1]
            if temp_cruda > 32767: # Si es mayor a 32767 (0x7FFF), es un número negativo
                temp_cruda -= 65536
            temperatura = temp_cruda / 10.0
            
            print("--- CWT-SOIL-TH-S (Suelo) ---")
            print(f"Humedad: {humedad:.1f} %RH | Temperatura: {temperatura:.1f} °C")
            print("-" * 30)
        else:
            print(f"--- CWT-SOIL-TH-S (Suelo) ---")
            print(f"Error de lectura: {error_suelo}")
            print("-" * 30)

        time.sleep(2)

except KeyboardInterrupt:
    print("\nPrograma detenido por el usuario.")
except Exception as error:
    print(f"Error fatal: {error}")