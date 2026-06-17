"""Acceso a InfluxDB para el dashboard de sensores.

Incluye el InfluxWriter (escritura, igual al que ya usás) y un InfluxReader
(consultas para los gráficos y el reporte). La librería influxdb-client se
importa de forma perezosa: si no está instalada, se lanza un RuntimeError claro
solo al usarse, sin romper el arranque del servidor.

Config por variables de entorno (con los mismos defaults que ya tenías).
OJO: el token por defecto es el de tu instalación; idealmente ponelo en la
variable de entorno INFLUXDB_TOKEN y no lo dejes en el código.
"""
import os
import re
from importlib import import_module

DEFAULT_BUCKET = os.getenv("INFLUXDB_BUCKET", "ltdic")
DEFAULT_ORG = os.getenv("INFLUXDB_ORG", "ltdic")
DEFAULT_TOKEN = os.getenv(
    "INFLUXDB_TOKEN",
    "PCH67KisAftQrJgJ5R5lnhsXO7X9LzdxIodRMyh3XmZ2plUA6vk1oONdxjSTTq4e-lniOEC8rhnlpWxfNivgHw==",
)
DEFAULT_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
DEFAULT_MEASUREMENT = os.getenv("INFLUXDB_MEASUREMENT", "sensor_data")


def _sane_every(every, fallback="30s"):
    """Valida una ventana Flux tipo '30s'/'1m'/'2h' para evitar inyección."""
    return every if isinstance(every, str) and re.fullmatch(r"\d+[smhd]", every) else fallback


class InfluxWriter:
    def __init__(self, url=DEFAULT_URL, token=DEFAULT_TOKEN, org=DEFAULT_ORG, bucket=DEFAULT_BUCKET):
        try:
            influxdb_client = import_module("influxdb_client")
            write_api_module = import_module("influxdb_client.client.write_api")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Falta la dependencia 'influxdb-client'. Instalala con: pip install influxdb-client"
            ) from error

        self.bucket = bucket
        self.org = org
        self.client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=write_api_module.SYNCHRONOUS)
        self.point_class = influxdb_client.Point

    def write_sensor_data(self, sensor_name, fields, tags=None, measurement=DEFAULT_MEASUREMENT):
        point = self.point_class(measurement).tag("sensor", sensor_name)
        if tags:
            for key, value in tags.items():
                if value is not None:
                    point.tag(key, str(value))
        for key, value in fields.items():
            if value is None:
                continue
            point.field(key, value)
        self.write_api.write(bucket=self.bucket, org=self.org, record=point)

    def write_evento(self, measurement, fields, tags=None, ts=None):
        """Escribe un punto genérico (p. ej. el registro de un barrido ejecutado).

        fields: dict de campos (números o strings). tags: dict opcional.
        ts: datetime del evento (si None, usa el momento de escritura).
        """
        point = self.point_class(measurement)
        if tags:
            for key, value in tags.items():
                if value is not None:
                    point.tag(key, str(value))
        for key, value in fields.items():
            if value is not None:
                point.field(key, value)
        if ts is not None:
            point.time(ts)
        self.write_api.write(bucket=self.bucket, org=self.org, record=point)

    def close(self):
        self.client.close()


class InfluxReader:
    """Consultas de lectura para el dashboard y el reporte."""

    def __init__(self, url=DEFAULT_URL, token=DEFAULT_TOKEN, org=DEFAULT_ORG,
                 bucket=DEFAULT_BUCKET, measurement=DEFAULT_MEASUREMENT):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.measurement = measurement
        self._client = None
        self._query_api_obj = None

    def _query_api(self):
        if self._query_api_obj is None:
            try:
                influxdb_client = import_module("influxdb_client")
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "Falta la dependencia 'influxdb-client'. Instalala con: pip install influxdb-client"
                ) from error
            self._client = influxdb_client.InfluxDBClient(
                url=self.url, token=self.token, org=self.org)
            self._query_api_obj = self._client.query_api()
        return self._query_api_obj

    def datos_recientes(self, minutos=60, cada="30s"):
        """Series por sensor y campo de los últimos `minutos`.

        Devuelve { sensor: { campo: {"t": [iso...], "v": [float...]} } }.
        """
        minutos = max(1, int(minutos))
        cada = _sane_every(cada, "30s")
        q = f'''
import "types"
from(bucket: "{self.bucket}")
  |> range(start: -{minutos}m)
  |> filter(fn: (r) => r._measurement == "{self.measurement}")
  |> filter(fn: (r) => types.isType(v: r._value, type: "float") or types.isType(v: r._value, type: "int"))
  |> aggregateWindow(every: {cada}, fn: mean, createEmpty: false)
  |> sort(columns: ["_time"])
'''
        tablas = self._query_api().query(q, org=self.org)
        result = {}
        for tabla in tablas:
            for rec in tabla.records:
                sensor = rec.values.get("sensor", "sensor")
                campo = rec.get_field()
                serie = result.setdefault(sensor, {}).setdefault(campo, {"t": [], "v": []})
                serie["t"].append(rec.get_time().isoformat())
                serie["v"].append(rec.get_value())
        return result

    def ultimos_valores(self, ventana_min=120):
        """Último valor numérico de cada campo (de cualquier sensor) en la ventana.

        Devuelve { campo: {"value": float, "sensor": str, "t": iso} }.
        """
        ventana_min = max(1, int(ventana_min))
        # last() funciona también con strings (p. ej. direccion_viento), así que
        # acá NO filtramos por tipo (a diferencia de los gráficos, que promedian).
        q = f'''
from(bucket: "{self.bucket}")
  |> range(start: -{ventana_min}m)
  |> filter(fn: (r) => r._measurement == "{self.measurement}")
  |> last()
'''
        tablas = self._query_api().query(q, org=self.org)
        res = {}
        for tabla in tablas:
            for rec in tabla.records:
                campo = rec.get_field()
                t = rec.get_time()
                prev = res.get(campo)
                if prev is None or (t and prev["t"] and t > prev["t"]):
                    res[campo] = {"value": rec.get_value(),
                                  "sensor": rec.values.get("sensor"), "t": t}
        return {k: {"value": v["value"], "sensor": v["sensor"],
                    "t": v["t"].isoformat() if v["t"] else None}
                for k, v in res.items()}

    def reporte_rango(self, inicio_epoch, fin_epoch, cada="1m"):
        """Tabla ancha (pivot) entre dos instantes (epoch s). Para el CSV.

        Devuelve (headers, filas). headers = ['time', '<sensor>_<campo>', ...].
        """
        cada = _sane_every(cada, "1m")
        ini = int(float(inicio_epoch))
        fin = int(float(fin_epoch))
        q = f'''
import "types"
from(bucket: "{self.bucket}")
  |> range(start: {ini}, stop: {fin})
  |> filter(fn: (r) => r._measurement == "{self.measurement}")
  |> filter(fn: (r) => types.isType(v: r._value, type: "float") or types.isType(v: r._value, type: "int"))
  |> aggregateWindow(every: {cada}, fn: mean, createEmpty: false)
  |> pivot(rowKey:["_time"], columnKey: ["sensor", "_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
'''
        tablas = self._query_api().query(q, org=self.org)
        meta = {"result", "table", "_start", "_stop", "_measurement", "_time"}
        columnas = set()
        registros = []
        for tabla in tablas:
            for rec in tabla.records:
                vals = rec.values
                columnas.update(k for k in vals.keys() if k not in meta)
                registros.append(vals)
        columnas = sorted(columnas)
        headers = ["time"] + columnas
        filas = []
        for vals in registros:
            t = vals.get("_time")
            t = t.isoformat() if hasattr(t, "isoformat") else str(t)
            filas.append([t] + [vals.get(c, "") for c in columnas])
        return headers, filas

    def close(self):
        if self._client is not None:
            self._client.close()
