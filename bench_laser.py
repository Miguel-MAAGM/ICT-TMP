#!/usr/bin/env python
"""Benchmark headless del detector de laser (sin GUI).

Mide el tiempo del pipeline a resolucion nativa, separando etapas:
  1. _laser_score  (HSV -> score base)
  2. blur + cierre morfologico
  3. _detectar_desde_score (deteccion por fila + filtro MAD)

Uso:
    python bench_laser.py                      # usa preview_capture.jpg, 10 reps
    python bench_laser.py imagen.jpg -n 20     # imagen y nro de repeticiones
"""
import argparse
import time
import sys

import cv2
import numpy as np

from test import _laser_score

try:
    from scipy.ndimage import median_filter as _ndi_median
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ----------------------------------------------------------------------
# Detector vectorizado (copia autocontenida de tune_laser, sin matplotlib).
# ----------------------------------------------------------------------
def _detectar_desde_score(score_s,
                          score_min, factor_umbral,
                          ancho_min, ancho_max,
                          ventana_continuidad, mad_factor,
                          modo="cierre", sep_max=40):
    H, W = score_s.shape
    arreglo_x = np.full(H, np.nan, dtype=np.float32)

    max_por_fila = score_s.max(axis=1)
    filas_activas = np.where(max_por_fila >= score_min)[0]
    if filas_activas.size == 0:
        return arreglo_x
    umbrales = np.maximum(score_min, max_por_fila * factor_umbral)

    x_prev = None
    rachas = 0

    for y in filas_activas:
        fila = score_s[y]
        sobre = fila >= umbrales[y]

        diff = np.diff(sobre.view(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0]
        if sobre[0]:
            starts = np.concatenate(([0], starts))
        if sobre[-1]:
            ends = np.concatenate((ends, [W - 1]))
        if starts.size == 0:
            rachas += 1
            if rachas > 30:
                x_prev = None
            continue

        anchos = ends - starts
        ok = (anchos >= ancho_min) & (anchos <= ancho_max)
        if not ok.any():
            rachas += 1
            if rachas > 30:
                x_prev = None
            continue
        starts = starts[ok]; ends = ends[ok]

        peaks_x = np.empty(starts.size, dtype=np.int32)
        peaks_v = np.empty(starts.size, dtype=np.float32)
        for i, (s_i, e_i) in enumerate(zip(starts, ends)):
            seg = fila[s_i:e_i + 1]
            k = int(np.argmax(seg))
            peaks_x[i] = s_i + k
            peaks_v[i] = seg[k]

        ok_borde = (peaks_x > 0) & (peaks_x < W - 1)
        xs_sub = peaks_x.astype(np.float32)
        if ok_borde.any():
            xi = peaks_x[ok_borde]
            L = fila[xi - 1].astype(np.float32)
            C = fila[xi].astype(np.float32)
            R = fila[xi + 1].astype(np.float32)
            denom = L - 2.0 * C + R
            with np.errstate(divide='ignore', invalid='ignore'):
                delta = np.where(denom != 0, 0.5 * (L - R) / denom, 0.0)
            delta = np.clip(delta, -1.0, 1.0)
            xs_sub[ok_borde] = xi.astype(np.float32) + delta

        if modo == "midpoint":
            if x_prev is not None:
                cerca = np.abs(xs_sub - x_prev) <= ventana_continuidad
                if cerca.any():
                    xs_c = xs_sub[cerca]; vs_c = peaks_v[cerca]
                else:
                    xs_c = xs_sub; vs_c = peaks_v
            else:
                xs_c = xs_sub; vs_c = peaks_v

            i1 = int(np.argmax(vs_c))
            x1 = float(xs_c[i1]); v1 = float(vs_c[i1])
            d_a_x1 = np.abs(xs_c - x1)
            mascara_par = (d_a_x1 <= sep_max) & (d_a_x1 >= 1.0) & (vs_c >= 0.4 * v1)
            if mascara_par.any():
                i2 = int(np.argmax(np.where(mascara_par, vs_c, -np.inf)))
                x2 = float(xs_c[i2])
                x_sel = 0.5 * (x1 + x2)
            else:
                x_sel = x1
        else:
            if x_prev is not None:
                d = np.abs(xs_sub - x_prev)
                dentro = d <= ventana_continuidad
                if dentro.any():
                    idx = int(np.argmin(np.where(dentro, d, np.inf)))
                else:
                    idx = int(np.argmin(d))
            else:
                idx = int(np.argmax(peaks_v))
            x_sel = float(xs_sub[idx])

        arreglo_x[y] = x_sel
        x_prev = x_sel
        rachas = 0

    if (~np.isnan(arreglo_x)).sum() > 20:
        ventana = 31
        valid = ~np.isnan(arreglo_x)
        if _HAS_SCIPY:
            ys = np.arange(H)
            arr_fill = np.interp(ys, ys[valid], arreglo_x[valid])
            media_movil = _ndi_median(arr_fill.astype(np.float32),
                                      size=ventana, mode='nearest')
        else:
            media_movil = np.full(H, np.nan, dtype=np.float32)
            half = ventana // 2
            for y in range(H):
                y0 = max(0, y - half); y1 = min(H, y + half + 1)
                tramo = arreglo_x[y0:y1]
                tramo = tramo[~np.isnan(tramo)]
                if tramo.size >= 5:
                    media_movil[y] = np.median(tramo)
        residuo = np.abs(arreglo_x - media_movil)
        mad = np.nanmedian(residuo)
        if mad > 0:
            mala = residuo > mad_factor * max(mad, 1.0)
            arreglo_x = arreglo_x.copy()
            arreglo_x[mala] = np.nan
    return arreglo_x


# Parametros por defecto (los mismos valores init de los sliders de tune_laser)
PARAMS = dict(
    score_min=18.0,
    factor_umbral=0.55,
    ancho_max=60,
    ancho_min=2,
    kernel_suave_x=9,
    ventana_continuidad=40,
    mad_factor=4.0,
    sat_min=40,
    val_min=30,
    val_alto=220,
    cierre_x=0,
    sep_max=40,
    modo="cierre",
)


def _aplicar_blur_cierre(base, kernel, cierre):
    s = base if kernel <= 1 else cv2.GaussianBlur(base, (kernel, 1), 0)
    if cierre >= 2:
        kern = np.ones((1, cierre), dtype=np.uint8)
        s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, kern)
    return s


def _resumen(nombre, tiempos_ms):
    arr = np.array(tiempos_ms, dtype=np.float64)
    print(f"  {nombre:<22} "
          f"min={arr.min():7.1f}  media={arr.mean():7.1f}  "
          f"max={arr.max():7.1f}  std={arr.std():6.1f}  (ms)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark detector laser")
    parser.add_argument("imagen", nargs="?", default="preview_capture.jpg")
    parser.add_argument("-n", "--reps", type=int, default=10,
                        help="numero de repeticiones (default 10)")
    parser.add_argument("-w", "--warmup", type=int, default=2,
                        help="repeticiones de calentamiento descartadas (default 2)")
    parser.add_argument("-m", "--mitad", action="store_true",
                        help="recorta la imagen y se queda con la mitad izquierda")
    parser.add_argument("-d", "--downscale", type=float, default=1.0,
                        help="factor de reduccion en ambos ejes (ej. 2 = mitad)")
    parser.add_argument("-s", "--step", type=int, default=1,
                        help="procesa 1 de cada N filas (reduce costo de deteccion)")
    args = parser.parse_args()

    img = cv2.imread(args.imagen)
    if img is None:
        sys.stderr.write(f"No se pudo leer la imagen: {args.imagen}\n")
        sys.exit(1)
    H_orig, W_orig = img.shape[:2]

    if args.mitad:
        img = img[:, :W_orig // 2]
    if args.downscale > 1.0:
        img = cv2.resize(img, None, fx=1.0 / args.downscale,
                         fy=1.0 / args.downscale, interpolation=cv2.INTER_AREA)
    if args.step > 1:
        img = img[::args.step]
    H, W = img.shape[:2]

    print(f"OpenCV {cv2.__version__}  |  NumPy {np.__version__}")
    extras = []
    if args.mitad:
        extras.append("mitad-izq")
    if args.downscale > 1.0:
        extras.append(f"downscale x{args.downscale:g}")
    if args.step > 1:
        extras.append(f"step={args.step}")
    recorte = f"  ({', '.join(extras)} desde {W_orig}x{H_orig})" if extras else ""
    print(f"Imagen: {args.imagen}  ({W}x{H}){recorte}  |  reps={args.reps} "
          f"(warmup={args.warmup})")
    print("-" * 70)

    kernel = PARAMS["kernel_suave_x"]
    if kernel % 2 == 0:
        kernel += 1
    cierre = max(0, int(PARAMS["cierre_x"]))

    t_score, t_blur, t_detect, t_total = [], [], [], []
    n_validas = 0

    for i in range(args.reps + args.warmup):
        t0 = time.perf_counter()
        base = _laser_score(img, sat_min=PARAMS["sat_min"],
                            val_min=PARAMS["val_min"],
                            val_alto=PARAMS["val_alto"])
        t1 = time.perf_counter()
        score_s = _aplicar_blur_cierre(base, kernel, cierre)
        t2 = time.perf_counter()
        xs = _detectar_desde_score(
            score_s,
            score_min=PARAMS["score_min"],
            factor_umbral=PARAMS["factor_umbral"],
            ancho_min=PARAMS["ancho_min"],
            ancho_max=PARAMS["ancho_max"],
            ventana_continuidad=PARAMS["ventana_continuidad"],
            mad_factor=PARAMS["mad_factor"],
            modo=PARAMS["modo"],
            sep_max=PARAMS["sep_max"],
        )
        t3 = time.perf_counter()

        if i >= args.warmup:  # descartar warmup
            t_score.append((t1 - t0) * 1000)
            t_blur.append((t2 - t1) * 1000)
            t_detect.append((t3 - t2) * 1000)
            t_total.append((t3 - t0) * 1000)
            n_validas = int((~np.isnan(xs)).sum())

    _resumen("1. laser_score", t_score)
    _resumen("2. blur+cierre", t_blur)
    _resumen("3. deteccion", t_detect)
    print("-" * 70)
    _resumen("TOTAL pipeline", t_total)

    media_total = np.mean(t_total)
    fps = 1000.0 / media_total if media_total > 0 else 0.0
    print("-" * 70)
    print(f"Filas validas: {n_validas}/{H} ({100*n_validas/H:.1f}%)")
    print(f"Throughput aprox: {fps:.2f} imagenes/s  "
          f"({media_total:.1f} ms/imagen)")


if __name__ == "__main__":
    main()
