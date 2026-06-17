"""Núcleo de detección de línea láser — sin GUI, optimizado para RPI4.

Funciones públicas:
    laser_score(img_bgr, **hsv_params)  -> np.ndarray float32
    detect_laser(img_bgr, **params)     -> dict  {xs, stats}
    build_preview_jpeg(img_bgr, xs)     -> bytes  (JPEG para mostrar en web)
"""
import time
import cv2
import numpy as np

try:
    from scipy.ndimage import median_filter as _ndi_median
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# Detección de "meseta" del núcleo saturado: un segmento se considera núcleo
# quemado (plateau plano) si tiene al menos MESETA_MIN_ANCHO píxeles cuyo score
# llega al menos al MESETA_FRAC del máximo del segmento. En ese caso la posición
# se toma como el centroide de la meseta (subpíxel estable), en lugar del argmax
# + parábola, que en un techo plano salta y luego el filtro MAD elimina (huecos).
MESETA_FRAC = 0.92
MESETA_MIN_ANCHO = 4


# ─────────────────────────────────────────────────────────────────────────────
# 1. Score de color
# ─────────────────────────────────────────────────────────────────────────────

def laser_score(img_bgr, sat_min=40, val_min=30, val_alto=220, peso_nucleo=60.0):
    """Mapa float32  [0..255]  donde valores altos = pixel de láser rojo/magenta.

    El núcleo del láser satura el sensor (R≈G≈B altos, S HSV baja), por eso
    se añade un bonus de núcleo saturado para evitar la "línea doble".
    """
    b, g, r = cv2.split(img_bgr.astype(np.int16))
    rg       = r - g
    magenta  = np.minimum(r, b) - g

    v_raw       = np.maximum(np.maximum(r, g), b)
    nucleo      = np.clip(v_raw - val_alto, 0, 35).astype(np.float32) / 35.0
    no_verde    = (r >= g).astype(np.float32)
    bonus_nucleo = peso_nucleo * nucleo * no_verde

    score = 0.6 * rg + 0.8 * np.clip(magenta, 0, None) + bonus_nucleo
    score = np.clip(score, 0, 255).astype(np.float32)

    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    # Núcleo blanco sobreexpuesto: brillo muy alto y saturación baja (blanco),
    # evitando zonas verdosas (r>=g). "borde brillante" es el resguardo general.
    condicion_color_vivo      = (s_ch >= sat_min) & (v_ch >= val_min)
    condicion_borde_brillante = (v_ch >= val_alto) & (r >= g)
    mask = condicion_color_vivo | condicion_borde_brillante

    score *= mask.astype(np.float32)
    return score

# ─────────────────────────────────────────────────────────────────────────────
# 2. Detección por fila
# ─────────────────────────────────────────────────────────────────────────────

def _detectar_desde_score(score_s,
                          score_min, factor_umbral,
                          ancho_min, ancho_max,
                          ventana_continuidad, mad_factor):
    """Detecta la línea láser fila a fila sobre un mapa de score ya calculado.

    Devuelve array float32 de longitud H con la posición X (subpíxel) por fila,
    np.nan donde no se detectó nada válido.
    """
    H, W = score_s.shape
    arreglo_x = np.full(H, np.nan, dtype=np.float32)

    max_por_fila  = score_s.max(axis=1)
    filas_activas = np.where(max_por_fila >= score_min)[0]
    if filas_activas.size == 0:
        return arreglo_x
    umbrales = np.maximum(score_min, max_por_fila * factor_umbral)

    x_prev = None
    rachas = 0

    for y in filas_activas:
        fila   = score_s[y]
        sobre  = fila >= umbrales[y]

        diff   = np.diff(sobre.view(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends   = np.where(diff == -1)[0]
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
        ok     = (anchos >= ancho_min) & (anchos <= ancho_max)
        if not ok.any():
            rachas += 1
            if rachas > 30:
                x_prev = None
            continue
        starts = starts[ok]
        ends   = ends[ok]

        # Posición subpíxel de cada segmento candidato:
        #  - Núcleo saturado (meseta plana): centroide de la meseta → estable.
        #    Con argmax + parábola el techo plano salta de píxel y el filtro MAD
        #    posterior lo marca como outlier → huecos en la línea.
        #  - Pico nítido (láser no saturado): refinamiento parabólico de 3 puntos.
        peaks_x = np.empty(starts.size, dtype=np.float32)
        peaks_v = np.empty(starts.size, dtype=np.float32)
        for i, (s_i, e_i) in enumerate(zip(starts, ends)):
            seg        = fila[s_i:e_i + 1]
            seg_max    = float(seg.max())
            peaks_v[i] = seg_max

            techo = np.where(seg >= seg_max * MESETA_FRAC)[0]
            if techo.size >= MESETA_MIN_ANCHO:
                # Centroide ponderado de la meseta del núcleo (subpíxel estable)
                w = seg[techo].astype(np.float32)
                peaks_x[i] = s_i + float((techo * w).sum() / w.sum())
            else:
                k  = int(np.argmax(seg))
                xg = s_i + k
                if 0 < xg < W - 1:
                    L = float(fila[xg - 1]); C = float(fila[xg]); Rr = float(fila[xg + 1])
                    denom = L - 2.0 * C + Rr
                    delta = 0.5 * (L - Rr) / denom if denom != 0 else 0.0
                    peaks_x[i] = xg + float(np.clip(delta, -1.0, 1.0))
                else:
                    peaks_x[i] = float(xg)

        xs_sub = peaks_x

        # Selección del candidato:
        #  - Con pista previa: el más cercano dentro de la ventana de continuidad.
        #  - Si ninguno entra en la ventana, NO saltamos al más cercano (podría ser
        #    un objeto rojo / reflejo): elegimos el MÁS BRILLANTE, que es el láser.
        if x_prev is not None:
            d      = np.abs(xs_sub - x_prev)
            dentro = d <= ventana_continuidad
            idx    = int(np.argmin(np.where(dentro, d, np.inf))) if dentro.any() else int(np.argmax(peaks_v))
        else:
            idx = int(np.argmax(peaks_v))

        arreglo_x[y] = float(xs_sub[idx])
        x_prev = arreglo_x[y]
        rachas = 0

    # Filtro MAD contra mediana móvil
    return _filtro_mad(arreglo_x, mad_factor)


def _filtro_mad(arreglo_x, mad_factor):
    """Descarta como NaN los puntos cuyo residuo contra la mediana móvil supera
    mad_factor·MAD. Compartido por la detección clásica y la vectorizada."""
    H = arreglo_x.shape[0]
    valid_mask = ~np.isnan(arreglo_x)
    if valid_mask.sum() <= 20:
        return arreglo_x
    ventana = 31
    ys      = np.arange(H)
    if _HAS_SCIPY:
        arr_fill    = np.interp(ys, ys[valid_mask], arreglo_x[valid_mask])
        media_movil = _ndi_median(arr_fill.astype(np.float32), size=ventana, mode='nearest')
    else:
        media_movil = np.full(H, np.nan, dtype=np.float32)
        half = ventana // 2
        for y in range(H):
            tramo = arreglo_x[max(0, y - half):min(H, y + half + 1)]
            tramo = tramo[~np.isnan(tramo)]
            if tramo.size >= 5:
                media_movil[y] = np.median(tramo)

    residuo = np.abs(arreglo_x - media_movil)
    mad     = np.nanmedian(residuo)
    if mad > 0:
        arreglo_x = arreglo_x.copy()
        arreglo_x[residuo > mad_factor * max(mad, 1.0)] = np.nan
    return arreglo_x


def _detectar_vectorizado(score_s, score_min, factor_umbral,
                          ancho_min, ancho_max, mad_factor):
    """Detección por fila SIN loop Python: todo en operaciones sobre la matriz.

    Por cada fila toma el pico más brillante (argmax), valida el ancho del tramo
    contiguo sobre umbral y refina subpíxel (centroide de meseta saturada o
    parábola). No usa la continuidad fila-a-fila del método clásico; para una
    línea láser limpia da prácticamente el mismo resultado, mucho más rápido.
    """
    H, W = score_s.shape
    arreglo_x = np.full(H, np.nan, dtype=np.float32)
    if W < 3:
        return arreglo_x

    rows     = np.arange(H)
    cols     = np.arange(W, dtype=np.int32)
    max_fila = score_s.max(axis=1)
    umbral   = np.maximum(score_min, max_fila * factor_umbral)      # (H,)
    sobre    = score_s >= umbral[:, None]                           # (H,W)
    peak     = score_s.argmax(axis=1)                               # (H,) primer máx

    # Ancho del tramo contiguo 'sobre' que contiene al pico (vectorizado):
    #   borde izq = último índice False ≤ peak ; borde der = primer False ≥ peak
    notover     = ~sobre
    left_false  = np.maximum.accumulate(
        np.where(notover, cols[None, :], np.int32(-1)), axis=1)
    right_false = np.minimum.accumulate(
        np.where(notover, cols[None, :], np.int32(W))[:, ::-1], axis=1)[:, ::-1]
    l_bound = left_false[rows, peak] + 1
    r_bound = right_false[rows, peak] - 1
    ancho   = r_bound - l_bound + 1

    # Subpíxel por meseta (centroide de la zona casi-máxima) o parábola
    plat  = (score_s >= (max_fila * MESETA_FRAC)[:, None]) & sobre
    nplat = plat.sum(axis=1)
    w     = score_s * plat
    sumw  = w.sum(axis=1)
    centroid = np.divide((w * cols[None, :]).sum(axis=1), sumw,
                         out=np.zeros(H, np.float64), where=sumw > 0)

    pk    = np.clip(peak, 1, W - 2)
    L     = score_s[rows, pk - 1].astype(np.float32)
    C     = score_s[rows, pk].astype(np.float32)
    Rr    = score_s[rows, pk + 1].astype(np.float32)
    denom = L - 2.0 * C + Rr
    delta = np.clip(np.divide(0.5 * (L - Rr), denom,
                              out=np.zeros(H, np.float32), where=denom != 0),
                    -1.0, 1.0)
    x_par = np.where((peak > 0) & (peak < W - 1),
                     pk.astype(np.float32) + delta, peak.astype(np.float32))

    x_sub = np.where(nplat >= MESETA_MIN_ANCHO, centroid.astype(np.float32), x_par)

    valido = (max_fila >= score_min) & (ancho >= ancho_min) & (ancho <= ancho_max)
    arreglo_x[valido] = x_sub[valido]

    return _filtro_mad(arreglo_x, mad_factor)


# ─────────────────────────────────────────────────────────────────────────────
# 3. API pública
# ─────────────────────────────────────────────────────────────────────────────

# Parámetros con valor fijo (no se exponen en la UI)
_DEFAULTS = dict(
    factor_umbral=0.55,
    ventana_continuidad=40,
    mad_factor=4.0,
    cierre_x=0,
)


def _find_laser_band(img_bgr, sat_min, val_min, val_alto, score_min,
                     scale=0.25, margin=120):
    """Localiza, sobre una imagen reducida (barata), la franja horizontal de
    columnas donde aparece el láser. Devuelve (x0, x1) en píxeles full-res, o
    None si no encuentra nada (→ el llamador procesa el frame completo).

    La idea: como la línea es casi vertical, basta con saber su rango en X para
    recortar una banda y procesar a full-res solo ahí (mucho menos píxeles).
    """
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    sc = laser_score(small, sat_min=sat_min, val_min=val_min, val_alto=val_alto)
    # Nº de filas con láser por columna. Más robusto que el máximo: un píxel de
    # ruido aislado no cuenta; la línea (casi vertical) acumula varias filas por
    # columna en su franja.
    col_count = (sc >= score_min).sum(axis=0)
    h_small = sc.shape[0]
    min_filas = max(3, int(0.01 * h_small))
    cols = np.where(col_count >= min_filas)[0]
    if cols.size == 0:
        return None
    W = img_bgr.shape[1]
    x0 = max(0, int(cols.min() / scale) - margin)
    x1 = min(W, int(cols.max() / scale) + margin + 1)
    return (x0, x1)


def detect_laser(img_bgr,
                 sat_min=40,
                 val_min=30,
                 val_alto=220,
                 score_min=18,
                 kernel_suave_x=9,
                 ancho_min=2,
                 ancho_max=60,
                 kernel_2d=3,
                 peso_nucleo=60.0,
                 cierre_x=0,
                 roi=False,
                 vectorizado=True,
                 # parámetros fijos (se pueden sobreescribir si se necesita)
                 factor_umbral=None,
                 ventana_continuidad=None,
                 mad_factor=None):
    """Detecta la línea láser en *img_bgr* y devuelve un dict con resultados.

    Parámetros expuestos en UI: sat_min, val_min, val_alto, score_min,
                                kernel_suave_x, ancho_min, ancho_max, kernel_2d.

    Returns:
        {
          "xs":        list[float|None],   # posición X por fila (None = no detectado)
          "pct":       float,              # % de filas con detección válida
          "ms":        float,              # tiempo de proceso en ms
          "H": int, "W": int               # dimensiones de la imagen
        }
    """
    t0 = time.perf_counter()

    # Rellenar defaults para parámetros fijos
    if factor_umbral      is None: factor_umbral      = _DEFAULTS["factor_umbral"]
    if ventana_continuidad is None: ventana_continuidad = _DEFAULTS["ventana_continuidad"]
    if mad_factor         is None: mad_factor         = _DEFAULTS["mad_factor"]
    if cierre_x           is None: cierre_x           = _DEFAULTS["cierre_x"]

    H, W = img_bgr.shape[:2]

    # 0) ROI: procesar solo una banda alrededor de la línea (mucho menos píxeles).
    #    Pasada gruesa en imagen reducida para ubicar la franja en X.
    t_a = time.perf_counter()
    x_off = 0
    work = img_bgr
    band_used = None
    if roi:
        banda = _find_laser_band(img_bgr, int(sat_min), int(val_min),
                                 int(val_alto), float(score_min))
        if banda is not None:
            x0, x1 = banda
            work = img_bgr[:, x0:x1]
            x_off = x0
            band_used = (x0, x1)
    t_roi = (time.perf_counter() - t_a) * 1000

    # 1) Score de color (sobre la banda si hay ROI)
    t_a = time.perf_counter()
    score = laser_score(work,
                        sat_min=int(sat_min),
                        val_min=int(val_min),
                        val_alto=int(val_alto),
                        peso_nucleo=float(peso_nucleo))
    t_score = (time.perf_counter() - t_a) * 1000

    # 2a) Blur 2D opcional (mejora conectividad entre píxeles vecinos)
    t_a = time.perf_counter()
    k2 = int(kernel_2d)
    if k2 >= 3:
        if k2 % 2 == 0:
            k2 += 1
        score = cv2.GaussianBlur(score, (k2, k2), 0)

    # 2b) Blur horizontal + cierre morfológico opcional
    k = int(kernel_suave_x)
    if k % 2 == 0:
        k += 1
    score_s = score if k <= 1 else cv2.GaussianBlur(score, (k, 1), 0)
    cx = int(cierre_x)
    if cx >= 2:
        kern    = np.ones((1, cx), dtype=np.uint8)
        score_s = cv2.morphologyEx(score_s, cv2.MORPH_CLOSE, kern)
    t_blur = (time.perf_counter() - t_a) * 1000

    # 3) Detección por fila
    t_a = time.perf_counter()
    if vectorizado:
        xs = _detectar_vectorizado(
            score_s,
            score_min=float(score_min),
            factor_umbral=float(factor_umbral),
            ancho_min=int(ancho_min),
            ancho_max=int(ancho_max),
            mad_factor=float(mad_factor),
        )
    else:
        xs = _detectar_desde_score(
            score_s,
            score_min=float(score_min),
            factor_umbral=float(factor_umbral),
            ancho_min=int(ancho_min),
            ancho_max=int(ancho_max),
            ventana_continuidad=int(ventana_continuidad),
            mad_factor=float(mad_factor),
        )
    t_detect = (time.perf_counter() - t_a) * 1000

    # Desplazar las X de la banda al sistema de la imagen completa (NaN se mantiene)
    if x_off:
        xs = xs + x_off

    banda_txt = f"banda={work.shape[1]}px " if roi else ""
    print(f"⏱️  detect_laser [{W}x{H}]: {banda_txt}roi={t_roi:.0f}ms  "
          f"score={t_score:.0f}ms  blur={t_blur:.0f}ms  detect={t_detect:.0f}ms  "
          f"(scipy={'sí' if _HAS_SCIPY else 'NO'})")

    ms     = (time.perf_counter() - t0) * 1000
    n_ok   = int((~np.isnan(xs)).sum())
    pct    = 100.0 * n_ok / H if H > 0 else 0.0

    # Serializar: NaN → None para JSON
    xs_list = [None if np.isnan(v) else round(float(v), 3) for v in xs]

    return {
        "xs":     xs_list,
        "pct":    round(pct, 1),
        "ms":     round(ms, 1),
        "H":      H,
        "W":      W,
        "score_s": score_s,   # score post-blur para heatmap (de la banda si hay ROI)
        "band":    list(band_used) if band_used else None,  # [x0,x1] usado, o None
    }


def build_preview_jpeg(img_bgr, xs_list, scale=0.5, quality=75):
    """Dibuja la línea detectada sobre la imagen y devuelve bytes JPEG.

    xs_list: list[float|None] de longitud H (salida de detect_laser).
    scale:   factor de reducción para aligerar la transferencia.
    """
    if scale < 1.0:
        img_show = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
    else:
        img_show = img_bgr.copy()

    H_s, W_s = img_show.shape[:2]

    for y_orig, x_val in enumerate(xs_list):
        if x_val is None:
            continue
        y_s = int(round(y_orig * scale))
        x_s = int(round(x_val  * scale))
        if 0 <= y_s < H_s and 0 <= x_s < W_s:
            cv2.line(img_show,
                     (max(0, x_s - 1), y_s),
                     (min(W_s - 1, x_s + 1), y_s),
                     (0, 255, 0), 1)

    ok, buf = cv2.imencode('.jpg', img_show, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf) if ok else b''


def build_heatmap_jpeg(score, scale=0.5, quality=75):
    """Convierte el mapa de score float32 en imagen de calor (INFERNO) como JPEG.

    score: np.ndarray float32 H×W con valores 0..255 (salida de laser_score).
    """
    if scale < 1.0:
        h, w    = score.shape
        score   = cv2.resize(score,
                             (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)

    s_max = float(score.max())
    if s_max > 0:
        norm = (score * (255.0 / s_max)).astype(np.uint8)
    else:
        norm = score.astype(np.uint8)

    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    ok, buf = cv2.imencode('.jpg', heatmap, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf) if ok else b''
