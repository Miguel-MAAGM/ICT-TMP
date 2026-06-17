import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def _laser_score(img_bgr, sat_min=40, val_min=30, val_alto=220, peso_nucleo=60.0):
    """Mapa float32 donde valores altos = pixel parecido a láser rojo/magenta.

    Problema típico: el núcleo del láser SATURA el sensor (R=G=B altos, casi blanco),
    así que tiene saturación HSV baja y `R-G` cercano a 0 → quedaba un hueco
    central ("línea doble"). Para evitarlo:
      - Score = rojez base + bonus magenta + bonus núcleo saturado (V alto y
        no dominado por verde).
      - Máscara HSV: válido si (S>=sat_min y V>=val_min) O bien V>=val_alto
        (núcleo saturado) siempre que R no sea menor que G (no es verde).
    """
    b, g, r = cv2.split(img_bgr.astype(np.int16))
    rg = r - g                                    # rojez base
    magenta = np.minimum(r, b) - g                # bonus si hay rojo Y azul (=magenta)

    # Bonus núcleo: pixeles muy brillantes donde R >= G (no verdes).
    # Esto activa el centro saturado del láser sin disparar con verdes o sombras.
    v_raw = np.maximum(np.maximum(r, g), b)       # equivalente al V de HSV
    nucleo = np.clip(v_raw - val_alto, 0, 35).astype(np.float32) / 35.0   # 0..1
    no_verde = (r >= g).astype(np.float32)
    bonus_nucleo = peso_nucleo * nucleo * no_verde

    score = 0.6 * rg + 0.8 * np.clip(magenta, 0, None) + bonus_nucleo
    score = np.clip(score, 0, 255).astype(np.float32)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((s >= sat_min) & (v >= val_min)) | ((v >= val_alto) & (r >= g))
    score *= mask.astype(np.float32)
    return score


def _subpixel_parabola(y_left, y_center, y_right, x_center):
    denom = (y_left - 2.0 * y_center + y_right)
    if denom == 0:
        return float(x_center)
    delta = 0.5 * (y_left - y_right) / denom
    if delta < -1.0 or delta > 1.0:
        delta = 0.0
    return float(x_center) + delta


def detectar_laser(ruta_imagen,
                   score_min=18,
                   factor_umbral=0.55,
                   ancho_max=60,
                   ancho_min=2,
                   kernel_suave_x=9,
                   ventana_continuidad=40,
                   mad_factor=4.0,
                   debug=True):
    """Detección robusta de línea láser vertical.

    Pipeline:
      1) Score por color (R-G + bonus magenta), enmascarado HSV.
      2) Suavizado horizontal.
      3) Por fila: todos los picos sobre umbral dinámico; se elige por continuidad
         con la fila anterior (si la hay) o por mayor intensidad.
      4) Sub-pixel por ajuste parabólico de 3 puntos.
      5) Filtrado MAD contra mediana móvil para eliminar outliers.
    """
    if not os.path.isfile(ruta_imagen):
        raise FileNotFoundError(f"No existe el archivo: {ruta_imagen}")
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise ValueError(f"OpenCV no pudo decodificar: {ruta_imagen}")
    H, W = img.shape[:2]
    if debug:
        print(f"📸 {W}x{H}")

    score = _laser_score(img)
    if kernel_suave_x % 2 == 0:
        kernel_suave_x += 1
    score_s = cv2.GaussianBlur(score, (kernel_suave_x, 1), 0)

    arreglo_x = np.full(H, np.nan, dtype=np.float32)

    x_prev = None
    rachas_sin_deteccion = 0

    for y in range(H):
        fila = score_s[y, :]
        max_v = float(fila.max())
        if max_v < score_min:
            rachas_sin_deteccion += 1
            if rachas_sin_deteccion > 30:
                x_prev = None
            continue

        umbral = max(score_min, max_v * factor_umbral)
        sobre = fila >= umbral
        diff = np.diff(sobre.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0]
        if sobre[0]:
            starts = np.r_[0, starts]
        if sobre[-1]:
            ends = np.r_[ends, len(sobre) - 1]

        candidatos = []
        for s_i, e_i in zip(starts, ends):
            ancho = e_i - s_i
            if ancho < ancho_min or ancho > ancho_max:
                continue
            seg = fila[s_i:e_i + 1]
            k = int(np.argmax(seg))
            x_peak = s_i + k
            if 0 < x_peak < W - 1:
                x_sub = _subpixel_parabola(
                    float(fila[x_peak - 1]),
                    float(fila[x_peak]),
                    float(fila[x_peak + 1]),
                    x_peak,
                )
            else:
                x_sub = float(x_peak)
            candidatos.append((x_sub, float(fila[x_peak])))

        if not candidatos:
            rachas_sin_deteccion += 1
            if rachas_sin_deteccion > 30:
                x_prev = None
            continue

        if x_prev is not None:
            mejor = min(
                candidatos,
                key=lambda c: (abs(c[0] - x_prev) > ventana_continuidad,
                               abs(c[0] - x_prev), -c[1]),
            )
        else:
            mejor = max(candidatos, key=lambda c: c[1])

        arreglo_x[y] = mejor[0]
        x_prev = mejor[0]
        rachas_sin_deteccion = 0

    # Filtro MAD vs mediana móvil
    arreglo_x_filt = arreglo_x.copy()
    if (~np.isnan(arreglo_x_filt)).sum() > 20:
        ventana = 31
        media_movil = np.full(H, np.nan, dtype=np.float32)
        for y in range(H):
            y0 = max(0, y - ventana // 2)
            y1 = min(H, y + ventana // 2 + 1)
            tramo = arreglo_x_filt[y0:y1]
            tramo = tramo[~np.isnan(tramo)]
            if tramo.size >= 5:
                media_movil[y] = np.median(tramo)
        residuo = np.abs(arreglo_x_filt - media_movil)
        mad = np.nanmedian(residuo)
        if mad > 0:
            mala = residuo > mad_factor * max(mad, 1.0)
            arreglo_x_filt[mala] = np.nan
            if debug:
                print(f"🧹 MAD={mad:.2f}px, rechazados {int(mala.sum())} outliers")

    n_validas = int((~np.isnan(arreglo_x_filt)).sum())
    if debug:
        print(f"✅ Filas válidas: {n_validas}/{H} ({100*n_validas/H:.1f}%)")

        img_crudo = img.copy()
        img_final = img.copy()
        lienzo = np.zeros_like(img)
        for y in range(H):
            if not np.isnan(arreglo_x[y]):
                cv2.circle(img_crudo, (int(round(arreglo_x[y])), y), 1, (255, 0, 0), -1)
            if not np.isnan(arreglo_x_filt[y]):
                xpx = int(round(arreglo_x_filt[y]))
                cv2.circle(img_final, (xpx, y), 1, (0, 255, 0), -1)
                cv2.circle(lienzo, (xpx, y), 1, (0, 255, 0), -1)

        fig, ejes = plt.subplots(1, 5, figsize=(22, 9), sharey=True)
        ejes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ejes[0].set_title("1. Original")
        ejes[1].imshow(score_s, cmap='inferno');               ejes[1].set_title("2. Score láser (color)")
        ejes[2].imshow(cv2.cvtColor(img_crudo, cv2.COLOR_BGR2RGB)); ejes[2].set_title("3. Crudo (azul)")
        ejes[3].imshow(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)); ejes[3].set_title("4. Filtrado (verde)")
        ejes[4].imshow(cv2.cvtColor(lienzo, cv2.COLOR_BGR2RGB));    ejes[4].set_title("5. Perfil")
        for ax in ejes:
            ax.grid(True, linestyle=':', alpha=0.3, color='white')
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    return arreglo_x_filt


if __name__ == "__main__":
    archivo = "ALL.jpg"
    try:
        xs = detectar_laser(archivo)
        np.savetxt(
            "arreglo_laser_real.txt",
            np.column_stack([np.arange(xs.size), xs]),
            fmt=["%d", "%.3f"],
            header="y\tx_subpixel",
            comments="",
            delimiter="\t",
        )
        print("💾 Guardado en 'arreglo_laser_real.txt'")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}")
