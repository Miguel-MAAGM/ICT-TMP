"""Calibra el PLANO DEL LÁSER por triangulación con un chessboard.

Idea: en cada foto el láser cruza el tablero. Con solvePnP conocemos el plano
del tablero en coordenadas de cámara; retroproyectando los píxeles del láser que
caen DENTRO del tablero obtenemos puntos 3D que pertenecen al plano del láser.
Acumulando varias poses distintas, ajustamos el plano por SVD.

Requisitos de las fotos (¡importante!):
  * El láser debe caer VISIBLEMENTE SOBRE el tablero (no solo sobre el suelo).
  * Varias poses distintas: inclinaciones y distancias diferentes.
    Con todas las fotos fronto-paralelas el problema queda mal condicionado.
  * Chessboard completo y detectable (cols x rows esquinas internas).
  * Sistema quieto al disparar (rolling shutter).

Uso:
    python calibrate_laser_plane.py ^
        --images "calib_laser/*.jpg" ^
        --camera web/static/calibraciones/mi_calib.json ^
        --cols 7 --rows 6 --square-mm 25 ^
        --out laser_plane.json --debug-dir calib_laser/debug

Convención de coordenadas: OpenCV (X→derecha, Y→abajo, Z→adelante), en mm.
El plano del láser se guarda como  n · X = d  (n unitario).
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Intrínsecos
# ---------------------------------------------------------------------------
def load_camera_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["distortion_coefficients"], dtype=np.float64).reshape(-1)
    image_size = tuple(data.get("image_size", []))
    return K, dist, image_size


# ---------------------------------------------------------------------------
# Chessboard + pose
# ---------------------------------------------------------------------------
def detect_chessboard(img_bgr, cols, rows):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             + cv2.CALIB_CB_NORMALIZE_IMAGE
             + cv2.CALIB_CB_FAST_CHECK)
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def build_object_points(cols, rows, square_mm):
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_mm)
    return objp


def solve_pose(object_points, image_points, K, dist):
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP no convergió")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), rvec, tvec


def board_plane_camframe(R, t):
    """Plano del tablero en frame cámara como (n, d) con n·X = d, n unitario.

    El tablero está en Z=0 en su propio frame; su normal es R[:, 2] y pasa por t.
    """
    n = R[:, 2].astype(np.float64)
    n = n / np.linalg.norm(n)
    d = float(n @ t)
    return n, d


# ---------------------------------------------------------------------------
# Detección del láser (línea ~vertical: un punto por fila)
# ---------------------------------------------------------------------------
def detect_laser_points(img_bgr, brillo_min=40, factor_umbral=0.75,
                        ancho_max=80, kernel_suavizado=15, mask=None):
    """Devuelve array Nx2 de píxeles (u, v) del centro del láser por fila.

    Usa realce del canal rojo menos el promedio de verde/azul para aislar el
    láser rojo del fondo, y centroide ponderado sub-píxel por fila.
    Si se pasa `mask` (bool HxW), solo considera píxeles dentro de la máscara.
    """
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    # Realce de "rojez": fuerte donde el rojo domina sobre verde y azul.
    redness = r - 0.5 * (g + b)
    redness = np.clip(redness, 0, 255)

    if kernel_suavizado % 2 == 0:
        kernel_suavizado += 1
    redness = cv2.GaussianBlur(redness, (kernel_suavizado, 1), 0)

    if mask is not None:
        redness = np.where(mask, redness, 0.0)

    h, w = redness.shape
    puntos = []
    for y in range(h):
        fila = redness[y, :]
        max_brillo = fila.max()
        if max_brillo <= brillo_min:
            continue
        umbral = max_brillo * factor_umbral
        idx = np.where(fila >= umbral)[0]
        if idx.size == 0:
            continue
        izq, der = int(idx[0]), int(idx[-1])
        if (der - izq) >= ancho_max:
            continue
        seg = fila[izq:der + 1]
        pesos = seg - umbral
        np.maximum(pesos, 0, out=pesos)
        s = pesos.sum()
        if s <= 0:
            continue
        u = izq + float((pesos * np.arange(seg.size)).sum() / s)
        puntos.append((u, float(y)))
    return np.asarray(puntos, dtype=np.float64).reshape(-1, 2)


def board_mask(img_shape, corners, cols, rows, margin_frac=0.06):
    """Máscara booleana del interior del tablero (convex hull de las esquinas,
    encogida hacia el centro para evitar el borde blanco/negro)."""
    h, w = img_shape[:2]
    pts = corners.reshape(-1, 2)
    centroid = pts.mean(axis=0)
    shrunk = centroid + (pts - centroid) * (1.0 - margin_frac)
    hull = cv2.convexHull(shrunk.astype(np.float32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 1)
    return mask.astype(bool), hull


# ---------------------------------------------------------------------------
# Backprojection: píxel -> punto 3D sobre el plano del tablero
# ---------------------------------------------------------------------------
def pixels_to_plane_points(pixels_uv, K, dist, n, d):
    """Intersecta el rayo de cada píxel (corregido de distorsión) con el plano
    del tablero (n·X = d). Devuelve puntos 3D Nx3 en frame cámara (mm)."""
    if pixels_uv.shape[0] == 0:
        return np.empty((0, 3))

    # Corrige distorsión y normaliza: undistortPoints con P=None da coords
    # normalizadas (x, y) tal que el rayo es d = (x, y, 1).
    undist = cv2.undistortPoints(pixels_uv.reshape(-1, 1, 2), K, dist)
    undist = undist.reshape(-1, 2)
    rays = np.hstack([undist, np.ones((undist.shape[0], 1))])  # Nx3

    denom = rays @ n  # N
    valid = np.abs(denom) > 1e-9
    t = np.zeros_like(denom)
    t[valid] = d / denom[valid]
    pts = rays * t[:, None]
    return pts[valid]


# ---------------------------------------------------------------------------
# Ajuste de plano por SVD
# ---------------------------------------------------------------------------
def fit_plane(points):
    """Ajusta n·X = d (n unitario) a una nube Nx3 minimizando distancia
    ortogonal. Devuelve (n, d, residuos_rms_mm)."""
    if points.shape[0] < 3:
        raise ValueError("Se necesitan al menos 3 puntos para ajustar un plano")
    centroid = points.mean(axis=0)
    centered = points - centroid
    # La normal es el vector singular asociado al menor valor singular.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    d = float(n @ centroid)
    residuals = points @ n - d
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return n, d, rms


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Calibra el plano del láser usando un chessboard como referencia.")
    parser.add_argument("--images", required=True,
                        help="Glob de imágenes, p.ej. \"calib_laser/*.jpg\"")
    parser.add_argument("--camera", required=True, help="JSON intrínseco")
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--brillo-min", type=float, default=40.0)
    parser.add_argument("--factor-umbral", type=float, default=0.75)
    parser.add_argument("--ancho-max", type=int, default=80)
    parser.add_argument("--out", default="laser_plane.json")
    parser.add_argument("--debug-dir", default=None,
                        help="Si se indica, guarda overlays de cada imagen")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.images))
    if not paths:
        print(f"❌ No se encontraron imágenes con el patrón: {args.images}")
        sys.exit(1)
    if not os.path.isfile(args.camera):
        print(f"❌ No existe el JSON de cámara: {args.camera}")
        sys.exit(1)

    K, dist, cam_size = load_camera_json(args.camera)
    print(f"📷 Intrínsecos: {args.camera}")
    print(f"🖼️  {len(paths)} imágenes encontradas.\n")

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    objp = build_object_points(args.cols, args.rows, args.square_mm)
    all_points = []
    used = []
    per_image = []

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  ⏭️  {os.path.basename(path)}: no se pudo leer")
            continue

        ok, corners = detect_chessboard(img, args.cols, args.rows)
        if not ok:
            print(f"  ⏭️  {os.path.basename(path)}: chessboard NO detectado")
            continue

        R, t, rvec, tvec = solve_pose(objp, corners, K, dist)
        n_board, d_board = board_plane_camframe(R, t)

        mask, hull = board_mask(img.shape, corners, args.cols, args.rows)
        laser_uv = detect_laser_points(
            img, brillo_min=args.brillo_min, factor_umbral=args.factor_umbral,
            ancho_max=args.ancho_max, mask=mask)

        if laser_uv.shape[0] < 5:
            print(f"  ⚠️  {os.path.basename(path)}: láser sobre el tablero "
                  f"insuficiente ({laser_uv.shape[0]} px). ¿El láser cruza el tablero?")
            continue

        pts3d = pixels_to_plane_points(laser_uv, K, dist, n_board, d_board)
        all_points.append(pts3d)
        used.append(path)
        per_image.append((os.path.basename(path), laser_uv.shape[0], pts3d))
        print(f"  ✅ {os.path.basename(path)}: {laser_uv.shape[0]} px de láser "
              f"sobre tablero → {pts3d.shape[0]} puntos 3D")

        if args.debug_dir:
            dbg = img.copy()
            cv2.polylines(dbg, [hull.astype(np.int32)], True, (0, 255, 255), 2)
            for u, v in laser_uv:
                cv2.circle(dbg, (int(round(u)), int(round(v))), 2, (0, 255, 0), -1)
            out_dbg = os.path.join(args.debug_dir,
                                   f"dbg_{os.path.basename(path)}")
            cv2.imwrite(out_dbg, dbg)

    if not all_points:
        print("\n❌ Ninguna imagen aportó puntos. Revisa que el láser cruce el tablero.")
        sys.exit(2)

    points = np.vstack(all_points)
    print(f"\n🔧 Ajustando plano con {points.shape[0]} puntos de {len(used)} imágenes...")

    n, d, rms = fit_plane(points)
    # Normalizamos el signo para que d sea positivo (plano delante de la cámara).
    if d < 0:
        n, d = -n, -d

    print(f"   normal láser : ({n[0]:+.5f}, {n[1]:+.5f}, {n[2]:+.5f})")
    print(f"   d            : {d:.3f} mm")
    print(f"   RMS residual : {rms:.4f} mm  "
          f"[{'OK' if rms < 1.0 else 'WARN' if rms < 3.0 else 'BAD'}]")

    # Diversidad de poses: si todos los puntos casi colineales, avisar.
    _, sv, _ = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    if sv[1] < 1e-3 * sv[0]:
        print("   ⚠️  Los puntos son casi colineales (poca diversidad de poses). "
              "El plano puede estar mal determinado: añade fotos con el tablero "
              "más inclinado / a distintas distancias.")

    payload = {
        "plane_normal": n.tolist(),
        "plane_d_mm": d,
        "convention": "n . X = d ; X en mm, frame cámara OpenCV",
        "rms_residual_mm": rms,
        "num_points": int(points.shape[0]),
        "num_images_used": len(used),
        "images_used": [os.path.basename(p) for p in used],
        "camera_calibration": os.path.basename(args.camera),
        "chessboard": {"cols": args.cols, "rows": args.rows,
                       "square_mm": args.square_mm},
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n💾 Plano del láser guardado en: {args.out}")
    if args.debug_dir:
        print(f"🐛 Overlays de depuración en: {args.debug_dir}")


if __name__ == "__main__":
    main()
