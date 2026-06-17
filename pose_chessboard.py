"""Detecta un chessboard, resuelve su pose con solvePnP y reporta la desviación
angular del plano respecto al eje óptico de la cámara.

Uso típico:
    python pose_chessboard.py --image foto.jpg ^
        --camera web/static/calibraciones/mi_calib.json ^
        --cols 7 --rows 6 --square-mm 25 --show

Convenciones de coordenadas (OpenCV estándar):
    X → derecha, Y → abajo, Z → adelante (desde la cámara).
    Tablero fronto-paralelo perfecto  ⇔  normal del tablero = (0, 0, -1).
"""

import argparse
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Carga de intrínsecos
# ---------------------------------------------------------------------------
def load_camera_json(path: str):
    """Devuelve (K, dist, image_size) desde el JSON de /calibration/compute."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["distortion_coefficients"], dtype=np.float64).reshape(-1)
    image_size = tuple(data.get("image_size", []))  # (w, h) si existe
    return K, dist, image_size, data


# ---------------------------------------------------------------------------
# Detección del chessboard
# ---------------------------------------------------------------------------
def detect_chessboard(img_bgr, cols, rows):
    """Detecta esquinas internas (cols x rows). Devuelve (ok, corners_subpix)."""
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
    """Coordenadas 3D de las esquinas del tablero (Z=0 en su propio frame)."""
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_mm)
    return objp


# ---------------------------------------------------------------------------
# Pose y métricas
# ---------------------------------------------------------------------------
def solve_pose(object_points, image_points, K, dist):
    """solvePnP con refinamiento iterativo. Devuelve (R, t, rvec, tvec)."""
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP no convergió")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), rvec, tvec


def pose_metrics(R, t):
    """Calcula desviación angular y descomposición pitch/yaw/roll en grados.

    Para un chessboard fronto-paralelo perfecto, su normal en frame cámara
    apunta a (0, 0, -1), es decir R[:, 2] = (0, 0, -1).

    - θ_total: ángulo entre la normal del tablero y -Z_cam.
    - pitch  : inclinación arriba/abajo (rot. alrededor de X_cam).
    - yaw    : inclinación izq/der    (rot. alrededor de Y_cam).
    - roll   : giro alrededor del eje óptico (rot. alrededor de Z_cam).
    """
    n = R[:, 2]  # normal del tablero en frame cámara
    # OpenCV no fija signo único para R[:, 2] según el orden de detección de
    # esquinas; la perpendicularidad es la misma para n y -n. Usamos |n_z|
    # para medir desviación contra el eje óptico, sin importar el sentido.
    cos_theta = float(np.clip(abs(n[2]), 0.0, 1.0))
    theta_total = np.degrees(np.arccos(cos_theta))

    # Pitch/yaw/roll a partir de la matriz alineada (n proyectada sobre -Z_cam),
    # así los signos son consistentes aunque la normal venga invertida.
    R_aligned = R if n[2] < 0 else R @ np.diag([1.0, -1.0, -1.0])

    pitch = np.degrees(np.arctan2(-R_aligned[1, 2], R_aligned[2, 2]))
    yaw = np.degrees(np.arctan2(R_aligned[0, 2],
                                np.hypot(R_aligned[1, 2], R_aligned[2, 2])))
    roll = np.degrees(np.arctan2(-R_aligned[0, 1], R_aligned[0, 0]))

    distance_mm = float(np.linalg.norm(t))
    center_xyz = t.tolist()

    return {
        "theta_total_deg": float(theta_total),
        "pitch_deg": float(pitch),
        "yaw_deg": float(yaw),
        "roll_deg": float(roll),
        "distance_mm": distance_mm,
        "center_xyz_mm": center_xyz,
        "normal_camframe": n.tolist(),
    }


def reprojection_rms(object_points, image_points, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    diffs = proj - image_points.reshape(-1, 2)
    return float(np.sqrt((diffs ** 2).sum(axis=1).mean()))


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def semaforo(value, ok_max, warn_max):
    if value <= ok_max:
        return "OK "
    if value <= warn_max:
        return "WARN"
    return "BAD "


def draw_overlay(img_bgr, corners, cols, rows, K, dist, rvec, tvec, square_mm):
    """Dibuja el chessboard detectado y los ejes 3D del tablero."""
    out = img_bgr.copy()
    cv2.drawChessboardCorners(out, (cols, rows), corners, True)
    axis_len = float(square_mm) * 3.0
    cv2.drawFrameAxes(out, K, dist, rvec, tvec, axis_len, thickness=4)
    return out


def print_report(metrics, rms_px):
    th = metrics["theta_total_deg"]
    print()
    print("=" * 52)
    print(" DIAGNÓSTICO DE POSE DEL CHESSBOARD")
    print("=" * 52)
    print(f"  θ_total       : {th:+7.2f}°  [{semaforo(abs(th), 1.0, 3.0)}]"
          "   (perpendicularidad; objetivo < 1°)")
    print(f"  pitch (↕)     : {metrics['pitch_deg']:+7.2f}°")
    print(f"  yaw   (↔)     : {metrics['yaw_deg']:+7.2f}°")
    print(f"  roll  (⟲)     : {metrics['roll_deg']:+7.2f}°")
    print(f"  distancia ‖t‖ : {metrics['distance_mm']:7.1f} mm")
    cx, cy, cz = metrics["center_xyz_mm"]
    print(f"  centro tablero: X={cx:+8.1f}  Y={cy:+8.1f}  Z={cz:+8.1f}  mm")
    nx, ny, nz = metrics["normal_camframe"]
    print(f"  normal (cam)  : ({nx:+.4f}, {ny:+.4f}, {nz:+.4f})")
    print(f"  reproyección  : {rms_px:.3f} px  [{semaforo(rms_px, 0.5, 1.5)}]")
    print("=" * 52)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pose de chessboard + desviación angular respecto a cámara.")
    parser.add_argument("--image", required=True, help="Imagen con chessboard visible")
    parser.add_argument("--camera", required=True,
                        help="JSON intrínseco de /calibration/compute")
    parser.add_argument("--cols", type=int, default=7,
                        help="Esquinas internas en horizontal (default 7)")
    parser.add_argument("--rows", type=int, default=6,
                        help="Esquinas internas en vertical (default 6)")
    parser.add_argument("--square-mm", type=float, default=25.0,
                        help="Tamaño del cuadrado en mm (default 25)")
    parser.add_argument("--show", action="store_true",
                        help="Muestra ventana con ejes proyectados")
    parser.add_argument("--save-overlay", default=None,
                        help="Si se indica, guarda imagen con ejes dibujados")
    parser.add_argument("--out-json", default=None,
                        help="Si se indica, guarda métricas en JSON")
    args = parser.parse_args()

    # Cargar entradas
    if not os.path.isfile(args.image):
        print(f"❌ No existe la imagen: {args.image}")
        sys.exit(1)
    if not os.path.isfile(args.camera):
        print(f"❌ No existe el JSON de cámara: {args.camera}")
        sys.exit(1)

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"❌ OpenCV no pudo decodificar la imagen: {args.image}")
        sys.exit(1)

    K, dist, cam_size, _cam_raw = load_camera_json(args.camera)
    print(f"📷 Intrínsecos cargados desde: {args.camera}")
    if cam_size:
        h, w = img_bgr.shape[:2]
        if (w, h) != tuple(cam_size):
            print(f"⚠️  Tamaño imagen {(w, h)} ≠ tamaño de calibración {tuple(cam_size)}. "
                  "La pose puede ser incorrecta si la imagen está escalada.")

    # Detectar tablero
    ok, corners = detect_chessboard(img_bgr, args.cols, args.rows)
    if not ok:
        print(f"❌ No se detectó chessboard {args.cols}x{args.rows}. "
              "Verifica cols/rows y la iluminación.")
        sys.exit(2)
    print(f"✅ Chessboard {args.cols}x{args.rows} detectado.")

    # Pose
    objp = build_object_points(args.cols, args.rows, args.square_mm)
    R, t, rvec, tvec = solve_pose(objp, corners, K, dist)
    metrics = pose_metrics(R, t)
    rms = reprojection_rms(objp, corners, rvec, tvec, K, dist)

    print_report(metrics, rms)

    # Guardar JSON
    if args.out_json:
        payload = {
            "image": os.path.abspath(args.image),
            "camera_json": os.path.abspath(args.camera),
            "chessboard": {"cols": args.cols, "rows": args.rows,
                           "square_mm": args.square_mm},
            "metrics": metrics,
            "reprojection_rms_px": rms,
            "rvec": rvec.reshape(-1).tolist(),
            "tvec": tvec.reshape(-1).tolist(),
            "R": R.tolist(),
            "date": datetime.now().isoformat(timespec="seconds"),
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"💾 Métricas guardadas en: {args.out_json}")

    # Overlay
    if args.show or args.save_overlay:
        overlay = draw_overlay(img_bgr, corners, args.cols, args.rows,
                               K, dist, rvec, tvec, args.square_mm)
        if args.save_overlay:
            cv2.imwrite(args.save_overlay, overlay)
            print(f"💾 Overlay guardado en: {args.save_overlay}")
        if args.show:
            # Ajustar ventana a algo manejable para 4K.
            h, w = overlay.shape[:2]
            max_w = 1600
            if w > max_w:
                scale = max_w / w
                overlay = cv2.resize(overlay, (max_w, int(h * scale)))
            cv2.imshow("pose_chessboard", overlay)
            print("Pulsa cualquier tecla en la ventana para salir.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
