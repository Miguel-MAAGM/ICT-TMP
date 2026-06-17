#!/usr/bin/env python
"""
Ejecuta LaserTracker sobre una imagen estática (no cámara).
Ejemplo:
    python laser_tracker_image.py preview_capture.jpg
    python laser_tracker_image.py preview_capture.jpg -u 0 -U 10 -s 150 -v 220 -d
"""
import argparse
import sys
import cv2
from laser_tracker import LaserTracker


def main():
    parser = argparse.ArgumentParser(description='LaserTracker sobre imagen')
    parser.add_argument('image', help='Ruta de la imagen (jpg/png)')
    parser.add_argument('-u', '--huemin', default=20, type=int)
    parser.add_argument('-U', '--huemax', default=160, type=int)
    parser.add_argument('-s', '--satmin', default=100, type=int)
    parser.add_argument('-S', '--satmax', default=255, type=int)
    parser.add_argument('-v', '--valmin', default=200, type=int)
    parser.add_argument('-V', '--valmax', default=255, type=int)
    parser.add_argument('-d', '--display', action='store_true',
                        help='Mostrar ventanas de los canales HSV')
    parser.add_argument('-o', '--output', default='laser_result.jpg',
                        help='Archivo de salida con el láser marcado')
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        sys.stderr.write(f"No se pudo leer la imagen: {args.image}\n")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Imagen cargada: {w}x{h}")

    tracker = LaserTracker(
        cam_width=w, cam_height=h,
        hue_min=args.huemin, hue_max=args.huemax,
        sat_min=args.satmin, sat_max=args.satmax,
        val_min=args.valmin, val_max=args.valmax,
        display_thresholds=args.display,
    )

    hsv_image = tracker.detect(frame)

    if tracker.previous_position:
        print(f"Láser detectado en (x, y) = {tracker.previous_position}")
    else:
        print("No se detectó láser con los umbrales actuales.")

    cv2.imwrite(args.output, frame)
    print(f"Imagen anotada guardada en: {args.output}")

    cv2.imshow('Resultado', frame)
    cv2.imshow('Mascara Laser', tracker.channels['laser'])
    if args.display:
        cv2.imshow('HSV combinado', hsv_image)
        cv2.imshow('Hue',        tracker.channels['hue'])
        cv2.imshow('Saturation', tracker.channels['saturation'])
        cv2.imshow('Value',      tracker.channels['value'])

    print("Pulsa cualquier tecla en la ventana para salir.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
