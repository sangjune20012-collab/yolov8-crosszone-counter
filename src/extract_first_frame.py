import argparse
import os

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Extract the first frame from a video.")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output image path")
    return parser.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.source}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read first frame from: {args.source}")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(args.output, frame)
    print(f"Saved first frame to: {args.output}")


if __name__ == "__main__":
    main()