import argparse
import os

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 polygon zone counting")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--model", default="weights/yolov8s.pt", help="YOLO model path")
    parser.add_argument("--classes", nargs="+", default=["person"], help="Target class names")
    parser.add_argument(
        "--polygon",
        nargs="+",
        type=int,
        required=True,
        help="Polygon coordinates: x1 y1 x2 y2 x3 y3 ..."
    )
    parser.add_argument("--output-width", type=int, default=None)
    parser.add_argument("--output-height", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def get_class_ids(model, class_names):
    name_to_id = {name: class_id for class_id, name in model.names.items()}
    missing = [name for name in class_names if name not in name_to_id]

    if missing:
        raise ValueError(
            f"Unknown class names: {missing}. "
            f"Available classes: {list(name_to_id.keys())}"
        )

    return [name_to_id[name] for name in class_names]


def main():
    args = parse_args()

    if len(args.polygon) < 6 or len(args.polygon) % 2 != 0:
        raise ValueError("--polygon must contain x y pairs and at least 3 points.")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model = YOLO(args.model)
    selected_class_ids = get_class_ids(model, args.classes)

    src_info = sv.VideoInfo.from_video_path(args.source)
    src_w, src_h = src_info.resolution_wh

    out_w = args.output_width if args.output_width else src_w
    out_h = args.output_height if args.output_height else src_h

    polygon = np.array(args.polygon, dtype=np.float32).reshape(-1, 2)

    if (out_w, out_h) != (src_w, src_h):
        polygon[:, 0] *= out_w / src_w
        polygon[:, 1] *= out_h / src_h

    polygon = polygon.astype(np.int64)

    zone = sv.PolygonZone(
        polygon=polygon,
        triggering_anchors=(sv.Position.BOTTOM_CENTER,)
    )

    box_annotator = sv.BoxAnnotator(thickness=3)
    label_annotator = sv.LabelAnnotator()
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.WHITE,
        thickness=4
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, src_info.fps, (out_w, out_h))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {args.output}")

    frame_generator = sv.get_video_frames_generator(args.source)

    for frame_index, frame in enumerate(frame_generator):
        if (out_w, out_h) != (src_w, src_h):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        result = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[np.isin(detections.class_id, selected_class_ids)]

        zone_mask = zone.trigger(detections)
        detections_in_zone = detections[zone_mask]

        labels = [
            f"{model.names[int(class_id)]} {float(confidence):.2f}"
            for class_id, confidence in zip(
                detections_in_zone.class_id,
                detections_in_zone.confidence
            )
        ]

        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections_in_zone
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections_in_zone,
            labels=labels
        )
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)

        writer.write(annotated_frame)

        if frame_index % 100 == 0:
            print(f"Processed frame: {frame_index}")

    writer.release()
    print(f"Saved video to: {args.output}")


if __name__ == "__main__":
    main()