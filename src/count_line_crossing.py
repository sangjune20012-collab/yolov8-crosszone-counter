import argparse
import os

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 line crossing counting")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--model", default="weights/yolov8s.pt", help="YOLO model path")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["car", "motorcycle", "bus", "truck"],
        help="Target class names"
    )
    parser.add_argument(
        "--line",
        nargs=4,
        type=int,
        required=True,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Line coordinates"
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


def annotate_line(line_annotator, frame, line_zone):
    try:
        return line_annotator.annotate(scene=frame, line_counter=line_zone)
    except TypeError:
        return line_annotator.annotate(scene=frame, line_zone=line_zone)


def main():
    args = parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model = YOLO(args.model)
    selected_class_ids = get_class_ids(model, args.classes)

    src_info = sv.VideoInfo.from_video_path(args.source)
    src_w, src_h = src_info.resolution_wh

    out_w = args.output_width if args.output_width else src_w
    out_h = args.output_height if args.output_height else src_h

    x1, y1, x2, y2 = args.line

    if (out_w, out_h) != (src_w, src_h):
        x1 = int(x1 * out_w / src_w)
        x2 = int(x2 * out_w / src_w)
        y1 = int(y1 * out_h / src_h)
        y2 = int(y2 * out_h / src_h)

    line_zone = sv.LineZone(
        start=sv.Point(x1, y1),
        end=sv.Point(x2, y2)
    )

    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=int(src_info.fps),
        minimum_consecutive_frames=3
    )
    byte_tracker.reset()

    box_annotator = sv.BoxAnnotator(thickness=3)
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator(thickness=3, trace_length=50)
    line_annotator = sv.LineZoneAnnotator(
        thickness=4,
        text_thickness=2,
        text_scale=1.2
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
        detections = byte_tracker.update_with_detections(detections)

        line_zone.trigger(detections)

        labels = [
            f"#{tracker_id} {model.names[int(class_id)]} {float(confidence):.2f}"
            for tracker_id, class_id, confidence in zip(
                detections.tracker_id,
                detections.class_id,
                detections.confidence
            )
        ]

        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        annotated_frame = annotate_line(line_annotator, annotated_frame, line_zone)

        writer.write(annotated_frame)

        if frame_index % 100 == 0:
            print(f"Processed frame: {frame_index}")

    writer.release()
    print(f"Saved video to: {args.output}")


if __name__ == "__main__":
    main()