"""
YOLOv8 道路车辆检测系统 V5.0
支持多种车辆类型检测和准确计数 — 高性能版

性能优化:
  - 批量 GPU→CPU 传输
  - FP16 半精度推理
  - H.264 编码器
  - 多线程采集管道
  - 自适应跳帧
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import torch
import threading
import queue
import time

# ─── 全局优化 ──────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class VehicleDetector:
    """车辆检测器类 - 高性能版"""

    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        4: 'airplane',
        5: 'bus',
        7: 'truck',
        8: 'boat',
    }

    VEHICLE_COLORS = {
        'car': (255, 0, 0),
        'motorcycle': (0, 255, 255),
        'bus': (0, 255, 0),
        'truck': (255, 0, 255),
        'airplane': (0, 128, 255),
        'boat': (128, 0, 255),
    }

    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.5,
                 use_fp16: bool = True, use_threading: bool = True, frame_skip: int = 0):
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")

        self.model = YOLO(model_path)
        self.model.to(self.device)

        # FP16
        self.use_fp16 = use_fp16 and self.device == 'cuda'
        if self.use_fp16:
            self.model.model.half()
            print("已启用 FP16 半精度推理")

        self.vehicle_classes_filter = list(self.VEHICLE_CLASSES.keys())

        # 多线程
        self.use_threading = use_threading and self.device != 'cpu'
        self._frame_skip = frame_skip
        self._adaptive_skip = (frame_skip == 0)
        self._processing_times: List[float] = []

    def detect_vehicles(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        """检测单帧中的车辆 — 批量 GPU→CPU 传输"""
        if self.use_fp16:
            results = self.model(frame, conf=self.conf_threshold, verbose=False, half=True)
        else:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)

        vehicle_counts = defaultdict(int)
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # ── 批量 GPU→CPU 传输 ──
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            xyxy_all = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for i in range(len(cls_ids)):
                class_id = cls_ids[i]
                if class_id not in self.vehicle_classes_filter:
                    continue

                class_name = self.VEHICLE_CLASSES[class_id]
                confidence = float(confs[i])
                x1, y1, x2, y2 = xyxy_all[i]

                color = self.VEHICLE_COLORS.get(class_name, (255, 255, 255))

                # 边界框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # 标签
                label = f"{class_name} {confidence:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                vehicle_counts[class_name] += 1

        vehicle_counts['total'] = sum(vehicle_counts.values())
        return annotated_frame, dict(vehicle_counts)

    def _capture_worker(self, cap, q, stop):
        """采集线程"""
        while not stop.is_set():
            ret, frame = cap.read()
            if not ret:
                q.put(None)
                break
            if q.qsize() < 2:
                q.put(frame)

    def detect_video(self, video_path: str, output_path: Optional[str] = None,
                     show_live: bool = True, save_frames: bool = False):
        """检测视频中的车辆 — 多线程管道"""
        if isinstance(video_path, str) and video_path.isdigit():
            cap = cv2.VideoCapture(int(video_path))
        else:
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"无法打开视频源: {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 输出编码器
        out = None
        if output_path:
            for codec, fourcc in [('avc1', cv2.VideoWriter_fourcc(*'avc1')),
                                   ('H264', cv2.VideoWriter_fourcc(*'H264')),
                                   ('mp4v', cv2.VideoWriter_fourcc(*'mp4v'))]:
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    print(f"输出编码器: {codec}")
                    break

        # 采集线程
        stop_event = threading.Event()
        cap_queue = queue.Queue(maxsize=4)
        if self.use_threading:
            ct = threading.Thread(target=self._capture_worker,
                                  args=(cap, cap_queue, stop_event), daemon=True)
            ct.start()

        total_frame_count = 0
        all_counts = []
        fps_frame_count = 0
        fps_start = time.perf_counter()
        current_fps = 0.0
        skip_counter = 0

        print("开始视频检测...")

        try:
            while True:
                if self.use_threading:
                    try:
                        frame = cap_queue.get(timeout=5.0)
                    except queue.Empty:
                        break
                    if frame is None:
                        break
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break

                # 自适应跳帧
                if self._adaptive_skip and self._processing_times:
                    avg_t = sum(self._processing_times[-30:]) / len(self._processing_times[-30:])
                    target_t = 1.0 / max(fps, 1)
                    skip_counter = max(1, int(avg_t / target_t)) - 1 if avg_t > target_t * 1.5 else 0
                elif self._frame_skip > 0:
                    skip_counter = self._frame_skip

                total_frame_count += 1
                if skip_counter > 0 and total_frame_count % (skip_counter + 1) != 0:
                    continue

                t0 = time.perf_counter()
                annotated_frame, counts = self.detect_vehicles(frame)
                self._processing_times.append(time.perf_counter() - t0)

                all_counts.append(counts)
                self._draw_summary(annotated_frame, counts, total_frame_count)

                # FPS
                fps_frame_count += 1
                if fps_frame_count >= 10:
                    now = time.perf_counter()
                    elapsed = now - fps_start
                    current_fps = fps_frame_count / elapsed if elapsed > 0 else 0
                    fps_frame_count = 0
                    fps_start = now

                cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if out:
                    out.write(annotated_frame)

                if show_live:
                    cv2.imshow('Vehicle Detection', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self._save_frame(annotated_frame, counts)

        finally:
            stop_event.set()
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            self._print_final_stats(all_counts)

    def detect_image(self, image_path: str, output_path: Optional[str] = None):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"无法读取图像: {image_path}")
            return
        annotated_frame, counts = self.detect_vehicles(frame)
        self._draw_summary(annotated_frame, counts, 1)
        if output_path:
            cv2.imwrite(output_path, annotated_frame)
            print(f"结果已保存到: {output_path}")
        self._display_counts(counts)
        cv2.imshow('Vehicle Detection', annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_camera(self, camera_id: int = 0, output_path: Optional[str] = None):
        self.detect_video(str(camera_id), output_path, show_live=True)

    def _draw_summary(self, frame, counts, frame_num):
        y_off, x_off = 30, 10
        cv2.rectangle(frame, (5, 5), (300, 25 + len(counts) * 25), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (300, 25 + len(counts) * 25), (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_num}", (x_off, y_off),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_off += 25
        for vt, count in counts.items():
            if vt != 'total':
                color = self.VEHICLE_COLORS.get(vt, (255, 255, 255))
                cv2.putText(frame, f"{vt}: {count}", (x_off, y_off),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y_off += 25
        cv2.putText(frame, f"Total: {counts.get('total', 0)}", (x_off, y_off),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _display_counts(self, counts):
        print("\n检测结果:\n" + "=" * 30)
        for vt, count in counts.items():
            if vt != 'total':
                print(f"{vt:15s}: {count:4d}")
        print("=" * 30 + f"\n{'总计':15s}: {counts.get('total', 0):4d}\n")

    def _save_frame(self, frame, counts):
        import time
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detected_frame_{ts}.jpg"
        cv2.imwrite(filename, frame)
        print(f"帧已保存: {filename}")

    def _print_final_stats(self, all_counts):
        if not all_counts:
            return
        total_counts = defaultdict(int)
        for counts in all_counts:
            for vt, count in counts.items():
                if vt != 'total':
                    total_counts[vt] += count
        print("\n视频处理完成!\n" + "=" * 50 + "\n总体统计:\n" + "=" * 50)
        for vt, total in sorted(total_counts.items()):
            print(f"{vt:15s}: {total:6d}")
        print("=" * 50)
        print(f"{'总检测数':15s}: {sum(total_counts.values()):6d}")
        print(f"{'总帧数':15s}: {len(all_counts):6d}\n")


def main():
    detector = VehicleDetector(model_path='yolov8n.pt', conf_threshold=0.5)
    import argparse
    parser = argparse.ArgumentParser(description='YOLOv8 道路车辆检测系统 V5.0')
    parser.add_argument('--mode', type=str, default='camera',
                       choices=['image', 'video', 'camera'])
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--no-fp16', action='store_true', help='禁用 FP16')
    parser.add_argument('--no-thread', action='store_true', help='禁用多线程')

    args = parser.parse_args()
    detector = VehicleDetector(
        model_path=args.model, conf_threshold=args.conf,
        use_fp16=not args.no_fp16, use_threading=not args.no_thread,
    )

    if args.mode == 'image':
        detector.detect_image(args.source, args.output)
    elif args.mode == 'video':
        detector.detect_video(args.source, args.output)
    elif args.mode == 'camera':
        camera_id = int(args.source) if args.source.isdigit() else 0
        detector.detect_camera(camera_id, args.output)


if __name__ == '__main__':
    main()
