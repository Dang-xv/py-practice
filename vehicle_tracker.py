"""
高级车辆追踪和计数系统 V5.0
使用IOU追踪算法实现准确的多目标追踪 — 高性能版

性能优化:
  - 批量 GPU→CPU 传输
  - 向量化 IOU 匹配
  - FP16 半精度推理
  - 多线程采集管道
  - 自适应跳帧
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from ultralytics import YOLO
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


@dataclass
class VehicleTrack:
    """车辆轨迹类"""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    frames_missing: int = 0
    total_counted: bool = False
    direction: str = "unknown"
    speed: float = 0.0


class VehicleTracker:
    """车辆追踪器 - 高性能版"""

    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.5,
                 iou_threshold: float = 0.3, max_frames_missing: int = 30,
                 use_fp16: bool = True, use_threading: bool = True, frame_skip: int = 0):
        self.detector = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector.to(self.device)

        self.use_fp16 = use_fp16 and self.device == 'cuda'
        if self.use_fp16:
            self.detector.model.half()
            print("已启用 FP16 半精度推理")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_frames_missing = max_frames_missing

        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

        self.tracks: Dict[int, VehicleTrack] = {}
        self.next_track_id = 1
        self.counts_by_direction = defaultdict(lambda: defaultdict(int))
        self.counts_by_class = defaultdict(int)
        self.crossed_tracks = set()

        # 颜色池
        rng = np.random.RandomState(42)
        self.vehicle_colors = [
            (int(rng.randint(0, 255)), int(rng.randint(0, 255)), int(rng.randint(0, 255)))
            for _ in range(50)
        ]

        self.counting_line_y = None
        self.counting_line_direction = "vertical"

        # 多线程
        self.use_threading = use_threading and self.device != 'cpu'
        self._frame_skip = frame_skip
        self._adaptive_skip = (frame_skip == 0)
        self._processing_times: List[float] = []

    def set_counting_line(self, y_position: int, direction: str = "horizontal"):
        self.counting_line_y = y_position
        self.counting_line_direction = direction

    # ─── 向量化 IOU ─────────────────────────────────────────

    @staticmethod
    def _box_iou_batch(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """向量化 IOU 计算"""
        xi1 = np.maximum(box[0], boxes[:, 0])
        yi1 = np.maximum(box[1], boxes[:, 1])
        xi2 = np.minimum(box[2], boxes[:, 2])
        yi2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
        area_a = (box[2] - box[0]) * (box[3] - box[1])
        area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_a + area_b - inter
        return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)

    @staticmethod
    def _get_bbox_center(bbox):
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    # ─── 轨迹更新 ───────────────────────────────────────────

    def update_tracks(self, detections: List[Tuple]) -> Dict[int, VehicleTrack]:
        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid].frames_missing += 1
                if self.tracks[tid].frames_missing > self.max_frames_missing:
                    del self.tracks[tid]
            return self.tracks

        det_boxes = np.array([d[0] for d in detections], dtype=np.int32)
        det_class_ids = np.array([d[1] for d in detections], dtype=np.int32)
        matched_dets = set()

        for tid, track in self.tracks.items():
            if track.class_name not in [self.vehicle_classes.get(c, '') for c in det_class_ids]:
                track.frames_missing += 1
                continue

            # 相同类别的候选
            same_mask = np.array([
                self.vehicle_classes.get(det_class_ids[j], '') == track.class_name
                for j in range(len(detections))
            ])
            available = [j for j in range(len(detections))
                         if same_mask[j] and j not in matched_dets]
            if not available:
                track.frames_missing += 1
                continue

            cand_boxes = det_boxes[available]
            ious = self._box_iou_batch(np.array(track.bbox, dtype=np.int32), cand_boxes)
            best_local = np.argmax(ious)

            if ious[best_local] >= self.iou_threshold:
                best_idx = available[best_local]
                matched_dets.add(best_idx)
                track.bbox = tuple(det_boxes[best_idx].tolist())
                track.center = self._get_bbox_center(track.bbox)
                track.frames_missing = 0
                if self.counting_line_y is not None:
                    self._check_crossing(track)
            else:
                track.frames_missing += 1

        # 创建新轨迹
        for idx, det in enumerate(detections):
            if idx not in matched_dets:
                bbox, class_id, conf = det
                class_name = self.vehicle_classes.get(class_id, 'unknown')
                self.tracks[self.next_track_id] = VehicleTrack(
                    track_id=self.next_track_id, class_name=class_name,
                    bbox=bbox, center=self._get_bbox_center(bbox),
                )
                self.next_track_id += 1

        # 清理
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].frames_missing > self.max_frames_missing:
                del self.tracks[tid]

        return self.tracks

    def _check_crossing(self, track: VehicleTrack):
        if track.track_id in self.crossed_tracks:
            return
        if self.counting_line_direction == "horizontal":
            center_y = track.center[1]
            if self.counting_line_y is not None and abs(center_y - self.counting_line_y) < 20:
                self.crossed_tracks.add(track.track_id)
                track.total_counted = True
                if track.direction == "unknown":
                    track.direction = "down" if track.center[1] > self.counting_line_y else "up"
                self.counts_by_direction[track.direction][track.class_name] += 1
                self.counts_by_class[track.class_name] += 1

    # ─── 检测 + 追踪 ────────────────────────────────────────

    def detect_and_track(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if self.use_fp16:
            results = self.detector(frame, conf=self.conf_threshold, verbose=False, half=True)
        else:
            results = self.detector(frame, conf=self.conf_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # 批量 GPU→CPU
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            xyxy_all = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for i in range(len(cls_ids)):
                class_id = cls_ids[i]
                if class_id in self.vehicle_classes:
                    detections.append((
                        (xyxy_all[i, 0], xyxy_all[i, 1], xyxy_all[i, 2], xyxy_all[i, 3]),
                        class_id, float(confs[i]),
                    ))

        tracks = self.update_tracks(detections)
        annotated_frame = self._draw_annotations(frame, tracks)
        stats = {
            'active_tracks': len(tracks),
            'total_counted': sum(self.counts_by_class.values()),
            'by_class': dict(self.counts_by_class),
            'by_direction': {k: dict(v) for k, v in self.counts_by_direction.items()},
        }
        return annotated_frame, stats

    # ─── 绘制 ───────────────────────────────────────────────

    def _draw_annotations(self, frame: np.ndarray, tracks: Dict[int, VehicleTrack]) -> np.ndarray:
        annotated = frame  # 避免额外拷贝
        fh, fw = frame.shape[:2]

        if self.counting_line_y is not None:
            if self.counting_line_direction == "horizontal":
                cv2.line(annotated, (0, self.counting_line_y),
                        (fw, self.counting_line_y), (0, 0, 255), 3)
            else:
                cv2.line(annotated, (self.counting_line_y, 0),
                        (self.counting_line_y, fh), (0, 0, 255), 3)

        for tid, track in tracks.items():
            x1, y1, x2, y2 = track.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            color = self.vehicle_colors[tid % len(self.vehicle_colors)]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"ID:{tid} {track.class_name}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(annotated, track.center, 4, color, -1)

        self._draw_stats(annotated)
        return annotated

    def _draw_stats(self, frame: np.ndarray):
        y_off, x_off = 30, 10
        cv2.rectangle(frame, (5, 5), (350, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (350, 200), (255, 255, 255), 2)
        cv2.putText(frame, "Vehicle Stats:", (x_off, y_off),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_off += 30
        for class_name, count in self.counts_by_class.items():
            cv2.putText(frame, f"{class_name}: {count}", (x_off, y_off),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_off += 25
        cv2.putText(frame, f"Total: {sum(self.counts_by_class.values())}",
                   (x_off, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ─── 视频处理 ───────────────────────────────────────────

    def _capture_worker(self, cap, q, stop):
        while not stop.is_set():
            ret, frame = cap.read()
            if not ret:
                q.put(None)
                break
            if q.qsize() < 2:
                q.put(frame)

    def process_video(self, video_path: str, output_path: Optional[str] = None,
                      show_live: bool = True, counting_line_y: int = None):
        if counting_line_y is not None:
            self.set_counting_line(counting_line_y, "horizontal")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        out = None
        if output_path:
            for codec, fourcc in [('avc1', cv2.VideoWriter_fourcc(*'avc1')),
                                   ('H264', cv2.VideoWriter_fourcc(*'H264')),
                                   ('mp4v', cv2.VideoWriter_fourcc(*'mp4v'))]:
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    print(f"输出编码器: {codec}")
                    break

        stop_event = threading.Event()
        cap_queue = queue.Queue(maxsize=4)
        if self.use_threading:
            ct = threading.Thread(target=self._capture_worker,
                                  args=(cap, cap_queue, stop_event), daemon=True)
            ct.start()

        print("开始追踪处理...")
        fps_frame_count = 0
        fps_start = time.perf_counter()
        current_fps = 0.0
        skip_counter = 0
        total_frame_count = 0

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

                total_frame_count += 1
                if self._adaptive_skip and self._processing_times:
                    avg_t = sum(self._processing_times[-30:]) / len(self._processing_times[-30:])
                    target_t = 1.0 / max(fps, 1)
                    skip_counter = max(1, int(avg_t / target_t)) - 1 if avg_t > target_t * 1.5 else 0
                elif self._frame_skip > 0:
                    skip_counter = self._frame_skip
                if skip_counter > 0 and total_frame_count % (skip_counter + 1) != 0:
                    continue

                t0 = time.perf_counter()
                annotated_frame, stats = self.detect_and_track(frame)
                self._processing_times.append(time.perf_counter() - t0)

                fps_frame_count += 1
                if fps_frame_count >= 10:
                    now = time.perf_counter()
                    e = now - fps_start
                    current_fps = fps_frame_count / e if e > 0 else 0
                    fps_frame_count = 0
                    fps_start = now

                cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}",
                           (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if out:
                    out.write(annotated_frame)

                if show_live:
                    cv2.imshow('Vehicle Tracking', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            stop_event.set()
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            self.print_final_statistics()

    def print_final_statistics(self):
        print("\n" + "=" * 60 + "\n最终统计结果:\n" + "=" * 60)
        print("\n按车辆类型统计:")
        for class_name, count in sorted(self.counts_by_class.items()):
            print(f"  {class_name:15s}: {count:6d}")
        print(f"  {'总计':15s}: {sum(self.counts_by_class.values()):6d}")
        print("\n按方向统计:")
        for direction, class_counts in self.counts_by_direction.items():
            print(f"  方向 {direction}:")
            for class_name, count in class_counts.items():
                print(f"    {class_name:15s}: {count:6d}")
        print("=" * 60)


def demo():
    tracker = VehicleTracker(model_path='yolov8n.pt', conf_threshold=0.5, iou_threshold=0.3)
    video_path = r'D:\Vehicle driving detection\cap\17s.mp4'
    tracker.process_video(video_path, output_path='output')


if __name__ == '__main__':
    demo()
