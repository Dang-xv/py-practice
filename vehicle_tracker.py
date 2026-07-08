"""
高级车辆追踪和计数系统
使用IOU追踪算法实现准确的多目标追踪
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from ultralytics import YOLO
import torch


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
    """车辆追踪器 - 使用IOU匹配算法"""
    
    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.3,
        max_frames_missing: int = 30
    ):
        """
        初始化追踪器
        
        Args:
            model_path: YOLOv8模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值，用于匹配跟踪
            max_frames_missing: 最大允许丢失帧数
        """
        self.detector = YOLO(model_path)
        self.detector.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_frames_missing = max_frames_missing
        
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        self.tracks: Dict[int, VehicleTrack] = {}
        self.next_track_id = 1
        
        self.counts_by_direction = defaultdict(lambda: defaultdict(int))
        self.counts_by_class = defaultdict(int)
        self.crossed_tracks = set()
        
        self.vehicle_colors = self._generate_colors(50)
        
        self.counting_line_y = None
        self.counting_line_direction = "vertical"
    
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """生成随机颜色"""
        np.random.seed(42)
        colors = []
        for i in range(num_colors):
            color = (
                int(np.random.randint(0, 255)),
                int(np.random.randint(0, 255)),
                int(np.random.randint(0, 255))
            )
            colors.append(color)
        return colors
    
    def set_counting_line(self, y_position: int, direction: str = "horizontal"):
        """
        设置计数线
        
        Args:
            y_position: 计数线的Y坐标（垂直线）或X坐标（水平线）
            direction: 'horizontal' 或 'vertical'
        """
        self.counting_line_y = y_position
        self.counting_line_direction = direction
    
    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """计算两个边界框的IOU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def update_tracks(self, detections: List[Tuple]) -> Dict[int, VehicleTrack]:
        """
        更新轨迹
        
        Args:
            detections: 检测结果列表 [(bbox, class_id, confidence), ...]
            
        Returns:
            当前活跃的轨迹字典
        """
        if not detections:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].frames_missing += 1
                if self.tracks[track_id].frames_missing > self.max_frames_missing:
                    del self.tracks[track_id]
            return self.tracks
        
        detected_boxes = [det[0] for det in detections]
        detected_classes = [det[1] for det in detections]
        
        matched_tracks = set()
        unmatched_detections = []
        
        for track_id, track in self.tracks.items():
            if track.frames_missing > self.max_frames_missing:
                continue
            
            best_match_idx = -1
            best_match_iou = self.iou_threshold
            
            for idx, det_box in enumerate(detected_boxes):
                if idx in matched_tracks:
                    continue
                
                if track.class_name != detected_classes[idx]:
                    continue
                
                iou = self.calculate_iou(track.bbox, det_box)
                if iou > best_match_iou:
                    best_match_iou = iou
                    best_match_idx = idx
            
            if best_match_idx >= 0:
                matched_tracks.add(best_match_idx)
                track.bbox = detected_boxes[best_match_idx]
                track.center = self._get_bbox_center(detected_boxes[best_match_idx])
                track.frames_missing = 0
                
                if self.counting_line_y is not None:
                    self._check_crossing(track)
            else:
                track.frames_missing += 1
        
        for idx, det in enumerate(detections):
            if idx not in matched_tracks:
                unmatched_detections.append(det)
        
        for det in unmatched_detections:
            bbox, class_id, conf = det
            class_name = self.vehicle_classes.get(class_id, 'unknown')
            
            new_track = VehicleTrack(
                track_id=self.next_track_id,
                class_name=class_name,
                bbox=bbox,
                center=self._get_bbox_center(bbox)
            )
            
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1
        
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].frames_missing > self.max_frames_missing:
                del self.tracks[track_id]
        
        return self.tracks
    
    def _get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """获取边界框中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _check_crossing(self, track: VehicleTrack):
        """检查是否越过计数线"""
        if track.track_id in self.crossed_tracks:
            return
        
        if self.counting_line_direction == "horizontal":
            center_y = track.center[1]
            
            if self.counting_line_y is not None:
                if abs(center_y - self.counting_line_y) < 20:
                    self.crossed_tracks.add(track.track_id)
                    track.total_counted = True
                    
                    if track.direction == "unknown":
                        track.direction = "down" if track.center[1] > self.counting_line_y else "up"
                    
                    self.counts_by_direction[track.direction][track.class_name] += 1
                    self.counts_by_class[track.class_name] += 1
    
    def detect_and_track(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        检测并追踪车辆
        
        Args:
            frame: 输入帧
            
        Returns:
            标注后的帧和统计信息
        """
        results = self.detector(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    detections.append(((x1, y1, x2, y2), class_id, conf))
        
        tracks = self.update_tracks(detections)
        
        annotated_frame = self._draw_annotations(frame, tracks)
        
        stats = {
            'active_tracks': len(tracks),
            'total_counted': sum(self.counts_by_class.values()),
            'by_class': dict(self.counts_by_class),
            'by_direction': {k: dict(v) for k, v in self.counts_by_direction.items()}
        }
        
        return annotated_frame, stats
    
    def _draw_annotations(self, frame: np.ndarray, tracks: Dict[int, VehicleTrack]) -> np.ndarray:
        """绘制标注"""
        annotated = frame.copy()
        
        if self.counting_line_y is not None:
            if self.counting_line_direction == "horizontal":
                cv2.line(
                    annotated,
                    (0, self.counting_line_y),
                    (frame.shape[1], self.counting_line_y),
                    (0, 0, 255),
                    3
                )
            else:
                cv2.line(
                    annotated,
                    (self.counting_line_y, 0),
                    (self.counting_line_y, frame.shape[0]),
                    (0, 0, 255),
                    3
                )
        
        for track_id, track in tracks.items():
            x1, y1, x2, y2 = track.bbox
            color = self.vehicle_colors[track_id % len(self.vehicle_colors)]
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID:{track_id} {track.class_name}"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
            cv2.circle(annotated, track.center, 5, color, -1)
        
        self._draw_stats(annotated)
        
        return annotated
    
    def _draw_stats(self, frame: np.ndarray):
        """绘制统计信息"""
        y_offset = 30
        x_offset = 10
        
        cv2.rectangle(frame, (5, 5), (350, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (350, 200), (255, 255, 255), 2)
        
        cv2.putText(
            frame,
            "车辆统计:",
            (x_offset, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y_offset += 30
        
        for class_name, count in self.counts_by_class.items():
            cv2.putText(
                frame,
                f"{class_name}: {count}",
                (x_offset, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            y_offset += 25
        
        cv2.putText(
            frame,
            f"总计: {sum(self.counts_by_class.values())}",
            (x_offset, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_live: bool = True,
        counting_line_y: int = None
    ):
        """
        处理视频
        
        Args:
            video_path: 视频路径
            output_path: 输出路径
            show_live: 是否显示
            counting_line_y: 计数线Y坐标
        """
        if counting_line_y is not None:
            self.set_counting_line(counting_line_y, "horizontal")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("开始追踪处理...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame, stats = self.detect_and_track(frame)
            
            if out:
                out.write(annotated_frame)
            
            if show_live:
                cv2.imshow('Vehicle Tracking', annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        self.print_final_statistics()
    
    def print_final_statistics(self):
        """打印最终统计"""
        print("\n" + "=" * 60)
        print("最终统计结果:")
        print("=" * 60)
        
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
    """演示函数"""
    tracker = VehicleTracker(
        model_path='yolov8n.pt',
        conf_threshold=0.5,
        iou_threshold=0.3
    )
    
    print("请提供视频文件路径进行追踪")
    print("或者修改代码中的视频路径")


if __name__ == '__main__':
    demo()
