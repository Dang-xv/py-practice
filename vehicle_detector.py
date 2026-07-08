"""
YOLOv8 道路车辆检测系统
支持多种车辆类型检测和准确计数
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch


class VehicleDetector:
    """车辆检测器类"""
    
    VEHICLE_CLASSES = {
        2: 'car',           # 汽车
        3: 'motorcycle',    # 摩托车
        4: 'airplane',      # 飞机 (可能被误检)
        5: 'bus',           # 公交车
        7: 'truck',         # 卡车
        8: 'boat',          # 船 (可能误检)
    }
    
    VEHICLE_COLORS = {
        'car': (255, 0, 0),          # 蓝色
        'motorcycle': (0, 255, 255), # 黄色
        'bus': (0, 255, 0),          # 绿色
        'truck': (255, 0, 255),      # 紫色
        'airplane': (0, 128, 255),   # 橙色
        'boat': (128, 0, 255),       # 粉紫色
    }
    
    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.5):
        """
        初始化车辆检测器
        
        Args:
            model_path: YOLOv8模型路径，默认使用yolov8n.pt
            conf_threshold: 置信度阈值
        """
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        self.vehicle_classes_filter = list(self.VEHICLE_CLASSES.keys())
        
    def detect_vehicles(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        检测单帧中的车辆
        
        Args:
            frame: 输入图像帧
            
        Returns:
            带有检测结果的图像帧和各类车辆计数
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        vehicle_counts = defaultdict(int)
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                
                if class_id in self.vehicle_classes_filter:
                    class_name = self.VEHICLE_CLASSES[class_id]
                    confidence = float(box.conf[0])
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    color = self.VEHICLE_COLORS.get(class_name, (255, 255, 255))
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{class_name} {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        color,
                        -1
                    )
                    
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )
                    
                    vehicle_counts[class_name] += 1
        
        total_vehicles = sum(vehicle_counts.values())
        vehicle_counts['total'] = total_vehicles
        
        return annotated_frame, dict(vehicle_counts)
    
    def detect_video(
        self, 
        video_path: str, 
        output_path: Optional[str] = None,
        show_live: bool = True,
        save_frames: bool = False
    ):
        """
        检测视频中的车辆
        
        Args:
            video_path: 视频文件路径或摄像头ID
            output_path: 输出视频路径
            show_live: 是否实时显示
            save_frames: 是否保存帧
        """
        if video_path.isdigit():
            cap = cv2.VideoCapture(int(video_path))
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"无法打开视频源: {video_path}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_frame_count = 0
        all_counts = []
        
        print("开始视频检测...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            total_frame_count += 1
            annotated_frame, counts = self.detect_vehicles(frame)
            
            all_counts.append(counts)
            
            self._draw_summary(annotated_frame, counts, total_frame_count)
            
            if out:
                out.write(annotated_frame)
            
            if show_live:
                cv2.imshow('Vehicle Detection', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_frame(annotated_frame, counts)
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        self._print_final_stats(all_counts)
    
    def detect_image(self, image_path: str, output_path: Optional[str] = None):
        """
        检测单张图像中的车辆
        
        Args:
            image_path: 图像路径
            output_path: 输出图像路径
        """
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
        """
        从摄像头实时检测车辆
        
        Args:
            camera_id: 摄像头ID
            output_path: 输出视频路径
        """
        self.detect_video(str(camera_id), output_path, show_live=True)
    
    def _draw_summary(self, frame: np.ndarray, counts: Dict[str, int], frame_num: int):
        """在帧上绘制统计信息"""
        y_offset = 30
        x_offset = 10
        
        cv2.rectangle(frame, (5, 5), (300, 25 + len(counts) * 25), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (300, 25 + len(counts) * 25), (255, 255, 255), 2)
        
        cv2.putText(
            frame,
            f"Frame: {frame_num}",
            (x_offset, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        y_offset += 25
        
        for vehicle_type, count in counts.items():
            if vehicle_type != 'total':
                color = self.VEHICLE_COLORS.get(vehicle_type, (255, 255, 255))
                cv2.putText(
                    frame,
                    f"{vehicle_type}: {count}",
                    (x_offset, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
                y_offset += 25
        
        cv2.putText(
            frame,
            f"Total: {counts.get('total', 0)}",
            (x_offset, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    
    def _display_counts(self, counts: Dict[str, int]):
        """显示计数结果"""
        print("\n检测结果:")
        print("=" * 30)
        for vehicle_type, count in counts.items():
            if vehicle_type != 'total':
                print(f"{vehicle_type:15s}: {count:4d}")
        print("=" * 30)
        print(f"{'总计':15s}: {counts.get('total', 0):4d}")
        print()
    
    def _save_frame(self, frame: np.ndarray, counts: Dict[str, int]):
        """保存当前帧"""
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detected_frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"帧已保存: {filename}")
    
    def _print_final_stats(self, all_counts: List[Dict[str, int]]):
        """打印最终统计信息"""
        if not all_counts:
            return
        
        total_counts = defaultdict(int)
        for counts in all_counts:
            for vehicle_type, count in counts.items():
                if vehicle_type != 'total':
                    total_counts[vehicle_type] += count
        
        print("\n视频处理完成!")
        print("=" * 50)
        print("总体统计:")
        print("=" * 50)
        for vehicle_type, total in sorted(total_counts.items()):
            print(f"{vehicle_type:15s}: {total:6d}")
        print("=" * 50)
        print(f"{'总检测数':15s}: {sum(total_counts.values()):6d}")
        print(f"{'总帧数':15s}: {len(all_counts):6d}")
        print()


def main():
    """主函数 - 演示用法"""
    detector = VehicleDetector(model_path='yolov8n.pt', conf_threshold=0.5)
    
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8 道路车辆检测系统')
    parser.add_argument('--mode', type=str, default='camera',
                       choices=['image', 'video', 'camera'],
                       help='检测模式')
    parser.add_argument('--source', type=str, default='0',
                       help='输入源 (图像路径/视频路径/摄像头ID)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出路径')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLOv8模型路径')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值')
    
    args = parser.parse_args()
    
    if args.mode == 'image':
        detector.detect_image(args.source, args.output)
    elif args.mode == 'video':
        detector.detect_video(args.source, args.output)
    elif args.mode == 'camera':
        camera_id = int(args.source) if args.source.isdigit() else 0
        detector.detect_camera(camera_id, args.output)


if __name__ == '__main__':
    main()
