"""
YOLOv8 道路车辆检测 - 使用示例
"""

from vehicle_detector import VehicleDetector
from vehicle_tracker import VehicleTracker


def example_basic_detection():
    """基本检测示例"""
    print("=" * 60)
    print("示例1: 基本车辆检测")
    print("=" * 60)
    
    detector = VehicleDetector(model_path='yolov8n.pt', conf_threshold=0.5)
    
    print("\n检测图像中的车辆:")
    detector.detect_image('path/to/your/image.jpg', 'output_image.jpg')
    
    print("\n检测视频文件:")
    detector.detect_video('path/to/your/video.mp4', 'output_video.mp4')
    
    print("\n使用摄像头实时检测:")
    detector.detect_camera(camera_id=0)


def example_with_tracking():
    """带追踪的检测示例"""
    print("=" * 60)
    print("示例2: 带追踪的车辆检测和计数")
    print("=" * 60)
    
    tracker = VehicleTracker(
        model_path='yolov8n.pt',
        conf_threshold=0.5,
        iou_threshold=0.3
    )
    
    print("\n处理视频并设置计数线:")
    print("  - 计数线设置在Y=400像素位置")
    print("  - 当车辆越过此线时进行计数")
    
    tracker.process_video(
        video_path='path/to/your/video.mp4',
        output_path='tracked_output.mp4',
        show_live=True,
        counting_line_y=400
    )


def example_camera_detection():
    """摄像头实时检测示例"""
    print("=" * 60)
    print("示例3: 摄像头实时检测")
    print("=" * 60)
    
    detector = VehicleDetector(model_path='yolov8n.pt', conf_threshold=0.5)
    
    print("\n启动摄像头实时检测...")
    print("  - 按 'q' 键退出")
    print("  - 按 's' 键保存当前帧")
    
    detector.detect_camera(camera_id=0, output_path='camera_output.mp4')


def example_batch_processing():
    """批量处理示例"""
    print("=" * 60)
    print("示例4: 批量处理多个视频")
    print("=" * 60)
    
    import os
    from pathlib import Path
    
    detector = VehicleDetector(model_path='yolov8n.pt', conf_threshold=0.5)
    
    input_dir = Path('videos/')
    output_dir = Path('output/')
    output_dir.mkdir(exist_ok=True)
    
    video_files = list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.avi'))
    
    print(f"\n找到 {len(video_files)} 个视频文件")
    
    for video_file in video_files:
        print(f"\n处理: {video_file.name}")
        output_file = output_dir / f"detected_{video_file.name}"
        
        detector.detect_video(
            str(video_file),
            str(output_file),
            show_live=False
        )


def example_custom_classes():
    """自定义检测类别示例"""
    print("=" * 60)
    print("示例5: 自定义检测类别")
    print("=" * 60)
    
    detector = VehicleDetector(model_path='yolov8n.pt', conf_threshold=0.5)
    
    print("\n支持的车辆类型:")
    for class_id, class_name in detector.VEHICLE_CLASSES.items():
        print(f"  {class_id}: {class_name}")
    
    print("\n检测置信度阈值: 0.5")
    print("可调整范围: 0.0 - 1.0")
    print("  - 较低阈值: 检测更多目标，可能包含误检")
    print("  - 较高阈值: 检测更准确，但可能漏检")


def example_performance_optimization():
    """性能优化示例"""
    print("=" * 60)
    print("示例6: 性能优化建议")
    print("=" * 60)
    
    print("\n模型选择建议:")
    print("  - yolov8n.pt:  速度最快，精度最低")
    print("  - yolov8s.pt:  速度快，精度适中")
    print("  - yolov8m.pt:  速度适中，精度较好")
    print("  - yolov8l.pt:  速度较慢，精度高")
    print("  - yolov8x.pt:  速度最慢，精度最高")
    
    print("\n优化建议:")
    print("  1. 使用GPU加速（如果有NVIDIA显卡）")
    print("  2. 降低处理帧率（每隔N帧处理一次）")
    print("  3. 减小输入图像尺寸")
    print("  4. 提高置信度阈值减少检测数量")


def main():
    """主函数 - 运行所有示例"""
    print("\n" + "=" * 60)
    print("YOLOv8 道路车辆检测系统 - 使用示例")
    print("=" * 60)
    
    examples = [
        ("基本检测", example_basic_detection),
        ("带追踪的检测", example_with_tracking),
        ("摄像头实时检测", example_camera_detection),
        ("批量处理", example_batch_processing),
        ("自定义类别", example_custom_classes),
        ("性能优化", example_performance_optimization),
    ]
    
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n请选择一个示例运行，或直接参考代码修改使用")
    print("\n快速开始:")
    print("  1. 安装依赖: pip install -r requirements.txt")
    print("  2. 下载YOLOv8模型（自动下载）")
    print("  3. 运行示例或修改代码")
    
    print("\n命令行使用:")
    print("  检测图像: python vehicle_detector.py --mode image --source image.jpg")
    print("  检测视频: python vehicle_detector.py --mode video --source video.mp4")
    print("  摄像头:   python vehicle_detector.py --mode camera --source 0")


if __name__ == '__main__':
    main()
