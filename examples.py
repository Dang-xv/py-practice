"""
YOLOv8 道路车辆检测 - 使用示例
包含过线计数和车辆去重功能
"""

from vehicle_counter import VehicleCounter
from vehicle_tracker import VehicleTracker


def example_line_crossing_counting():
    """过线计数示例"""
    print("=" * 70)
    print("示例1: 过线计数")
    print("=" * 70)
    
    counter = VehicleCounter(
        model_path='yolov8n.pt',
        conf_threshold=0.5
    )
    
    counter.add_counting_line(400, "主线")
    counter.add_counting_line(200, "副线")
    
    print("\n已设置两条计数线:")
    print("  - 主线 at Y=400: 主要计数线")
    print("  - 副线 at Y=200: 辅助计数线")
    
    counter.process_video(
        video_path='path/to/video.mp4',
        output_path='counting_output.mp4',
        counting_line_y=400
    )


def example_single_line_counting():
    """单线计数示例"""
    print("=" * 70)
    print("示例2: 单线双向计数")
    print("=" * 70)
    
    counter = VehicleCounter(model_path='yolov8n.pt')
    
    print("\n设置单条计数线，自动区分上行/下行车辆")
    counter.add_counting_line(350, "双向计数线")
    
    counter.process_video(
        video_path='path/to/video.mp4',
        counting_line_y=350
    )


def example_camera_realtime_counting():
    """摄像头实时计数示例"""
    print("=" * 70)
    print("示例3: 摄像头实时计数")
    print("=" * 70)
    
    counter = VehicleCounter(model_path='yolov8n.pt', conf_threshold=0.6)
    counter.add_counting_line(300, "摄像头计数线")
    
    print("\n启动摄像头实时计数...")
    print("  - 按 'q' 退出")
    print("  - 按 'r' 重置计数")
    print("  - 按 's' 保存截图")
    
    counter.process_video(
        video_path='0',
        counting_line_y=300
    )


def example_detection_zone():
    """检测区域示例"""
    print("=" * 70)
    print("示例4: 多区域车辆统计")
    print("=" * 70)
    
    counter = VehicleCounter(model_path='yolov8n.pt')
    
    counter.add_counting_line(400, "主线")
    
    counter.add_detection_zone(
        [(100, 200), (300, 200), (300, 350), (100, 350)],
        "区域A"
    )
    counter.add_detection_zone(
        [(400, 200), (600, 200), (600, 350), (400, 350)],
        "区域B"
    )
    
    print("\n已设置:")
    print("  - 一条计数线")
    print("  - 两个检测区域")
    
    counter.process_video(
        video_path='path/to/video.mp4',
        counting_line_y=400
    )


def example_vehicle_deduplication():
    """车辆去重示例"""
    print("=" * 70)
    print("示例5: 车辆去重说明")
    print("=" * 70)
    
    print("\n去重机制说明:")
    print("-" * 70)
    print("1. 基于IOU匹配:")
    print("   - 计算检测框与已有轨迹的交并比(IoU)")
    print("   - IoU > 0.3 认为匹配成功")
    print("   - 位置接近的同一车辆不会重复计数")
    
    print("\n2. 特征签名匹配:")
    print("   - 提取车辆的大小、宽高比等特征")
    print("   - 结合视觉特征(简化版HOG)")
    print("   - 相似特征认为是同一车辆")
    
    print("\n3. 轨迹连续性:")
    print("   - 追踪每辆车的移动轨迹")
    print("   - 连续多帧出现认为是同一车辆")
    print("   - 短暂消失(≤30帧)仍保持追踪")
    
    print("\n4. 唯一标识:")
    print("   - 每辆车分配唯一Track ID")
    print("   - 同一车辆全程使用相同ID")
    print("   - 避免重复计数")
    
    print("\n去重效果:")
    print("  ✓ 同一车辆多次检测只计数一次")
    print("  ✓ 车辆在画面中停留不会重复计数")
    print("  ✓ 遮挡后重新出现仍识别为同一车辆")
    print("  ✓ 不同车辆即使位置接近也能区分")
    
    counter = VehicleCounter(model_path='yolov8n.pt')
    counter.add_counting_line(400, "计数线")
    
    print("\n实际测试:")
    print("  运行视频，观察同一车辆的Track ID是否保持不变")


def example_multi_line_counting():
    """多线计数示例"""
    print("=" * 70)
    print("示例6: 多车道分别计数")
    print("=" * 70)
    
    counter = VehicleCounter(model_path='yolov8n.pt')
    
    counter.add_counting_line(200, "车道1")
    counter.add_counting_line(300, "车道2")
    counter.add_counting_line(400, "车道3")
    counter.add_counting_line(500, "车道4")
    
    print("\n已设置4条计数线，分别统计不同车道的车辆")
    
    counter.process_video(
        video_path='path/to/video.mp4',
        counting_line_y=200
    )


def example_batch_with_export():
    """批量处理并导出数据"""
    print("=" * 70)
    print("示例7: 批量处理并导出数据")
    print("=" * 70)
    
    import os
    from pathlib import Path
    from datetime import datetime
    
    counter = VehicleCounter(model_path='yolov8n.pt')
    counter.add_counting_line(400, "主线")
    
    input_dir = Path('videos/')
    output_dir = Path('output/')
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"\n创建示例目录: {input_dir}")
        input_dir.mkdir(exist_ok=True)
        print("请将视频文件放入 videos/ 目录")
        return
    
    video_files = list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.avi'))
    
    if not video_files:
        print("\n未找到视频文件，请将视频放入 videos/ 目录")
        return
    
    print(f"\n找到 {len(video_files)} 个视频文件")
    
    for video_file in video_files:
        print(f"\n处理: {video_file.name}")
        
        output_file = output_dir / f"counted_{video_file.name}"
        
        counter.process_video(
            str(video_file),
            str(output_file),
            show_live=False,
            counting_line_y=400
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = output_dir / f"data_{video_file.stem}_{timestamp}.json"
        counter.export_data(str(export_file))
        
        counter.reset_counts()


def example_adjust_threshold():
    """调整检测阈值示例"""
    print("=" * 70)
    print("示例8: 调整检测参数")
    print("=" * 70)
    
    print("\n参数调整建议:")
    print("-" * 70)
    print("1. 置信度阈值 (conf_threshold):")
    print("   - 范围: 0.0 - 1.0")
    print("   - 默认值: 0.5")
    print("   - 低阈值(0.3): 检测更多，可能误检")
    print("   - 高阈值(0.7): 检测更准，可能漏检")
    
    print("\n2. IOU阈值 (iou_threshold):")
    print("   - 范围: 0.0 - 1.0")
    print("   - 默认值: 0.3")
    print("   - 高阈值(0.5): 更严格的匹配")
    print("   - 低阈值(0.2): 更宽松的匹配")
    
    print("\n3. 最大丢失帧数 (max_frames_missing):")
    print("   - 范围: 整数")
    print("   - 默认值: 30")
    print("   - 较大值: 更长追踪距离，轨迹可能混乱")
    print("   - 较小值: 更严格的追踪，频繁丢失")
    
    counter = VehicleCounter(
        model_path='yolov8n.pt',
        conf_threshold=0.5,
        iou_threshold=0.3,
        max_frames_missing=30
    )
    
    counter.add_counting_line(400)
    
    print("\n使用建议的参数进行检测:")
    counter.process_video('path/to/video.mp4', counting_line_y=400)


def example_interactive_counting():
    """交互式计数示例"""
    print("=" * 70)
    print("示例9: 交互式计数演示")
    print("=" * 70)
    
    counter = VehicleCounter(model_path='yolov8n.pt')
    
    import argparse
    parser = argparse.ArgumentParser(description='车辆计数系统')
    parser.add_argument('--video', type=str, help='视频文件路径')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    parser.add_argument('--line', type=int, default=400, help='计数线Y坐标')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='模型路径')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    
    args = parser.parse_args()
    
    counter = VehicleCounter(
        model_path=args.model,
        conf_threshold=args.conf
    )
    
    counter.add_counting_line(args.line, "主计数线")
    
    source = args.video if args.video else str(args.camera)
    
    print(f"\n启动参数:")
    print(f"  视频源: {source}")
    print(f"  计数线: Y={args.line}")
    print(f"  模型: {args.model}")
    print(f"  置信度: {args.conf}")
    
    counter.process_video(
        video_path=source,
        output_path=args.output,
        counting_line_y=args.line
    )


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("YOLOv8 道路车辆检测系统 - 增强版")
    print("包含过线计数和车辆去重功能")
    print("=" * 70)
    
    examples = [
        ("过线计数", example_line_crossing_counting),
        ("单线双向计数", example_single_line_counting),
        ("摄像头实时计数", example_camera_realtime_counting),
        ("多区域统计", example_detection_zone),
        ("车辆去重说明", example_vehicle_deduplication),
        ("多车道计数", example_multi_line_counting),
        ("批量处理导出", example_batch_with_export),
        ("参数调整", example_adjust_threshold),
        ("交互式计数", example_interactive_counting),
    ]
    
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n快速开始:")
    print("  python vehicle_counter.py --video video.mp4 --line 400")
    print("  python vehicle_counter.py --camera 0 --line 300")
    
    print("\n交互控制:")
    print("  - 按 'q': 退出程序")
    print("  - 按 'r': 重置计数")
    print("  - 按 's': 保存当前截图")


if __name__ == '__main__':
    main()
