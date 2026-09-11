"""
快速启动脚本 - 一键运行车辆检测 V5.0
高性能版: FP16推理 + 多线程采集 + 纹理缓存 + 自适应跳帧
"""

import sys
import os

sys.stdout.reconfigure(encoding='utf-8')


def check_dependencies():
    """检查依赖是否已安装"""
    required_packages = ['ultralytics', 'cv2', 'numpy', 'torch']
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        print("缺少必要的依赖包:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\n请运行以下命令安装:")
        print("  pip install -r requirements.txt")
        return False
    return True


def download_model():
    """确保模型文件存在"""
    from ultralytics import YOLO
    print("检查YOLOv8模型...")
    try:
        _ = YOLO('yolov8n.pt')
        print("模型已就绪")
        return True
    except Exception as e:
        print(f"模型下载失败: {e}")
        return False


def get_bool_input(prompt: str, default: bool = True) -> bool:
    """获取布尔输入"""
    default_str = "Y" if default else "n"
    choice = input(f"{prompt} (Y/n, 默认={default_str}): ").strip().lower()
    if not choice:
        return default
    return choice in ('y', 'yes', 'true', '1')


def main():
    """主函数"""
    print("=" * 70)
    print("YOLOv8 道路车辆检测系统 V5.0 - 高性能版")
    print("FP16推理 | 多线程采集 | 纹理缓存 | 自适应跳帧")
    print("=" * 70)

    print("\n1. 检查依赖...")
    if not check_dependencies():
        return

    print("\n2. 检查模型...")
    if not download_model():
        return

    print("\n3. 性能配置...")
    use_fp16 = get_bool_input("启用 FP16 半精度推理 (GPU加速约2x)", True)
    use_threading = get_bool_input("启用多线程采集管道", True)

    frame_skip_str = input("跳帧策略 (0=自适应, 1=每帧, 2=隔一帧, 默认=0): ").strip()
    try:
        frame_skip = int(frame_skip_str) if frame_skip_str else 0
    except ValueError:
        frame_skip = 0

    print(f"\n配置: FP16={'开' if use_fp16 else '关'}, "
          f"线程={'开' if use_threading else '关'}, "
          f"跳帧={frame_skip}")

    print("\n4. 启动检测系统...")

    from vehicle_counter import VehicleCounter

    detector = VehicleCounter(
        model_path='yolov8n.pt',
        conf_threshold=0.5,
        use_fp16=use_fp16,
        use_threading=use_threading,
        frame_skip=frame_skip,
    )

    print("\n选择检测模式:")
    print("  1. 图片检测")
    print("  2. 视频文件检测")
    print("  3. 摄像头实时检测")
    print("  0. 退出")

    try:
        choice = input("\n请输入选项 (0-3): ").strip()

        if choice == '1':
            image_path = input("请输入图片路径: ").strip()
            if image_path and os.path.exists(image_path):
                output_path = input("输出路径 (可选，直接回车跳过): ").strip()
                annotated_frame, stats = detector.detect_image(
                    image_path,
                    output_path if output_path else None
                )
                print("\n检测结果:")
                print(f"总检测数: {stats['by_class'].get('总计', 0)}")
                for class_name, count in stats['by_class'].items():
                    if class_name != '总计':
                        print(f"  {class_name}: {count}")
                print("\n按任意键关闭显示窗口...")
                import cv2
                cv2.imshow('Detection Result', annotated_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("图片文件不存在")

        elif choice == '2':
            video_path = input("请输入视频路径: ").strip()
            if video_path and os.path.exists(video_path):
                line_y = input("计数线Y坐标 (默认400): ").strip()
                line_y = int(line_y) if line_y else 400
                output_path = input("输出路径 (可选，直接回车跳过): ").strip()
                detector.process_video(
                    video_path,
                    output_path if output_path else None,
                    counting_line_y=line_y
                )
            else:
                print("视频文件不存在")

        elif choice == '3':
            camera_id = input("摄像头ID (默认0): ").strip()
            camera_id = int(camera_id) if camera_id else 0
            line_y = input("计数线Y坐标 (默认400): ").strip()
            line_y = int(line_y) if line_y else 400
            output_path = input("输出路径 (可选，直接回车跳过): ").strip()
            detector.process_video(
                str(camera_id),
                output_path if output_path else None,
                counting_line_y=line_y
            )

        elif choice == '0':
            print("退出程序")

        else:
            print("无效的选项")

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
