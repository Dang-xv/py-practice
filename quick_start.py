"""
快速启动脚本 - 一键运行车辆检测
"""

import sys
import os

def check_dependencies():
    """检查依赖是否已安装"""
    required_packages = [
        'ultralytics',
        'cv2',
        'numpy',
        'torch'
    ]
    
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
        model = YOLO('yolov8n.pt')
        print("模型已就绪")
        return True
    except Exception as e:
        print(f"模型下载失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("YOLOv8 道路车辆检测系统 - 快速启动")
    print("=" * 60)
    
    print("\n1. 检查依赖...")
    if not check_dependencies():
        return
    
    print("\n2. 检查模型...")
    if not download_model():
        return
    
    print("\n3. 启动检测系统...")
    
    from vehicle_detector import VehicleDetector
    
    detector = VehicleDetector(model_path='yolov8n.pt', conf_threshold=0.5)
    
    print("\n选择检测模式:")
    print("  1. 图像检测")
    print("  2. 视频检测")
    print("  3. 摄像头实时检测")
    print("  4. 带追踪的检测")
    print("  0. 退出")
    
    try:
        choice = input("\n请输入选项 (0-4): ").strip()
        
        if choice == '1':
            image_path = input("请输入图像路径: ").strip()
            if image_path and os.path.exists(image_path):
                output_path = input("输出路径 (可选): ").strip()
                detector.detect_image(
                    image_path,
                    output_path if output_path else None
                )
            else:
                print("图像文件不存在")
        
        elif choice == '2':
            video_path = input("请输入视频路径: ").strip()
            if video_path and os.path.exists(video_path):
                output_path = input("输出路径 (可选): ").strip()
                detector.detect_video(
                    video_path,
                    output_path if output_path else None
                )
            else:
                print("视频文件不存在")
        
        elif choice == '3':
            camera_id = input("摄像头ID (默认0): ").strip()
            camera_id = int(camera_id) if camera_id else 0
            output_path = input("输出路径 (可选): ").strip()
            detector.detect_camera(
                camera_id,
                output_path if output_path else None
            )
        
        elif choice == '4':
            from vehicle_tracker import VehicleTracker
            
            video_path = input("请输入视频路径: ").strip()
            if video_path and os.path.exists(video_path):
                counting_line = input("计数线Y坐标 (默认400): ").strip()
                counting_line = int(counting_line) if counting_line else 400
                output_path = input("输出路径 (可选): ").strip()
                
                tracker = VehicleTracker()
                tracker.process_video(
                    video_path,
                    output_path if output_path else None,
                    counting_line_y=counting_line
                )
            else:
                print("视频文件不存在")
        
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
