# YOLOv8 道路车辆检测系统

基于YOLOv8的实时道路车辆检测和计数系统，支持多种车辆类型的识别和追踪。

## 功能特性

- ✅ 多种车辆类型检测（汽车、摩托车、公交车、卡车）
- ✅ 实时视频流检测
- ✅ 视频文件批量处理
- ✅ 车辆追踪和轨迹记录
- ✅ 双向计数（上行/下行）
- ✅ 高精度目标追踪（IOU算法）
- ✅ GPU加速支持
- ✅ 可视化结果输出

## 支持的车辆类型

| 类别ID | 名称 | 颜色标识 |
|--------|------|----------|
| 2 | car (汽车) | 蓝色 |
| 3 | motorcycle (摩托车) | 黄色 |
| 5 | bus (公交车) | 绿色 |
| 7 | truck (卡车) | 紫色 |

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 环境要求

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- Ultralytics YOLOv8

### 3. GPU支持（可选）

如需使用GPU加速，确保安装了CUDA和cuDNN：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 使用方法

### 基本检测

```python
from vehicle_detector import VehicleDetector

# 初始化检测器
detector = VehicleDetector(model_path='yolov8n.pt', conf_threshold=0.5)

# 检测图像
detector.detect_image('path/to/image.jpg', 'output.jpg')

# 检测视频
detector.detect_video('path/to/video.mp4', 'output.mp4')

# 摄像头实时检测
detector.detect_camera(camera_id=0)
```

### 带追踪的检测

```python
from vehicle_tracker import VehicleTracker

# 初始化追踪器
tracker = VehicleTracker(model_path='yolov8n.pt', conf_threshold=0.5)

# 处理视频并设置计数线
tracker.process_video(
    video_path='path/to/video.mp4',
    output_path='tracked_output.mp4',
    counting_line_y=400
)
```

### 命令行使用

```bash
# 检测图像
python vehicle_detector.py --mode image --source image.jpg --output result.jpg

# 检测视频
python vehicle_detector.py --mode video --source video.mp4 --output result.mp4

# 摄像头实时检测
python vehicle_detector.py --mode camera --source 0

# 自定义模型和阈值
python vehicle_detector.py --mode video --source video.mp4 --model yolov8s.pt --conf 0.6
```

## 模型选择

YOLOv8提供多种预训练模型，平衡速度和精度：

| 模型 | 速度 | 精度 | 推荐场景 |
|------|------|------|----------|
| yolov8n.pt | 最快 | 最低 | 实时应用、边缘设备 |
| yolov8s.pt | 快 | 中等 | 一般实时应用 |
| yolov8m.pt | 中等 | 较好 | 需要更高精度 |
| yolov8l.pt | 慢 | 高 |离线批处理 |
| yolov8x.pt | 最慢 | 最高 | 追求最高精度 |

## 项目结构

```
d:\Vehicle driving detection\
├── vehicle_detector.py    # 主要检测模块
├── vehicle_tracker.py     # 追踪和计数模块
├── examples.py           # 使用示例
├── config.yaml           # 配置文件
├── requirements.txt     # 依赖列表
└── README.md            # 项目说明
```

## 配置说明

编辑 `config.yaml` 文件自定义检测参数：

```yaml
model:
  name: yolov8n.pt              # 选择模型
  confidence_threshold: 0.5    # 置信度阈值

vehicle_classes:
  # 可自定义检测的车辆类型

tracking:
  iou_threshold: 0.3          # IOU匹配阈值
  counting_line:
    y_position: 400           # 计数线位置
```

## 性能优化建议

1. **使用GPU**: 系统会自动检测并使用GPU（如果有）
2. **选择轻量模型**: `yolov8n.pt` 速度最快
3. **降低帧率**: 每隔N帧处理一次
4. **调整阈值**: 提高置信度阈值减少检测数量
5. **减小分辨率**: 缩小输入图像尺寸

## 交互控制

在实时检测过程中：
- `q` - 退出程序
- `s` - 保存当前帧

## 注意事项

1. 首次运行会自动下载YOLOv8模型（约6MB）
2. 确保有足够的磁盘空间存储输出视频
3. 摄像头ID根据系统实际情况调整（0通常是默认摄像头）
4. 计数线的位置需要根据实际视频调整

## 常见问题

**Q: 检测不到车辆？**
- 降低置信度阈值（conf_threshold）
- 确保视频/图像质量足够
- 检查车辆是否在模型支持类别中

**Q: 追踪不稳定？**
- 调整IOU阈值（iou_threshold）
- 确保车辆没有完全遮挡
- 检查视频帧率是否过低

**Q: 计数不准确？**
- 调整计数线位置
- 确保车辆穿过计数线
- 根据实际情况调整计数线阈值

## 许可证

本项目仅供学习和研究使用。

## 参考资料

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
