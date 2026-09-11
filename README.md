# YOLOv8 道路车辆检测系统

基于YOLOv8的实时道路车辆检测和计数系统，支持多种车辆类型的识别、追踪、过线计数和车辆去重。

## 功能特性

- ✅ **多种车辆类型检测**（汽车、摩托车、公交车、卡车）
- ✅ **实时视频流检测**
- ✅ **视频文件批量处理**
- ✅ **车辆追踪和轨迹记录**
- ✅ **过线计数**（上行/下行分别统计）
- ✅ **车辆去重**（避免重复计数）
- ✅ **多线计数**（支持多条计数线）
- ✅ **多区域统计**（自定义检测区域）
- ✅ **数据导出**（JSON格式）
- ✅ **GPU加速支持**
- ✅ **可视化结果输出**

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

## 核心功能详解

### 🚦 过线计数功能

系统支持在视频中设置虚拟计数线，当车辆穿越该线时自动进行计数。

**功能特点：**
- ✅ 单线双向计数（自动区分上行和下行）
- ✅ 多线独立计数（支持多车道）
- ✅ 实时显示计数结果
- ✅ 可视化计数线

**使用示例：**

```python
from vehicle_counter import VehicleCounter

counter = VehicleCounter(model_path='yolov8n.pt', conf_threshold=0.5)

counter.add_counting_line(400, "主线")

counter.process_video('video.mp4', counting_line_y=400)
```

### 🔄 车辆去重功能

系统实现了多层去重机制，确保每辆车只被计数一次。

**去重机制：**

1. **基于IOU匹配**
   - 计算检测框与已有轨迹的交并比(IoU)
   - IoU > 0.3 认为匹配成功
   - 位置接近的同一车辆不会重复计数

2. **特征签名匹配**
   - 提取车辆的大小、宽高比等特征
   - 结合视觉特征(简化版HOG)
   - 相似特征认为是同一车辆

3. **轨迹连续性**
   - 追踪每辆车的移动轨迹
   - 连续多帧出现认为是同一车辆
   - 短暂消失(≤30帧)仍保持追踪

4. **唯一标识**
   - 每辆车分配唯一Track ID
   - 同一车辆全程使用相同ID
   - 避免重复计数

**去重效果：**
- ✓ 同一车辆多次检测只计数一次
- ✓ 车辆在画面中停留不会重复计数
- ✓ 遮挡后重新出现仍识别为同一车辆
- ✓ 不同车辆即使位置接近也能区分

### 📊 多区域统计

支持自定义检测区域，统计进入该区域的车辆数量。

```python
counter.add_detection_zone(
    [(100, 200), (300, 200), (300, 350), (100, 350)],
    "区域A"
)
```

## 使用方法

### 基本检测

```python
from vehicle_counter import VehicleCounter

counter = VehicleCounter(model_path='yolov8n.pt', conf_threshold=0.5)

counter.add_counting_line(400, "主线")

counter.process_video('video.mp4', output='output.mp4', counting_line_y=400)
```

### 摄像头实时检测

```python
counter.process_video('0', counting_line_y=300)
```

### 命令行使用

```bash
python vehicle_counter.py --video video.mp4 --line 400

python vehicle_counter.py --camera 0 --line 300

python vehicle_counter.py --video video.mp4 --line 400 --conf 0.6 --output result.mp4
```

### 数据导出

```python
counter.export_data('counting_data.json')
```

导出的JSON格式：

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "frame_count": 1000,
  "statistics": {
    "by_line": {
      "主线": {
        "up": {"car": 10, "truck": 5},
        "down": {"car": 8, "bus": 3}
      }
    }
  },
  "lines": [...]
}
```

## 模型选择

YOLOv8提供多种预训练模型，平衡速度和精度：

| 模型 | 速度 | 精度 | 推荐场景 |
|------|------|------|----------|
| yolov8n.pt | 最快 | 最低 | 实时应用、边缘设备 |
| yolov8s.pt | 快 | 中等 | 一般实时应用 |
| yolov8m.pt | 中等 | 较好 | 需要更高精度 |
| yolov8l.pt | 慢 | 高 | 离线批处理 |
| yolov8x.pt | 最慢 | 最高 | 追求最高精度 |

## 项目结构

```
d:\Vehicle driving detection\
├── vehicle_detector.py    # 基础检测模块
├── vehicle_tracker.py     # 追踪模块
├── vehicle_counter.py     # 计数和去重模块 (核心)
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

tracking:
  iou_threshold: 0.3          # IOU匹配阈值
  max_frames_missing: 30      # 最大丢失帧数

counting:
  counting_line:
    y_position: 400           # 计数线位置
```

## 交互控制

在实时检测过程中：
- `q` - 退出程序
- `r` - 重置所有计数
- `s` - 保存当前截图

## 参数调优建议

### 置信度阈值 (conf_threshold)
- **范围**: 0.0 - 1.0
- **默认值**: 0.5
- **低阈值(0.3)**: 检测更多目标，可能包含误检
- **高阈值(0.7)**: 检测更准确，但可能漏检

### IOU阈值 (iou_threshold)
- **范围**: 0.0 - 1.0
- **默认值**: 0.3
- **高阈值(0.5)**: 更严格的匹配
- **低阈值(0.2)**: 更宽松的匹配

### 最大丢失帧数 (max_frames_missing)
- **范围**: 整数
- **默认值**: 30
- **较大值**: 更长追踪距离，可能轨迹混乱
- **较小值**: 更严格的追踪，可能频繁丢失目标

## 性能优化建议

1. **使用GPU**: 系统会自动检测并使用GPU（如果有）
2. **选择轻量模型**: `yolov8n.pt` 速度最快
3. **降低帧率**: 每隔N帧处理一次
4. **调整阈值**: 提高置信度阈值减少检测数量
5. **减小分辨率**: 缩小输入图像尺寸

## 计数准确性说明

### 为什么需要去重？
- 同一车辆在视频中可能出现多次（进出画面）
- 目标检测模型可能在连续帧中多次检测到同一车辆
- 车辆可能被部分遮挡后又出现

### 去重如何工作？
1. **第一层防护**: 基于IOU的位置匹配
2. **第二层防护**: 特征签名相似度
3. **第三层防护**: 轨迹连续性验证
4. **第四层防护**: 唯一Track ID标识

### 计数不准确的可能原因
- 车辆被严重遮挡
- 车辆行驶方向与计数线不垂直
- 多个车辆紧密相邻
- 视频质量差或光照变化大

## 常见问题

**Q: 检测不到车辆？**
- 降低置信度阈值（conf_threshold）
- 确保视频/图像质量足够
- 检查车辆是否在模型支持类别中

**Q: 计数重复？**
- 确保车辆去重功能已启用
- 调整IOU阈值
- 检查计数线位置是否合理

**Q: 追踪不稳定？**
- 调整IOU阈值（iou_threshold）
- 确保车辆没有完全遮挡
- 检查视频帧率是否过低

**Q: 计数不准确？**
- 调整计数线位置
- 确保车辆穿过计数线
- 根据实际情况调整计数线阈值

## 进阶功能

### 自定义检测区域

```python
counter.add_detection_zone(
    [(100, 200), (300, 200), (300, 350), (100, 350)],
    "入口区域"
)
```

### 多线计数

```python
counter.add_counting_line(200, "车道1")
counter.add_counting_line(300, "车道2")
counter.add_counting_line(400, "车道3")
```

### 批量处理

```python
counter.process_video('video1.mp4')
counter.export_data('video1.json')
counter.reset_counts()

counter.process_video('video2.mp4')
counter.export_data('video2.json')
```

## 许可证

本项目仅供学习和研究使用。

## 参考资料

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
