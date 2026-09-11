"""
车辆计数系统 V5.0 - 高性能版
支持车辆检测、追踪、过线计数和可调节窗口

性能优化:
  1. 批量 GPU→CPU 传输 (减少同步开销)
  2. PIL 中文纹理缓存 (避免热路径中 BGR↔RGB 转换)
  3. FP16 半精度推理 (GPU 加速 ~2x)
  4. 多线程采集管道 (采集/推理并行)
  5. H.264 硬件编码 (替换 mp4v)
  6. 自适应跳帧 (防止积压)
  7. 减少轨迹历史长度 (20→8)
  8. 向量化 IOU 匹配
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw, ImageFont
import threading
import queue
import time


# ─── 全局优化设置 ───────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # 启用 TF32 (Ampere+ GPU)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ─── 中文纹理缓存 (避免热路径 PIL 转换) ──────────────────────
class ChineseTextCache:
    """预渲染中文文本为 numpy 纹理，热路径中零 PIL 开销"""

    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}
        self._font = None
        self._font_size = 18
        self._load_font()

    def _load_font(self):
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
        ]
        for fp in font_paths:
            try:
                self._font = ImageFont.truetype(fp, self._font_size)
                return
            except (IOError, OSError):
                continue
        self._font = ImageFont.load_default()

    def get_texture(self, text: str, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """返回 RGBA numpy 纹理，直接可叠加到 BGR 帧"""
        key = (text, color)
        if key in self._cache:
            return self._cache[key]

        # 测量文本大小
        dummy_img = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text, font=self._font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 渲染到 RGBA 图像
        img = Image.new('RGBA', (tw + 4, th + 4), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((2, 2), text, font=self._font, fill=color + (255,))

        texture = np.array(img)  # RGBA
        self._cache[key] = texture
        return texture


# 全局单例
_text_cache = ChineseTextCache()


def overlay_texture(bgr_frame: np.ndarray, texture: np.ndarray, x: int, y: int) -> None:
    """将 RGBA 纹理叠加到 BGR 帧上 (in-place)"""
    th, tw = texture.shape[:2]
    fh, fw = bgr_frame.shape[:2]

    # 裁剪
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + tw), min(fh, y + th)
    tx1, ty1 = x1 - x, y1 - y
    tx2, ty2 = x2 - x, y2 - y

    if x2 <= x1 or y2 <= y1:
        return

    roi = bgr_frame[y1:y2, x1:x2]
    tex_roi = texture[ty1:ty2, tx1:tx2]
    alpha = tex_roi[:, :, 3:4] / 255.0
    # 注意: 纹理是 RGB, 需要转换为 BGR
    rgb = tex_roi[:, :, :3]
    bgr = rgb[:, :, ::-1]
    roi[:] = (alpha * bgr + (1 - alpha) * roi).astype(np.uint8)


def put_chinese_text_fast(bgr_frame: np.ndarray, text: str, position: Tuple[int, int],
                          color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    """高性能中文文本绘制 — 使用预渲染缓存"""
    texture = _text_cache.get_texture(text, color)
    overlay_texture(bgr_frame, texture, position[0], position[1])


# ─── 数据结构 ────────────────────────────────────────────────

class VehicleTrack:
    """车辆轨迹类"""
    __slots__ = ('track_id', 'class_name', 'class_id', 'bbox', 'center',
                 'history', 'first_seen_frame', 'last_seen_frame',
                 'frames_missing', 'prev_center', 'has_crossed')

    def __init__(self, track_id: int, class_name: str, class_id: int,
                 bbox: Tuple[int, int, int, int], center: Tuple[int, int]):
        self.track_id = track_id
        self.class_name = class_name
        self.class_id = class_id
        self.bbox = bbox
        self.center = center
        self.history: List[Tuple[int, int]] = [center]
        self.first_seen_frame = 0
        self.last_seen_frame = 0
        self.frames_missing = 0
        self.prev_center = None
        self.has_crossed = False


class CountingLine:
    """计数线类"""
    def __init__(self, y: int, name: str = "主线"):
        self.y = y
        self.name = name
        self.counted_vehicles: Set[int] = set()
        self.count_up: Dict[str, int] = defaultdict(int)
        self.count_down: Dict[str, int] = defaultdict(int)

    def mark_counted(self, track_id: int):
        self.counted_vehicles.add(track_id)

    def is_counted(self, track_id: int) -> bool:
        return track_id in self.counted_vehicles

    def get_total(self) -> int:
        return sum(self.count_up.values()) + sum(self.count_down.values())


# ─── 主计数器 ────────────────────────────────────────────────

class VehicleCounter:
    """车辆计数器 - 高性能版 V5.0"""

    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        conf_threshold: float = 0.5,
        max_frames_missing: int = 30,
        min_cross_distance: int = 5,
        window_title: str = 'Vehicle Counting System',
        use_fp16: bool = True,
        use_threading: bool = True,
        frame_skip: int = 0,  # 0=自适应, 1=处理每帧, 2=隔一帧, 3=隔两帧...
    ):
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        # FP16 半精度加速
        self.use_fp16 = use_fp16 and self.device == 'cuda'
        if self.use_fp16:
            self.model.model.half()
            print("已启用 FP16 半精度推理")

        print(f"车辆计数器已初始化，使用设备: {self.device}")

        self.conf_threshold = conf_threshold
        self.max_frames_missing = max_frames_missing
        self.min_cross_distance = min_cross_distance

        self.vehicle_classes = {
            2: ('car', '汽车'),
            3: ('motorcycle', '摩托车'),
            5: ('bus', '公交车'),
            7: ('truck', '卡车'),
        }

        self.vehicle_colors = {
            2: (255, 0, 0),
            3: (0, 255, 255),
            5: (0, 255, 0),
            7: (255, 0, 255),
        }

        self.vehicle_names_cn = {
            'car': '汽车',
            'motorcycle': '摩托车',
            'bus': '公交车',
            'truck': '卡车',
        }

        self.tracks: Dict[int, VehicleTrack] = {}
        self.next_track_id = 1
        self.counting_lines: List[CountingLine] = []
        self.frame_count = 0

        # 颜色池 - 预分配
        self.track_colors = np.random.RandomState(42).randint(50, 255, (1000, 3))
        self.track_colors = [tuple(c.tolist()) for c in self.track_colors]

        self.window_title = window_title
        self.current_scale = 1.0
        self.min_scale = 0.2
        self.max_scale = 3.0

        # 多线程
        self.use_threading = use_threading and self.device != 'cpu'
        self._capture_queue = None
        self._capture_thread = None

        # 自适应跳帧
        self._frame_skip = frame_skip
        self._adaptive_skip = (frame_skip == 0)
        self._processing_times: List[float] = []

    # ─── 计数线管理 ─────────────────────────────────────────

    def add_counting_line(self, y: int, name: str = "主线") -> CountingLine:
        line = CountingLine(y, name)
        self.counting_lines.append(line)
        print(f"已添加计数线: {name} at Y={y}")
        return line

    def reset_counts(self):
        for line in self.counting_lines:
            line.counted_vehicles.clear()
            line.count_up.clear()
            line.count_down.clear()
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0
        print("计数已重置")

    # ─── 向量化 IOU 计算 ────────────────────────────────────

    @staticmethod
    def _bbox_iou_batch(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """向量化 IOU: 计算单个框与多个框的 IOU"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter
        return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)

    # ─── 过线检测 ───────────────────────────────────────────

    def check_crossing(self, track: VehicleTrack):
        if track.prev_center is None or track.has_crossed:
            return

        prev_y = track.prev_center[1]
        curr_y = track.center[1]

        for line in self.counting_lines:
            if track.track_id in line.counted_vehicles:
                continue

            ly = line.y
            crossed = False
            direction = None

            if prev_y < ly <= curr_y:
                crossed = True
                direction = "down"
            elif prev_y > ly >= curr_y:
                crossed = True
                direction = "up"

            if crossed and direction:
                if abs(curr_y - prev_y) >= self.min_cross_distance:
                    line.mark_counted(track.track_id)
                    if direction == "up":
                        line.count_up[track.class_name] += 1
                    else:
                        line.count_down[track.class_name] += 1
                    track.has_crossed = True
                    break

    # ─── 轨迹更新 ───────────────────────────────────────────

    def update_tracks(self, detections: List[Tuple], frame: np.ndarray):
        """更新轨迹 — 优化版匹配算法"""
        self.frame_count += 1

        if not detections:
            # 所有轨迹失联 +1
            for tid in list(self.tracks.keys()):
                self.tracks[tid].frames_missing += 1
                if self.tracks[tid].frames_missing > self.max_frames_missing:
                    del self.tracks[tid]
            return

        # 解包检测结果
        det_boxes = np.array([d[0] for d in detections], dtype=np.int32)
        det_class_ids = np.array([d[1] for d in detections], dtype=np.int32)

        n_dets = len(detections)
        matched_dets = set()
        matched_tracks = set()

        # 对每个活跃轨迹查找最佳匹配
        for tid, track in self.tracks.items():
            if track.class_id not in det_class_ids:
                track.frames_missing += 1
                continue

            # 相同类别的检测索引
            same_class_mask = det_class_ids == track.class_id
            same_class_indices = np.where(same_class_mask)[0]
            available_mask = ~np.isin(same_class_indices, list(matched_dets))
            candidate_indices = same_class_indices[available_mask]

            if len(candidate_indices) == 0:
                track.frames_missing += 1
                continue

            candidate_boxes = det_boxes[candidate_indices]
            ious = self._bbox_iou_batch(
                np.array(track.bbox, dtype=np.int32), candidate_boxes
            )

            best_local_idx = np.argmax(ious)
            if ious[best_local_idx] >= 0.15:
                best_det_idx = candidate_indices[best_local_idx]
                matched_dets.add(best_det_idx)
                matched_tracks.add(tid)

                # 更新轨迹
                old_center = track.center
                new_box = tuple(det_boxes[best_det_idx].tolist())
                new_center = (
                    (new_box[0] + new_box[2]) // 2,
                    (new_box[1] + new_box[3]) // 2,
                )
                track.bbox = new_box
                track.prev_center = old_center
                track.center = new_center
                track.last_seen_frame = self.frame_count
                track.frames_missing = 0
                track.history.append(new_center)
                if len(track.history) > 8:  # 从20降到8
                    track.history.pop(0)

                if self.counting_lines:
                    self.check_crossing(track)
            else:
                track.frames_missing += 1

        # 标记未匹配的轨迹
        for tid in list(self.tracks.keys()):
            if tid not in matched_tracks:
                self.tracks[tid].frames_missing += 1
                if self.tracks[tid].frames_missing > self.max_frames_missing:
                    del self.tracks[tid]

        # 为未匹配的检测创建新轨迹
        for idx in range(n_dets):
            if idx not in matched_dets:
                bbox = tuple(det_boxes[idx].tolist())
                class_id = int(det_class_ids[idx])
                class_name = self.vehicle_classes[class_id][0]
                center = (
                    (bbox[0] + bbox[2]) // 2,
                    (bbox[1] + bbox[3]) // 2,
                )
                new_track = VehicleTrack(
                    track_id=self.next_track_id,
                    class_name=class_name,
                    class_id=class_id,
                    bbox=bbox,
                    center=center,
                )
                new_track.first_seen_frame = self.frame_count
                new_track.last_seen_frame = self.frame_count
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1

    # ─── 检测 + 计数 ────────────────────────────────────────

    def detect_and_count(self, frame: np.ndarray):
        """检测并计数 — 批量 GPU→CPU 传输"""

        # FP16 输入 (如果启用)
        if self.use_fp16:
            results = self.model(frame, conf=self.conf_threshold, verbose=False, half=True)
        else:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # ── 关键优化: 批量 GPU→CPU ──
            cls_ids = boxes.cls.cpu().numpy().astype(int)  # 全部类别
            xyxy_all = boxes.xyxy.cpu().numpy().astype(int)  # 全部边界框
            confs = boxes.conf.cpu().numpy()  # 全部置信度

            for i in range(len(cls_ids)):
                class_id = cls_ids[i]
                if class_id in self.vehicle_classes:
                    detections.append((
                        (xyxy_all[i, 0], xyxy_all[i, 1], xyxy_all[i, 2], xyxy_all[i, 3]),
                        class_id,
                        float(confs[i]),
                    ))

        self.update_tracks(detections, frame)
        annotated_frame = self.draw_annotations(frame)
        stats = self.get_statistics()
        return annotated_frame, stats

    # ─── 统计 ───────────────────────────────────────────────

    def get_statistics(self) -> Dict:
        stats = {
            'frame': self.frame_count,
            'active_tracks': len(self.tracks),
            'by_line': {},
            'by_class': defaultdict(int),
        }

        for line in self.counting_lines:
            total_up = sum(line.count_up.values())
            total_down = sum(line.count_down.values())
            stats['by_line'][line.name] = {
                'up': dict(line.count_up),
                'down': dict(line.count_down),
                'total_up': total_up,
                'total_down': total_down,
                'total': total_up + total_down,
            }
            for class_name, count in line.count_up.items():
                stats['by_class'][class_name] += count
            for class_name, count in line.count_down.items():
                stats['by_class'][class_name] += count

        stats['by_class'] = dict(stats['by_class'])
        total_all = sum(line.get_total() for line in self.counting_lines)
        stats['by_class']['总计'] = total_all
        return stats

    # ─── 绘制 ───────────────────────────────────────────────

    def draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """绘制标注 — 使用纹理缓存替代 PIL"""
        annotated = frame  # 直接修改，减少一次拷贝

        # 计数线
        for line in self.counting_lines:
            cv2.line(annotated, (0, line.y), (frame.shape[1], line.y), (0, 0, 255), 3)
            total_up = sum(line.count_up.values())
            total_down = sum(line.count_down.values())
            cv2.putText(annotated, f"UP {line.name}: {total_up}", (10, line.y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"DOWN {line.name}: {total_down}", (10, line.y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 轨迹
        fh, fw = frame.shape[:2]
        for tid, track in self.tracks.items():
            x1, y1, x2, y2 = track.bbox
            # 边界裁剪
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)

            color = self.track_colors[tid % len(self.track_colors)]

            # 边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 标签背景
            label_w = 140
            label_h = 28
            label_y1 = max(0, y1 - label_h)
            cv2.rectangle(annotated, (x1, label_y1), (x1 + label_w, y1), color, -1)

            # ID (英文，直接用 putText)
            cv2.putText(annotated, f"ID:{tid}", (x1 + 3, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            # 中文类名 — 使用纹理缓存
            class_name_cn = self.vehicle_names_cn.get(track.class_name, track.class_name)
            put_chinese_text_fast(annotated, class_name_cn, (x1 + 65, label_y1 + 3), (255, 255, 255))

            # 中心点
            cv2.circle(annotated, track.center, 4, color, -1)

            # 轨迹线 (仅最近 8 点)
            hist = track.history
            if len(hist) > 1:
                pts = np.array(hist[-8:], dtype=np.int32)
                cv2.polylines(annotated, [pts], False, color, 2)

        # 统计面板
        if self.counting_lines:
            self.draw_stats_panel(annotated)

        return annotated

    def draw_stats_panel(self, frame: np.ndarray) -> np.ndarray:
        """绘制统计面板"""
        panel_w, panel_h = 280, 120
        x_offset = max(10, frame.shape[1] - panel_w - 10)
        y_offset = 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (x_offset, y_offset),
                     (x_offset + panel_w, y_offset + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, dst=frame)

        cv2.rectangle(frame, (x_offset, y_offset),
                     (x_offset + panel_w, y_offset + panel_h), (0, 255, 255), 3)

        put_chinese_text_fast(frame, "车辆计数", (x_offset + 20, y_offset + 30),
                             color=(255, 255, 255))

        stats = self.get_statistics()
        total = stats['by_class'].get('总计', 0)
        put_chinese_text_fast(frame, f"总车辆数: {total}", (x_offset + 20, y_offset + 80),
                             color=(0, 255, 255))

        scale_text = f"缩放: {self.current_scale:.1f}x"
        cv2.putText(frame, scale_text, (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        return frame

    # ─── 窗口管理 ───────────────────────────────────────────

    def create_resizable_window(self):
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_title, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)

    def resize_frame(self, frame: np.ndarray, scale: float) -> np.ndarray:
        if scale == 1.0:
            return frame
        h, w = frame.shape[:2]
        return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # ─── 视频处理 (多线程采集管道) ──────────────────────────

    def _capture_worker(self, cap: cv2.VideoCapture, queue: queue.Queue, stop_event: threading.Event):
        """采集线程: 持续读取帧到队列"""
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                queue.put(None)  # 结束信号
                break
            if queue.qsize() < 2:  # 限制队列深度防止内存爆炸
                queue.put(frame)

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_live: bool = True,
        counting_line_y: int = None,
    ):
        """处理视频 — 多线程管道版"""
        if counting_line_y is not None and not self.counting_lines:
            self.add_counting_line(counting_line_y, "主线")
        if not self.counting_lines:
            print("警告: 没有添加计数线，将使用默认位置 400")
            self.add_counting_line(400, "主线")

        # 打开视频
        if isinstance(video_path, str) and video_path.isdigit():
            cap = cv2.VideoCapture(int(video_path))
        else:
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 设置采集缓冲区 (减少延迟)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ── 输出编码器优化: 使用 H.264 ──
        out = None
        if output_path:
            # 尝试硬件编码器
            codecs_to_try = [
                ('avc1', cv2.VideoWriter_fourcc(*'avc1')),  # H.264
                ('H264', cv2.VideoWriter_fourcc(*'H264')),
                ('X264', cv2.VideoWriter_fourcc(*'X264')),
                ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # 回退
            ]
            for codec_name, fourcc in codecs_to_try:
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    print(f"输出编码器: {codec_name}")
                    break
            else:
                out = None
                print("警告: 无法创建输出视频")

        print("=" * 70)
        print("车辆计数系统 V5.0 - 高性能版")
        print("=" * 70)
        print(f"视频分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
        print(f"设备: {self.device}, FP16: {self.use_fp16}, 线程: {self.use_threading}")
        print(f"计数线位置: Y={self.counting_lines[0].y}")
        print("\n按键说明: q=退出 r=重置 s=截图 +/-=缩放 0=原始 f=全屏 t=切换线程")
        print("=" * 70)

        # ── 启动采集线程 ──
        stop_event = threading.Event()
        self._capture_queue = queue.Queue(maxsize=4)

        if self.use_threading:
            self._capture_thread = threading.Thread(
                target=self._capture_worker,
                args=(cap, self._capture_queue, stop_event),
                daemon=True,
            )
            self._capture_thread.start()

        if show_live:
            self.create_resizable_window()

        # FPS 计数器
        fps_values = []  # 滚动窗口
        fps_start_time = time.perf_counter()
        fps_frame_count = 0
        current_fps_display = 0.0

        frame_idx = 0
        skip_counter = 0

        try:
            while True:
                # ── 获取帧 ──
                if self.use_threading:
                    try:
                        frame = self._capture_queue.get(timeout=5.0)
                    except queue.Empty:
                        print("采集超时，退出")
                        break
                    if frame is None:
                        break
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break

                # ── 自适应跳帧 ──
                if self._adaptive_skip and self._processing_times:
                    avg_time = sum(self._processing_times[-30:]) / len(self._processing_times[-30:])
                    target_time = 1.0 / max(fps, 1)  # 目标帧间隔
                    if avg_time > target_time * 1.5:
                        skip_counter = max(1, int(avg_time / target_time)) - 1
                    else:
                        skip_counter = 0
                elif self._frame_skip > 0:
                    skip_counter = self._frame_skip

                if skip_counter > 0:
                    frame_idx += 1
                    if frame_idx % (skip_counter + 1) != 0:
                        continue
                    frame_idx += 1
                else:
                    frame_idx += 1

                # ── 推理 + 绘制 ──
                t_start = time.perf_counter()
                annotated_frame, stats = self.detect_and_count(frame)
                t_infer = time.perf_counter() - t_start
                self._processing_times.append(t_infer)

                # FPS
                fps_frame_count += 1
                if fps_frame_count >= 10:
                    now = time.perf_counter()
                    elapsed = now - fps_start_time
                    if elapsed > 0:
                        current_fps_display = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_start_time = now

                # FPS 叠加
                fps_text = f"FPS: {current_fps_display:.1f}"
                cv2.putText(annotated_frame, fps_text, (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 写入输出
                if out:
                    out.write(annotated_frame)

                # 显示
                if show_live:
                    display_frame = self.resize_frame(annotated_frame, self.current_scale)
                    cv2.imshow(self.window_title, display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self.reset_counts()
                    elif key == ord('s'):
                        self.save_screenshot(annotated_frame)
                    elif key in (ord('+'), ord('=')):
                        self.current_scale = min(self.current_scale + 0.1, self.max_scale)
                    elif key in (ord('-'), ord('_')):
                        self.current_scale = max(self.current_scale - 0.1, self.min_scale)
                    elif key == ord('0'):
                        self.current_scale = 1.0
                    elif key == ord('f'):
                        current_flags = cv2.getWindowProperty(
                            self.window_title, cv2.WND_PROP_FULLSCREEN
                        )
                        if current_flags == cv2.WINDOW_FULLSCREEN:
                            cv2.setWindowProperty(
                                self.window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
                            )
                        else:
                            cv2.setWindowProperty(
                                self.window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                            )
                    elif key == ord('t'):
                        self.use_threading = not self.use_threading
                        print(f"线程模式: {'开启' if self.use_threading else '关闭'}")

        finally:
            stop_event.set()
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=2.0)
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            self.print_final_report()

    # ─── 图片检测 ───────────────────────────────────────────

    def detect_image(self, image_path: str, output_path: Optional[str] = None):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"无法读取图片: {image_path}")
        annotated_frame, stats = self.detect_and_count(frame)
        if output_path:
            cv2.imwrite(output_path, annotated_frame)
        return annotated_frame, stats

    def save_screenshot(self, frame: np.ndarray):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"截图已保存: {filename}")

    def print_final_report(self):
        print("\n" + "=" * 70)
        print("Final Statistics Report")
        print("=" * 70)
        stats = self.get_statistics()
        total_all = 0
        for line in self.counting_lines:
            ls = stats['by_line'].get(line.name, {})
            tu = ls.get('total_up', 0)
            td = ls.get('total_down', 0)
            lt = tu + td
            total_all += lt
            print(f"\n[{line.name}]")
            print(f"  UP: {tu}")
            for cn, c in ls.get('up', {}).items():
                print(f"    - {self.vehicle_names_cn.get(cn, cn)}: {c}")
            print(f"  DOWN: {td}")
            for cn, c in ls.get('down', {}).items():
                print(f"    - {self.vehicle_names_cn.get(cn, cn)}: {c}")
            print(f"  Line Total: {lt}")

        print(f"\n[TOTAL]")
        for cn, c in stats['by_class'].items():
            if cn != '总计':
                print(f"  {self.vehicle_names_cn.get(cn, cn)}: {c}")
        print(f"\n  Grand Total: {total_all}")
        print(f"\nFrames Processed: {stats['frame']}")
        print("=" * 70)


# ─── 命令行入口 ─────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Vehicle Counting System V5.0')
    parser.add_argument('--mode', type=str, default='video',
                       choices=['image', 'video', 'camera'])
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--line', type=int, default=None)
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--window-title', type=str, default='Vehicle Counting System V5.0')
    parser.add_argument('--no-fp16', action='store_true', help='禁用 FP16')
    parser.add_argument('--no-thread', action='store_true', help='禁用多线程')
    parser.add_argument('--frame-skip', type=int, default=0,
                       help='跳帧: 0=自适应, 1=每帧, 2=隔一帧...')

    args = parser.parse_args()

    counter = VehicleCounter(
        model_path=args.model,
        conf_threshold=args.conf,
        window_title=args.window_title,
        use_fp16=not args.no_fp16,
        use_threading=not args.no_thread,
        frame_skip=args.frame_skip,
    )

    if args.line:
        counter.add_counting_line(args.line, "主线")

    if args.mode == 'image':
        if not args.source:
            print("Error: Image mode requires --source parameter")
            return
        annotated_frame, stats = counter.detect_and_count(cv2.imread(args.source))
        print("\n检测结果:")
        detection_count = sum(1 for _ in counter.tracks.values())
        print(f"检测到车辆数: {detection_count}")
        for class_name, count in stats['by_class'].items():
            if class_name != '总计':
                ccn = counter.vehicle_names_cn.get(class_name, class_name)
                print(f"  {ccn}: {count}")
        counter.create_resizable_window()
        cv2.imshow(counter.window_title, annotated_frame)
        print("\n按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode == 'video':
        if not args.source:
            print("Error: Video mode requires --source parameter")
            return
        counter.process_video(
            video_path=args.source,
            output_path=args.output,
            counting_line_y=args.line,
        )

    elif args.mode == 'camera':
        camera_id = int(args.source) if args.source else 0
        counter.process_video(
            video_path=str(camera_id),
            output_path=args.output,
            counting_line_y=args.line,
        )


if __name__ == '__main__':
    main()
