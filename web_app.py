"""
车辆检测系统 - Web 前端服务
基于 Flask + SocketIO 的现代化 Web 界面
"""

import cv2
import numpy as np
import os
import sys
import json
import time
import base64
import threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from flask import (
    Flask, render_template, request, jsonify,
    Response, send_from_directory, session
)
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vehicle_counter import VehicleCounter

# ─── 应用初始化 ───────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vehicle-detection-web-2024'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB 上传限制
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading',
                    ping_timeout=60, ping_interval=25)

# ─── 全局状态 ─────────────────────────────────────────────────

# 活跃的检测器实例
detector: VehicleCounter = None
detector_lock = threading.Lock()

# 视频处理状态
video_processing = False
video_stop_event = threading.Event()

# 实时统计缓存
current_stats = {
    'frame': 0,
    'active_tracks': 0,
    'by_line': {},
    'by_class': {},
    'fps': 0.0,
    'timestamp': '',
}

stats_lock = threading.Lock()

# 允许的文件扩展名
ALLOWED_IMAGE_EXT = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'}
ALLOWED_VIDEO_EXT = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}


# ─── 辅助函数 ─────────────────────────────────────────────────

def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set


def get_or_create_detector(model_path='yolov8n.pt', conf=0.5, use_fp16=True,
                           use_threading=True, frame_skip=0):
    """获取或创建全局检测器实例"""
    global detector
    with detector_lock:
        if detector is None:
            detector = VehicleCounter(
                model_path=model_path,
                conf_threshold=conf,
                use_fp16=use_fp16,
                use_threading=use_threading,
                frame_skip=frame_skip,
            )
        return detector


def reset_detector():
    """重置检测器"""
    global detector
    with detector_lock:
        if detector is not None:
            detector.reset_counts()
        detector = None


def stats_to_json(stats: dict) -> dict:
    """将统计数据转换为可序列化的 JSON 格式"""
    result = {
        'frame': stats.get('frame', 0),
        'active_tracks': stats.get('active_tracks', 0),
        'by_line': {},
        'by_class': {},
        'timestamp': datetime.now().strftime('%H:%M:%S'),
    }

    # 处理 by_line
    for line_name, line_data in stats.get('by_line', {}).items():
        result['by_line'][line_name] = {
            'up': dict(line_data.get('up', {})),
            'down': dict(line_data.get('down', {})),
            'total_up': line_data.get('total_up', 0),
            'total_down': line_data.get('total_down', 0),
            'total': line_data.get('total', 0),
        }

    # 处理 by_class
    for class_name, count in stats.get('by_class', {}).items():
        result['by_class'][class_name] = count

    return result


def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """将 OpenCV BGR 帧编码为 JPEG 字节"""
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes()


def encode_frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """将 OpenCV BGR 帧编码为 base64 字符串"""
    jpeg_bytes = encode_frame_to_jpeg(frame, quality)
    return base64.b64encode(jpeg_bytes).decode('utf-8')


# ─── 页面路由 ─────────────────────────────────────────────────

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


# ─── 图片检测 API ─────────────────────────────────────────────

@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    """上传图片进行车辆检测"""
    if 'image' not in request.files:
        return jsonify({'error': '请上传图片文件'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXT):
        return jsonify({'error': f'不支持的图片格式，支持: {", ".join(ALLOWED_IMAGE_EXT)}'}), 400

    try:
        # 保存上传文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_name = f"upload_{timestamp}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        file.save(save_path)

        # 读取图片
        frame = cv2.imread(save_path)
        if frame is None:
            return jsonify({'error': '无法读取图片文件'}), 400

        # 初始化检测器
        conf = float(request.form.get('conf', 0.5))
        line_y = request.form.get('line_y', None)
        counter = VehicleCounter(conf_threshold=conf, use_fp16=False, use_threading=False)

        if line_y is not None:
            counter.add_counting_line(int(line_y), '主线')

        # 执行检测
        annotated_frame, stats = counter.detect_and_count(frame)

        # 编码结果
        result_base64 = encode_frame_to_base64(annotated_frame)

        # 保存结果图片
        result_filename = f"result_{timestamp}_{filename}"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, annotated_frame)

        # 清理上传的原始文件（可选）
        # os.remove(save_path)

        response_data = {
            'success': True,
            'image': f'data:image/jpeg;base64,{result_base64}',
            'stats': stats_to_json(stats),
            'result_path': f'/uploads/{result_filename}',
            'detections': len(counter.tracks),
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f'检测失败: {str(e)}'}), 500


# ─── 视频处理 API ─────────────────────────────────────────────

@app.route('/api/detect/video/info', methods=['POST'])
def get_video_info():
    """获取视频文件信息"""
    if 'video' not in request.files:
        return jsonify({'error': '请上传视频文件'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    if not allowed_file(file.filename, ALLOWED_VIDEO_EXT):
        return jsonify({'error': f'不支持的视频格式，支持: {", ".join(ALLOWED_VIDEO_EXT)}'}), 400

    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_name = f"video_{timestamp}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        file.save(save_path)

        cap = cv2.VideoCapture(save_path)
        if not cap.isOpened():
            return jsonify({'error': '无法打开视频文件'}), 400

        info = {
            'path': save_path,
            'filename': filename,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': round(cap.get(cv2.CAP_PROP_FPS), 2),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1), 2),
        }
        cap.release()

        return jsonify({'success': True, 'info': info})

    except Exception as e:
        return jsonify({'error': f'读取视频信息失败: {str(e)}'}), 500


# ─── 视频流处理 (WebSocket) ───────────────────────────────────

@socketio.on('start_video_process')
def handle_video_process(data):
    """通过 WebSocket 启动视频处理并实时推送帧"""
    global video_processing, video_stop_event

    video_path = data.get('path', '')
    conf = float(data.get('conf', 0.5))
    line_y = data.get('line_y', None)
    frame_skip = int(data.get('frame_skip', 0))

    if not video_path or not os.path.exists(video_path):
        emit('video_error', {'message': '视频文件不存在'})
        return

    try:
        # 初始化检测器
        counter = VehicleCounter(
            conf_threshold=conf,
            use_fp16=False,  # Web 模式下不使用 FP16
            use_threading=False,
            frame_skip=frame_skip,
        )

        if line_y is not None:
            counter.add_counting_line(int(line_y), '主线')
        elif not counter.counting_lines:
            counter.add_counting_line(400, '主线')

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            emit('video_error', {'message': '无法打开视频文件'})
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        video_processing = True
        video_stop_event.clear()

        emit('video_started', {
            'total_frames': total_frames,
            'fps': fps,
        })

        frame_idx = 0
        fps_frame_count = 0
        fps_start_time = time.perf_counter()
        current_fps = 0.0

        while video_processing and not video_stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # 处理帧
            t_start = time.perf_counter()
            annotated_frame, stats = counter.detect_and_count(frame)

            # FPS 计算
            fps_frame_count += 1
            if fps_frame_count >= 10:
                now = time.perf_counter()
                elapsed = now - fps_start_time
                if elapsed > 0:
                    current_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = now

            # 编码帧
            jpeg_bytes = encode_frame_to_jpeg(annotated_frame, quality=75)
            frame_b64 = base64.b64encode(jpeg_bytes).decode('utf-8')

            # 发送帧和统计
            stats_json = stats_to_json(stats)
            stats_json['fps'] = round(current_fps, 1)
            stats_json['progress'] = round(frame_idx / max(total_frames, 1) * 100, 1)
            stats_json['current_frame'] = frame_idx

            emit('video_frame', {
                'frame': f'data:image/jpeg;base64,{frame_b64}',
                'stats': stats_json,
            })

            # 控制帧率（避免过快发送）
            socketio.sleep(0.03)  # ~30 FPS 最大

        cap.release()
        video_processing = False

        # 发送最终结果
        final_stats = stats_to_json(counter.get_statistics())
        emit('video_complete', {'stats': final_stats})

    except Exception as e:
        video_processing = False
        emit('video_error', {'message': f'视频处理失败: {str(e)}'})


@socketio.on('stop_video_process')
def handle_stop_video():
    """停止视频处理"""
    global video_processing, video_stop_event
    video_processing = False
    video_stop_event.set()
    emit('video_stopped', {'message': '视频处理已停止'})


# ─── 摄像头实时流 API ─────────────────────────────────────────

camera_active = False
camera_stop_event = threading.Event()
camera_thread = None


def camera_stream_thread(camera_id, conf, line_y):
    """摄像头采集线程"""
    global camera_active, current_stats

    try:
        counter = VehicleCounter(
            conf_threshold=conf,
            use_fp16=False,
            use_threading=False,
        )

        if line_y is not None:
            counter.add_counting_line(int(line_y), '主线')
        elif not counter.counting_lines:
            counter.add_counting_line(400, '主线')

        cap = cv2.VideoCapture(int(camera_id) if camera_id.isdigit() else camera_id)
        if not cap.isOpened():
            socketio.emit('camera_error', {'message': f'无法打开摄像头: {camera_id}'})
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        fps_frame_count = 0
        fps_start_time = time.perf_counter()
        current_fps = 0.0

        while camera_active and not camera_stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            # 检测
            annotated_frame, stats = counter.detect_and_count(frame)

            # FPS
            fps_frame_count += 1
            if fps_frame_count >= 10:
                now = time.perf_counter()
                elapsed = now - fps_start_time
                if elapsed > 0:
                    current_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = now

            # 更新全局统计
            stats_json = stats_to_json(stats)
            stats_json['fps'] = round(current_fps, 1)
            with stats_lock:
                current_stats.update(stats_json)

            # 编码发送
            jpeg_bytes = encode_frame_to_jpeg(annotated_frame, quality=70)
            frame_b64 = base64.b64encode(jpeg_bytes).decode('utf-8')

            socketio.emit('camera_frame', {
                'frame': f'data:image/jpeg;base64,{frame_b64}',
                'stats': stats_json,
            })

            socketio.sleep(0.04)  # ~25 FPS max

        cap.release()

    except Exception as e:
        socketio.emit('camera_error', {'message': f'摄像头错误: {str(e)}'})
    finally:
        camera_active = False


@socketio.on('start_camera')
def handle_start_camera(data):
    """启动摄像头"""
    global camera_active, camera_stop_event, camera_thread

    if camera_active:
        emit('camera_error', {'message': '摄像头已在运行中'})
        return

    camera_id = str(data.get('camera_id', 0))
    conf = float(data.get('conf', 0.5))
    line_y = data.get('line_y', None)

    camera_active = True
    camera_stop_event.clear()

    camera_thread = threading.Thread(
        target=camera_stream_thread,
        args=(camera_id, conf, line_y),
        daemon=True,
    )
    camera_thread.start()

    emit('camera_started', {'camera_id': camera_id})


@socketio.on('stop_camera')
def handle_stop_camera():
    """停止摄像头"""
    global camera_active, camera_stop_event
    camera_active = False
    camera_stop_event.set()
    emit('camera_stopped', {'message': '摄像头已停止'})


# ─── 统计 API ─────────────────────────────────────────────────

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取当前统计信息"""
    with stats_lock:
        return jsonify(current_stats)


# ─── 配置 API ─────────────────────────────────────────────────

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """获取或更新配置"""
    if request.method == 'GET':
        return jsonify({
            'conf_threshold': 0.5,
            'line_y': 400,
            'frame_skip': 0,
            'model': 'yolov8n.pt',
        })

    elif request.method == 'POST':
        data = request.get_json()
        if data:
            # 更新检测器配置（如果存在）
            with detector_lock:
                if detector is not None:
                    if 'conf_threshold' in data:
                        detector.conf_threshold = float(data['conf_threshold'])
            return jsonify({'success': True, 'config': data})
        return jsonify({'error': '无效配置'}), 400


# ─── 重置 API ─────────────────────────────────────────────────

@app.route('/api/reset', methods=['POST'])
def reset_counts():
    """重置计数"""
    with detector_lock:
        if detector is not None:
            detector.reset_counts()
    with stats_lock:
        current_stats.clear()
    return jsonify({'success': True, 'message': '计数已重置'})


# ─── 上传文件访问 ─────────────────────────────────────────────

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """访问上传的文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ─── 静态文件 ─────────────────────────────────────────────────

@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


# ─── 健康检查 ─────────────────────────────────────────────────

@app.route('/api/health')
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'camera_active': camera_active,
        'video_processing': video_processing,
    })


# ─── 错误处理 ─────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': '文件太大，最大支持 500MB'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': '页面未找到'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': '服务器内部错误'}), 500


# ─── 启动入口 ─────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  车辆检测系统 - Web 前端服务")
    print("  Vehicle Detection System - Web Interface")
    print("=" * 60)
    print(f"  访问地址: http://localhost:5000")
    print(f"  上传目录: {app.config['UPLOAD_FOLDER']}")
    print("=" * 60)

    # 确保上传目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
