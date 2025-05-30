# -*- coding: utf-8 -*-
# coding=utf-8
import os, sys
# Disable hardware acceleration, cause cv2.VideoCapture(i) fx slow
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from cv2_enumerate_cameras import enumerate_cameras
from flask import Flask, Response, jsonify, request, send_file, abort
import threading
import time
import json

# yolo model
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# 加載 YOLOv8 模型 (你可以使用預訓練模型或自己的模型)
model_obj = None # YOLO('yolov8n.pt') # 例如 yolov8n.pt, yolov8s.pt, etc.
# 載入 YOLOv8 姿態估計模型
# 您可以選擇不同大小的模型：yolov8n-pose.pt (nano), yolov8s-pose.pt (small), yolov8m-pose.pt (medium) 等
# 更大的模型通常更準確，但運行速度較慢，需要更多計算資源。
model_pose = YOLO('yolov8s-pose.pt') 

# --- 全局變數和鎖用於管理攝影機狀態 ---
# 攝影機實例的引用
current_camera = None
# 用於保護對 current_camera 的訪問，確保線程安全
camera_lock = threading.Lock()
# 當前活躍的攝影機ID和解析度
current_camera_info = {"id": -1, "width": 640, "height": 480, "fps": 30}
# 用於控制 generate_frames 停止的事件
stop_streaming_event = threading.Event()
# --- END 全局變數和鎖 ---
VID_IDLE_SCREEN_FILE = 'static/vid_idle.jpg'

SAVE_JPEG_CONFIG = {
    "enabled": False,           # 是否啟用 JPEG 儲存
    "save_path": "captured_frames", # 儲存檔案的目錄
    "interval_seconds": 3,      # 每隔多少秒儲存一張 (如果為 0 或更小則每幀都存)
    "last_save_time": 0         # 上次儲存的時間戳
}

# 檢查 TensorFlow 的 CUDA 和 cuDNN 版本
"""
TypeError: Descriptors cannot be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be 
regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python 
 parsing and will be much slower)
Check dependencies:
 pip3 install pipdeptree
 pipdeptree > dependencies.tx
"""
#os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
#import torch
#import tensorflow as tf
def show_gpuinfo():
    # nvcc --version
    # nvidia-smi
    print("check CUDA envrionment...")
    if torch.cuda.is_available():
        # 檢查 CUDA 是否可用
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")
        # 檢查 PyTorch 編譯時使用的 CUDA 版本 (這是最重要的，表示 PyTorch 預期什麼版本的 CUDA)
        print(f"PyTorch 編譯時使用的 CUDA 版本: {torch.version.cuda}")
        
        # 檢查 PyTorch 當前運行的 cuDNN 版本
        # 注意：這可能與系統實際安裝的 cuDNN 版本略有不同，但表示 PyTorch 正在使用的版本
        print(f"PyTorch 當前運行的 cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"cuDNN 是否可用: {torch.backends.cudnn.is_available()}")
    
        # 顯示更多關於 GPU 的信息 (可選)
        print(f"GPU 數量: {torch.cuda.device_count()}")
        print(f"當前 GPU 名稱: {torch.cuda.get_device_name(0)}") # 獲取第一個 GPU 的名稱
    else:
        print("CUDA 不可用。請檢查您的 NVIDIA 驅動程式和 PyTorch 安裝。")

    # 檢查 TensorFlow 是否使用 GPU
    print(f"TensorFlow 是否使用 GPU: {tf.config.list_physical_devices('GPU')}")
    
    # 檢查 TensorFlow 運行時使用的 CUDA 版本
    # 注意：這個 API 在 TensorFlow 2.x 中可用
    print(f"TensorFlow 運行時使用的 CUDA 版本: {tf.sysconfig.get_build_info().get('CUDA_VERSION')}")
    
    # 檢查 TensorFlow 運行時使用的 cuDNN 版本
    print(f"TensorFlow 運行時使用的 cuDNN 版本: {tf.sysconfig.get_build_info().get('CUDNN_VERSION')}")
    
    # 獲取更多關於 GPU 的信息 (可選)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPU 數量: {len(physical_devices)}")
        for i, device in enumerate(physical_devices):
            print(f"GPU {i} 名稱: {device.name}")
    else:
        print("TensorFlow 未檢測到 GPU。請檢查您的 NVIDIA 驅動程式和 TensorFlow 安裝。")
    
def frame_draw_pose(annotated_frame, result):
    r = result
    # 獲取關鍵點數據
    # r.keypoints.xy 是一個 (num_people, num_keypoints, 2) 的 NumPy 陣列
    # 其中 num_keypoints 預設為 17 (COCO 數據集)
    keypoints = r.keypoints.xy.cpu().numpy()
    
    # 獲取 bounding boxes (如果需要)
    # boxes = r.boxes.xyxy.cpu().numpy()

    # 定義 COCO 關鍵點連接方式 (骨架)
    # 這些是預設的連接，您也可以根據需要自定義
    skeleton_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), # 頭部與手臂
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11), # 軀幹與手臂
        (5, 11), (6, 12), # 軀幹與臀部
        (11, 12), (11, 13), (12, 14), (13, 15), (14, 16) # 臀部與腿部
    ]
    # COCO 關鍵點索引：
    # 0: 鼻子, 1: 左眼, 2: 右眼, 3: 左耳, 4: 右耳
    # 5: 左肩, 6: 右肩, 7: 左肘, 8: 右肘, 9: 左腕, 10: 右腕
    # 11: 左臀, 12: 右臀, 13: 左膝, 14: 右膝, 15: 左踝, 16: 右踝

    # 繪製每個人的骨架
    for kps in keypoints: # 遍歷每個人
        # 繪製關鍵點
        for kp_idx, (x, y) in enumerate(kps):
            # 檢查關鍵點是否有效 (通常 x, y 不會是 0,0 如果沒有檢測到)
            # 您可能需要根據模型的輸出檢查置信度 (keypoints.conf)
            if x > 0 and y > 0: # 簡單判斷是否有效關鍵點
                cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 255), -1) # 黃色點

        # 繪製連接線 (骨架)
        for connection in skeleton_connections:
            p1_idx, p2_idx = connection
            # 確保兩個關鍵點都存在且有效
            if kps[p1_idx][0] > 0 and kps[p1_idx][1] > 0 and \
               kps[p2_idx][0] > 0 and kps[p2_idx][1] > 0:
                
                pt1 = (int(kps[p1_idx][0]), int(kps[p1_idx][1]))
                pt2 = (int(kps[p2_idx][0]), int(kps[p2_idx][1]))
                cv2.line(annotated_frame, pt1, pt2, (0, 0, 255), 2) # 紅色線條
    return annotated_frame

def detect_and_draw_pose(model, frame):
    """
    對單幀影像進行人體姿態偵測並繪製骨架。
    Args:
        frame (np.array): 輸入的 OpenCV 影像幀。
    Returns:
        np.array: 繪製了骨架的影像幀。
    """
    # 運行姿態估計推理
    # verbose=False 可以禁用控制台的詳細輸出
    # conf 參數可以調整置信度閾值，box 參數可以控制是否顯示bounding box
    # show_conf=False, show_labels=False 可以關閉信心分數和標籤顯示
    results = model.predict(
        source=frame,
        conf=0.5,       # 置信度閾值，低於此值會被過濾
        iou=0.7,        # IoU 閾值，用於非極大值抑制
        show_conf=False, # 不顯示關鍵點的置信度
        show_labels=False, # 不顯示類別標籤（例如 'person'）
        save=False,     # 不將結果儲存到磁碟
        verbose=False,   # 禁用詳細的日誌輸出
        imgsz=frame.shape[1], #current_camera_info["width"], # width for inference size
        device='cuda' #'cpu' # Uncomment and set to '0' or 'cuda' if you have a GPU
    )

    # results 物件包含了偵測到的所有姿態資訊
    # 遍歷每個偵測到的結果 (results[0] 是當前幀的結果)
    # ultralytics v8.1.0 (或更高版本) 的 predict 會自動在圖像上繪製關鍵點和骨架
    # 你可以直接從 results 獲取繪製後的圖像
    
    """
    使用 YOLOv8-pose 獲取影像中的人體關鍵點。
    返回一個列表，每個元素代表一個人，包含其關鍵點。
    關鍵點格式為 [x1, y1, conf1, x2, y2, conf2, ...], 共 17 * 3 個值。
    """
    if not results or len(results) == 0 or results[0].keypoints is None:
        return frame 
  
    # 由於 model.predict 已經自動在 'frame' 上繪製了結果 (如果它是一個可變的numpy array)
    # 或者如果你需要繪製在一個新的副本上，可以這樣處理：
    annotated_frame = frame.copy() # 創建一個副本以繪製，避免修改原始幀
    
    num_people_detected = 0
    all_person_keypoints = []
    # Iterate over results and draw
    for r in results:
        # r.plot() 會在 r.orig_img (原始圖像) 上繪製結果並返回一個新的圖像。
        # 如果你的 frame 就是 r.orig_img，那麼 r.plot() 會基於它進行繪製。
        annotated_frame = r.plot() # 直接獲取繪製後的圖像
        
        if r.keypoints.is_empty:
           continue

        # r.keypoints.data 是一個 Tensor，其 shape 為 (檢測到的人數, 關鍵點數量, 3)
        # 例如 (N, 17, 3)，其中 N 是檢測到的人數，17 是 COCO 關鍵點數量，3 是 (x, y, conf)
        num_people_detected = r.keypoints.data.shape[0]
        print(f"Pose 偵測到 {num_people_detected} 個人")

        # 遍歷每個人檢測到的關鍵點
        for i, person_keypoints_tensor in enumerate(r.keypoints.data):
            # person_keypoints_tensor 的 shape 是 (17, 3)
            # 提取 (x, y) 座標和置信度 (conf)
            keypoints_xy_conf = person_keypoints_tensor.cpu().numpy() # 轉換為 NumPy 陣列
            
            # keypoints_xy_conf 格式為 [[x1, y1, conf1], [x2, y2, conf2], ...]
            # 其中 x, y 是像素座標，conf 是置信度

            print(f"\n--- 第 {i+1} 個人 ---")
            print("關鍵點 (x, y, conf):")
            # 可以選擇性地過濾掉置信度低的關鍵點
            valid_keypoints = []
            for kp_idx, (x, y, conf) in enumerate(keypoints_xy_conf):
                # 您可以設定一個閾值來判斷關鍵點是否有效
                if conf > 0.5: # 例如，只考慮置信度大於 0.3 的關鍵點
                    valid_keypoints.append((int(x), int(y), conf, kp_idx))
                    # print(f"  關鍵點 {kp_idx:2d}: x={int(x)}, y={int(y)}, conf={conf:.2f}")
                else:
                    pass # print(f"  關鍵點 {kp_idx:2d}: 置信度過低 ({conf:.2f})")
            
            # 將每個人的有效關鍵點列表儲存起來
            all_person_keypoints.append(valid_keypoints)
        
        # 手動遍歷每個檢測到的姿態,並在 frame 上繪製,這樣你可以更精細地控制繪製細節        
        # 檢查是否檢測到關鍵點
        # annotated_frame = frame_draw_pose(annotated_frame, r)


    return annotated_frame

# --- 輔助函數：釋放攝影機 ---
def release_current_camera():
    """安全地釋放當前活躍的攝影機資源。"""
    global current_camera_info
    global current_camera
    print(f"release_current_camera:...\n")
    with camera_lock:
        if current_camera and current_camera.isOpened():
            current_camera.release()
            print(f"攝影機 {current_camera_info['id']} 已釋放。")
        else:
            print(f"無攝影機被啟用")
        current_camera = None
        current_camera_info["id"] = -1 # 重置ID
        stop_streaming_event.set() # 發送停止串流信號
        time.sleep(0.5) # 給一點時間讓 generate_frames 檢測到停止事件
        stop_streaming_event.clear() # 清除停止信號，為下次啟動準備

def caminfo_properties(cap_id, cap):
    properties = {
        "CAP_PROP_FRAME_WIDTH": cv2.CAP_PROP_FRAME_WIDTH,
        "CAP_PROP_FRAME_HEIGHT": cv2.CAP_PROP_FRAME_HEIGHT,
        "CAP_PROP_FPS": cv2.CAP_PROP_FPS,
        "CAP_PROP_FOURCC": cv2.CAP_PROP_FOURCC,
    }
    for name, prop_id in properties.items():
        value = cap.get(prop_id)
        print(f"ID:{cap_id} {name}: {value}")
        
# --- API 1: 列出所有可用的 cv2.VideoCapture 裝置 ---
@app.route('/api/cameras', methods=['GET'])
def list_cameras():
    """
    偵測並列出系統上所有可用的 cv2.VideoCapture 裝置。
    返回一個包含裝置ID和其是否可開啟的字典列表。
    """
    # ID 1400: Logi C310 HD WebCam
    # ID 1401: USB2.0 HD UVC WebCam
    # ID 700: USB2.0 HD UVC WebCam
    # ID 701: Logi C310 HD WebCam
    # ID 702: OBS Virtual Camera
    # MS-Windows mode: https://pypi.org/project/cv2-enumerate-cameras/
    for camera_info in enumerate_cameras(cv2.CAP_MSMF):
        print(f'ID {camera_info.index}: {camera_info.name}')
    
    available_cameras = []
    # 嘗試從 0 到 9 測試攝影機，通常足夠了
    # 您可以根據需要調整這個範圍
    for i in range(10):
        # cap = cv2.VideoCapture(i, cv2.CAP_MSMF, {cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE})
        cap = cv2.VideoCapture(i)
        is_opened = cap.isOpened()
        if is_opened:
            caminfo_properties(i, cap)
            available_cameras.append({"id": i, "status": "available"})
            cap.release() # 立即釋放，避免佔用資源
        else:
            # available_cameras.append({"id": i, "status": "unavailable"})
            cap.release()
            # 某些情況下，即使 isOpened() 為 False，也可能是暫時性問題或正在被使用
            # 在實際應用中，可以考慮更精確的錯誤處理或重試機制
            # [ERROR:0@21.114] global obsensor_uvc_stream_channel.cpp:158 
            # cv::obsensor::getStreamChannelGroup Camera index out of range
            break
            
        # 增加延遲，避免過快地打開和關閉攝影機導致系統資源耗盡或錯誤
        time.sleep(0.1) 
            
    print(f"檢測到的攝影機列表: {json.dumps(available_cameras, indent=2)}")
    return jsonify(available_cameras)


# --- API 2 & 3: 指定開啟哪一個 cv2.VideoCapture 並設定參數 ---
@app.route('/api/open_camera', methods=['POST'])
def open_camera():
    """
    指定開啟特定的攝影機，並可選地設置其解析度和幀率。
    """
    global current_camera_info
    global current_camera
   
    data = request.get_json()
    camera_id = data.get('id', -1) # 預設為 -1，表示無效ID
    width = data.get('width', 640)
    height = data.get('height', 480)
    fps = data.get('fps', 30)

    if not isinstance(camera_id, int) or camera_id < 0:
        return jsonify({"status": "錯誤", "message": "請提供有效的攝影機ID (整數，>= 0)。"}), 400
    if not all(isinstance(val, int) and val > 0 for val in [width, height, fps]):
        return jsonify({"status": "錯誤", "message": "寬度、高度和幀率必須是正整數。"}), 400

    # 如果有其他攝影機正在運行，先釋放它
    if current_camera and current_camera.isOpened() and current_camera_info["id"] != camera_id:
        print(f"請求開啟新攝影機 {camera_id}，正在釋放舊攝影機 {current_camera_info['id']}。")
        release_current_camera()
        time.sleep(0.5) # 給點時間確保舊攝影機資源被釋放

    with camera_lock:
        # 如果請求開啟的正是當前已開啟的攝影機，且參數也一致，則不重複開啟
        if current_camera and current_camera.isOpened() and \
           current_camera_info["id"] == camera_id and \
           current_camera_info["width"] == width and \
           current_camera_info["height"] == height and \
           current_camera_info["fps"] == fps:
            print(f"攝影機 {camera_id} 已開啟且參數一致，無需重複操作。")
            return jsonify({"status": "成功", "message": f"攝影機 {camera_id} 已開啟，解析度 {width}x{height}@{fps} FPS。"})

        # 嘗試開啟攝影機
        current_camera = cv2.VideoCapture(camera_id)
        if not current_camera.isOpened():
            return jsonify({"status": "錯誤", "message": f"無法開啟攝影機 ID: {camera_id}。"}), 500

        # 設置攝影機參數
        current_camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        current_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        current_camera.set(cv2.CAP_PROP_FPS, fps)

        # 實際讀取設置後的參數，因為有些攝影機不支持所有指定的解析度或幀率
        actual_width = int(current_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(current_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = current_camera.get(cv2.CAP_PROP_FPS)

        current_camera_info["id"] = camera_id
        current_camera_info["width"] = actual_width
        current_camera_info["height"] = actual_height
        current_camera_info["fps"] = actual_fps

        print(f"成功開啟攝影機 ID: {camera_id}, 請求解析度: {width}x{height}@{fps} FPS, 實際解析度: {actual_width}x{actual_height}@{actual_fps} FPS。")
        return jsonify({"status": "成功", "message": f"攝影機 {camera_id} 已開啟。", 
                        "actual_resolution": {"width": actual_width, "height": actual_height, "fps": actual_fps}})


# --- API: 關閉當前攝影機 ---
@app.route('/api/close_camera', methods=['POST'])
def close_camera():
    """
    關閉當前活躍的攝影機。
    """
    release_current_camera()
    return jsonify({"status": "成功", "message": "攝影機已關閉。"}), 200

# --- MJPEG 串流函數 (與之前類似，但現在依賴 current_camera) ---
def generate_frames():
    """
    這個生成器函數會持續從當前活躍的網路攝影機讀取幀，
    將其編碼為 JPEG 格式，然後以 MJPEG over HTTP 的方式傳送。
    """
    global current_camera_info
    global current_camera
    global SAVE_JPEG_CONFIG
    print(f"generate_frames:...\n")
    
    while True:
        # 檢查是否有停止串流的信號
        if stop_streaming_event.is_set():
            print("收到停止串流信號，generate_frames 將停止。")
            break

        with camera_lock: # 在讀取幀時鎖定，確保 current_camera 不會被同時修改
            if current_camera is None or not current_camera.isOpened():
                # 如果攝影機未開啟或已關閉，等待一段時間後重試或退出
                # print("等待攝影機開啟...")
                time.sleep(0.5)
                continue

            success, frame = current_camera.read()

        if not success:
            print("錯誤：無法從網路攝影機讀取幀。可能攝影機已斷開或出現問題。")
            # 這裡可以考慮自動重啟攝影機或發送錯誤通知
            time.sleep(1) # 等待一會兒再重試
            continue
        else:
            if model_obj != None:
                # 執行 YOLOv8 檢測
                results = model_obj(frame, stream=True, verbose=False) # 使用 stream=True 更高效
                
                # 在幀上繪製檢測結果
                for r in results:
                    frame = r.plot() # ultralytics 內建的繪圖功能
                    
            if model_pose != None:
                try:
                    frame = detect_and_draw_pose(model_pose, frame)
                except Exception as e:
                    print(f"姿態估計時發生錯誤: {e}")
                    cv2.putText(frame, "Pose Error!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 您可以在這裡對 'frame' 進行任何 OpenCV 影像處理
            # 例如：灰度化、邊緣檢測、添加文字等
            # 範例：將影像轉換為灰度圖 (可選)
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ret, buffer = cv2.imencode('.jpg', gray_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            # 例如：添加當前時間、檢測結果等
            # cv2.putText(frame, f"Cam ID: {current_camera_info['id']} {time.strftime('%H:%M:%S')}", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 將幀編碼為 JPEG 格式
            # cv2.imencode 返回兩個值：一個布林值表示成功與否，和一個包含編碼後數據的 NumPy 數組
            # [int(cv2.IMWRITE_JPEG_QUALITY), 90] 表示 JPEG 品質設定為 90 (0-100, 數字越大品質越好，檔案越大)
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

            if not ret:
                print("錯誤：無法將幀編碼為 JPEG。")
                continue

            frame_bytes = buffer.tobytes()

            # --- 儲存 JPEG 檔案的邏輯 ---
            if SAVE_JPEG_CONFIG["enabled"]:
                current_time = time.time()
                # 判斷是否達到儲存間隔
                if current_time - SAVE_JPEG_CONFIG["last_save_time"] >= SAVE_JPEG_CONFIG["interval_seconds"]:
                    filename = os.path.join(
                        SAVE_JPEG_CONFIG["save_path"],
                        f"frame_{int(current_time)}.jpg" # 使用時間戳命名，確保唯一性
                    )
                    try:
                        with open(filename, 'wb') as f: # 以二進制寫入模式打開文件
                            f.write(frame_bytes) # 寫入 JPEG 位元組
                        print(f"已儲存幀: {filename}")
                        SAVE_JPEG_CONFIG["last_save_time"] = current_time # 更新上次儲存時間
                    except Exception as e:
                        print(f"儲存檔案時發生錯誤 {filename}: {e}")
            # --- END ---

            # 將 JPEG 數據包裝成 multipart/x-mixed-replace 格式
            # 每幀數據以 '--MJPGframe\r\n' 分隔，並包含 Content-Type 頭
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # 可選：控制幀率，避免 CPU/網路過載
        # 由於 generate_frames 是持續運行的，這裡可以調整延遲
        # 如果設置了 FPS，理論上 OpenCV 會嘗試提供那個幀率，但如果處理速度慢，則需要延遲
        time.sleep(0.05) # 例如，約 100 FPS
        
@app.route('/detect_results')
def detect_results():
    # 這個端點可以用來額外傳輸非圖像的檢測數據
    # 例如，你可以維護一個全局變量來存儲最新的檢測結果，並在這裡返回
    # 這會比每次都在圖像上疊加文字更靈活，C# 端可以解析 JSON
    return {"latest_detections": []} # 實際應用中會是動態數據

@app.route('/video_feed')
def video_feed():
    """
    這個路由處理視訊串流請求。
    它返回一個 Response 物件，其 mimetype 設定為 multipart/x-mixed-replace，
    並使用 generate_frames 函數作為內容生成器。
    """
    global current_camera
    
    if current_camera is None or not current_camera.isOpened():
        return "攝影機未開啟或無法訪問。請先使用 /api/open_camera 開啟攝影機。", 503 # Service Unavailable
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_idle')
def serve_video_idle():
    """
    提供一個位於伺服器上的 JPEG 檔案。
    """
    # 檢查檔案是否存在
    if not os.path.exists(VID_IDLE_SCREEN_FILE):
        # 如果檔案不存在，返回 404 錯誤
        abort(404, description="JPEG 檔案未找到。")
    try:
        # 使用 send_file 函數來傳送檔案
        # mimetype 參數是可選的，send_file 通常會自動偵測
        # 但明確指定可以確保正確性
        return send_file(VID_IDLE_SCREEN_FILE, mimetype='image/jpeg')
    except Exception as e:
        # 捕獲並處理傳送檔案時可能發生的錯誤
        print(f"傳送檔案時發生錯誤: {e}")
        abort(500, description="伺服器內部錯誤，無法提供檔案。")

@app.route('/')
def index():
    """
    提供一個簡單的 HTML 頁面，用於在瀏覽器中顯示視訊串流和調用 API 的說明。
    """
    return """
    <html>
    <head>
        <title>MJPEG 視訊串流與攝影機控制</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    </head>
    <body>
        <h1>網路攝影機 MJPEG 串流與 RESTful API 控制</h1>
        
        <h2>視訊串流</h2>
        <p>視訊串流在 /video_feed 處可用。請先使用 API 開啟攝影機。</p>
        <img id="videoStream" width="640" height="480" alt="視訊串流" src="/video_idle">

        <h2>API 測試</h2>
        <h3>1. 列出可用攝影機 (<code style="background-color:#eee; padding:2px 4px;">GET /api/cameras</code>)</h3>
        <button onclick="listCameras()">列出攝影機</button>
        <button onclick="openCamera()">開啟攝影機</button>
        <button onclick="closeCamera()">關閉攝影機</button>
        <pre id="cameraListResult"></pre>

        <h3>2. 開啟指定攝影機 (<code style="background-color:#eee; padding:2px 4px;">POST /api/open_camera</code>)</h3>
        <p>
            攝影機 ID: <input type="number" id="cameraId" value="0"> <br>
            寬度: <input type="number" id="width" value="640"> <br>
            高度: <input type="number" id="height" value="480"> <br>
            幀率 (FPS): <input type="number" id="fps" value="30"> <br>
        </p>
        <button onclick="openCamera()">開啟攝影機</button>
        <pre id="openCameraResult"></pre>

        <h3>3. 關閉攝影機 (<code style="background-color:#eee; padding:2px 4px;">POST /api/close_camera</code>)</h3>
        <button onclick="closeCamera()">關閉攝影機</button>
        <pre id="closeCameraResult"></pre>

        <script>
            function listCameras() {
                $.get("/api/cameras", function(data) {
                    $("#cameraListResult").text(JSON.stringify(data, null, 2));
                }).fail(function(jqXHR, textStatus, errorThrown) {
                    $("#cameraListResult").text("錯誤: " + errorThrown);
                });
            }

            function openCamera() {
                var cameraId = parseInt($("#cameraId").val());
                var width = parseInt($("#width").val());
                var height = parseInt($("#height").val());
                var fps = parseInt($("#fps").val());
                $.ajax({
                    url: "/api/open_camera",
                    type: "POST",
                    contentType: "application/json",
                    data: JSON.stringify({ id: cameraId, width: width, height: height, fps: fps }),
                    success: function(data) {
                        $("#openCameraResult").text(JSON.stringify(data, null, 2));
                        // 重新加載圖片以顯示新串流
                        $("#videoStream").attr("src", "/video_feed?" + new Date().getTime()); 
                    },
                    error: function(jqXHR, textStatus, errorThrown) {
                        $("#openCameraResult").text("錯誤: " + jqXHR.status + " - " + jqXHR.responseText);
                    }
                });
            }

            function closeCamera() {
                $.ajax({
                    url: "/api/close_camera",
                    type: "POST",
                    contentType: "application/json",
                    success: function(data) {
                        $("#closeCameraResult").text(JSON.stringify(data, null, 2));
                        $("#videoStream").attr("src", "/video_idle"); // 清空圖片源
                    },
                    error: function(jqXHR, textStatus, errorThrown) {
                        $("#closeCameraResult").text("錯誤: " + jqXHR.status + " - " + jqXHR.responseText);
                    }
                });
            }
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    # show_gpuinfo()
    # Flask 會在獨立的線程中運行請求處理器，但 generate_frames 會在一個協程中運行
    # 在應用程式退出前，確保攝影機資源被釋放
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        release_current_camera() # 確保程式終止時釋放攝影機
        print("伺服器已停止，攝影機已釋放。")