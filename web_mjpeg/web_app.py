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

app = Flask(__name__)

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
    "enabled": True,           # 是否啟用 JPEG 儲存
    "save_path": "captured_frames", # 儲存檔案的目錄
    "interval_seconds": 3,      # 每隔多少秒儲存一張 (如果為 0 或更小則每幀都存)
    "last_save_time": 0         # 上次儲存的時間戳
}

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
        time.sleep(0.1) # 例如，約 100 FPS


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
    # Flask 會在獨立的線程中運行請求處理器，但 generate_frames 會在一個協程中運行
    # 在應用程式退出前，確保攝影機資源被釋放
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        release_current_camera() # 確保程式終止時釋放攝影機
        print("伺服器已停止，攝影機已釋放。")