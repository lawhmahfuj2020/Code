import cv2
import time
import requests
import threading
import base64
import numpy as np
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CAM1_URL = "rtsp://admin:MTQUSN@146.196.49.41:554/ch1/main"
CAM2_URL = "rtsp://admin:aaaa5555@146.196.49.41:5554/cam/realmonitor?channel=8&subtype=0"
BOT_TOKEN = "8831097652:AAFluHl3A9c-mRFGg3yLBX2rQr-xBOMe8xc"
CHAT_ID = "2052275350"
GEMINI_API_KEY = "AQ.Ab8RN6K8v8l1IoNUfHlTVXdXCoL4fS-iG6-3QrtBh68N4-j3RQ"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

ALERT_COOLDOWN  = 120
GEMINI_COOLDOWN = 30
DETECT_EVERY    = 8

# ─── ZONES (1920x1080) ────────────────────────────────────────────────────────
CUSTOMER_ZONE = (220, 50, 1350, 370)
LEFT_DRAWER   = (500, 480, 720, 720)
RIGHT_DRAWER  = (720, 480, 940, 720)
LEFT_WRIST    = 9
RIGHT_WRIST   = 10

# ─── STATE ────────────────────────────────────────────────────────────────────
cam1_frame = None
cam2_frame = None
cam1_status = "Connecting..."
cam2_status = "Connecting..."
alerts = []
frame_lock1 = threading.Lock()
frame_lock2 = threading.Lock()
alert_lock  = threading.Lock()
last_alert_time = 0

app = Flask(__name__)

# ─── LOAD ALL MODELS ──────────────────────────────────────────────────────────
print("Loading AI models...")

# Person detection - YOLOv10s
person_model = YOLO("yolov10s.pt")
print("✓ YOLOv10s person detection loaded")

# Pose detection - YOLOv8x-pose (most accurate)
pose_model = YOLO("yolov8x-pose.pt")
print("✓ YOLOv8x-pose loaded")

# Object detection - YOLOv8x (detects cash, bags, phones)
object_model = YOLO("yolov8x.pt")
print("✓ YOLOv8x object detection loaded")

# MediaPipe hand landmarker
base_options = mp_python.BaseOptions(model_asset_path='/root/hand_landmarker.task')
hand_options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
print("✓ MediaPipe Hand Landmarker loaded")

print("All models loaded! Starting system...")

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def is_in_zone(x, y, zone):
    return zone[0] < x < zone[2] and zone[1] < y < zone[3]

def encode_image(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode('utf-8')

def send_telegram_photo(img_path, caption):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        with open(img_path, 'rb') as photo:
            requests.post(url, data={"chat_id": CHAT_ID, "caption": caption},
                          files={"photo": photo}, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

def add_alert(msg, level="THREAT"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with alert_lock:
        alerts.insert(0, {"time": ts, "msg": msg, "level": level})
        if len(alerts) > 50:
            alerts.pop()

def ask_gemini(frame, customer_present, extra_context="", camera_name="Camera"):
    try:
        img_b64 = encode_image(frame)
        question = (
            f"Security camera ({camera_name}) top-down view of cash counter. "
            f"{extra_context} "
            f"{'Customer was recently present.' if customer_present else 'NO customer present.'} "
            "Is this NORMAL or SUSPICIOUS cash handling? "
            "Reply ONLY: NORMAL or SUSPICIOUS"
        )
        payload = {"contents": [{"parts": [
            {"text": question},
            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
        ]}]}
        r = requests.post(GEMINI_URL, json=payload, timeout=15)
        answer = r.json()['candidates'][0]['content']['parts'][0]['text'].strip().upper()
        return "SUSPICIOUS" in answer
    except Exception as e:
        print(f"Gemini error: {e}")
        return False

def draw_zones(frame):
    h, w = frame.shape[:2]
    # Customer zone
    cv2.rectangle(frame, (CUSTOMER_ZONE[0], CUSTOMER_ZONE[1]),
                  (CUSTOMER_ZONE[2], CUSTOMER_ZONE[3]), (0,255,255), 2)
    cv2.putText(frame, "CUSTOMER ZONE",
                (CUSTOMER_ZONE[0]+5, CUSTOMER_ZONE[1]+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    # Cash zones
    cv2.rectangle(frame, (LEFT_DRAWER[0], LEFT_DRAWER[1]),
                  (LEFT_DRAWER[2], LEFT_DRAWER[3]), (0,0,255), 2)
    cv2.putText(frame, "CASH L",
                (LEFT_DRAWER[0]+5, LEFT_DRAWER[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.rectangle(frame, (RIGHT_DRAWER[0], RIGHT_DRAWER[1]),
                  (RIGHT_DRAWER[2], RIGHT_DRAWER[3]), (0,0,255), 2)
    cv2.putText(frame, "CASH R",
                (RIGHT_DRAWER[0]+5, RIGHT_DRAWER[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    # Timestamp
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w-100, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

def run_mediapipe_hands(frame):
    """Run MediaPipe hand landmarker and return hand positions"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect(mp_image)
    hand_points = []
    h, w = frame.shape[:2]
    if result.hand_landmarks:
        for hand_landmark in result.hand_landmarks:
            for lm in hand_landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                hand_points.append((x, y))
            # Draw all 21 finger points
            for lm in hand_landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (255, 0, 255), -1)
    return hand_points

# ─── CAMERA 1 LOOP ────────────────────────────────────────────────────────────
def camera1_loop():
    global cam1_frame, cam1_status, last_alert_time
    last_gemini_check = 0
    customer_last_seen = 0
    frame_count = 0

    while True:
        try:
            cap = cv2.VideoCapture(CAM1_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            if not cap.isOpened():
                cam1_status = "Offline"
                time.sleep(10)
                continue

            cam1_status = "Online"
            print("CAM1 connected!")

            while True:
                ret, frame = cap.read()
                if not ret:
                    cam1_status = "Offline"
                    break

                frame_count += 1
                current_time = time.time()
                draw_zones(frame)

                if frame_count % DETECT_EVERY == 0:
                    customer_present = False
                    hand_in_drawer = False
                    suspicious_objects = []
                    extra_context = ""

                    # ── 1. YOLOv10s Person Detection ──
                    person_results = person_model(frame, verbose=False, conf=0.5, imgsz=640)
                    for r in person_results:
                        for box in r.boxes:
                            if int(box.cls[0]) == 0:
                                x1,y1,x2,y2 = map(int, box.xyxy[0])
                                cx, cy = (x1+x2)//2, (y1+y2)//2
                                conf = float(box.conf[0])
                                if is_in_zone(cx, cy, CUSTOMER_ZONE):
                                    customer_present = True
                                    customer_last_seen = current_time
                                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                                    cv2.putText(frame,f"CUSTOMER {conf:.0%}",(x1,y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

                    # ── 2. YOLOv8x Object Detection ──
                    object_results = object_model(frame, verbose=False, conf=0.4, imgsz=640)
                    suspicious_classes = ['handbag', 'backpack', 'suitcase', 'cell phone', 'wallet']
                    for r in object_results:
                        for box in r.boxes:
                            cls_name = object_model.names[int(box.cls[0])]
                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            cx, cy = (x1+x2)//2, (y1+y2)//2
                            if cls_name in suspicious_classes:
                                if (is_in_zone(cx,cy,LEFT_DRAWER) or
                                    is_in_zone(cx,cy,RIGHT_DRAWER)):
                                    suspicious_objects.append(cls_name)
                                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,165,0),2)
                                    cv2.putText(frame,f"⚠ {cls_name}",(x1,y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,165,0),2)
                                    extra_context += f"{cls_name} detected near cash. "

                    # ── 3. YOLOv8x-pose Body Tracking ──
                    pose_results = pose_model(frame, verbose=False, conf=0.4, imgsz=640)
                    for r in pose_results:
                        if r.keypoints is not None:
                            for kp in r.keypoints:
                                kp_data = kp.data[0]
                                if len(kp_data) > RIGHT_WRIST:
                                    lw = kp_data[LEFT_WRIST]
                                    rw = kp_data[RIGHT_WRIST]
                                    lx,ly = int(lw[0]),int(lw[1])
                                    rx,ry = int(rw[0]),int(rw[1])
                                    if lx>0 and ly>0:
                                        cv2.circle(frame,(lx,ly),12,(0,255,255),-1)
                                        cv2.putText(frame,"LW",(lx-10,ly+5),
                                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                                    if rx>0 and ry>0:
                                        cv2.circle(frame,(rx,ry),12,(0,255,255),-1)
                                        cv2.putText(frame,"RW",(rx-10,ry+5),
                                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                                    if (is_in_zone(lx,ly,LEFT_DRAWER) or
                                        is_in_zone(rx,ry,LEFT_DRAWER) or
                                        is_in_zone(lx,ly,RIGHT_DRAWER) or
                                        is_in_zone(rx,ry,RIGHT_DRAWER)):
                                        hand_in_drawer = True
                                        extra_context += "Wrist detected in cash zone. "
                                        h2,w2 = frame.shape[:2]
                                        cv2.putText(frame,"⚠ DRAWER ACCESSED!",
                                            (w2//2-200,60),
                                            cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)

                    # ── 4. MediaPipe Finger Tracking ──
                    try:
                        hand_points = run_mediapipe_hands(frame)
                        for (hx, hy) in hand_points:
                            if (is_in_zone(hx,hy,LEFT_DRAWER) or
                                is_in_zone(hx,hy,RIGHT_DRAWER)):
                                hand_in_drawer = True
                                extra_context += "Finger detected in cash zone. "
                    except Exception as e:
                        pass

                    # ── 5. Gemini AI Verification ──
                    recent_customer = (current_time - customer_last_seen) < 60
                    trigger = hand_in_drawer or len(suspicious_objects) > 0

                    if trigger and current_time - last_gemini_check > GEMINI_COOLDOWN:
                        last_gemini_check = current_time
                        snapshot = frame.copy()

                        def check_gemini(snap, cust, ctx):
                            global last_alert_time
                            is_sus = ask_gemini(snap, cust, ctx, "CAM1-EZVIZ")
                            if is_sus and time.time() - last_alert_time > ALERT_COOLDOWN:
                                img_path = "/root/alert_cam1.jpg"
                                cv2.imwrite(img_path, snap)
                                ts2 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                caption = (
                                    f"🚨 THEFT ALERT!\nMamuns Shop CAM1\n{ts2}\n{ctx}"
                                    if not cust else
                                    f"🚨 SUSPICIOUS ACTIVITY!\nMamuns Shop CAM1\n{ts2}\n{ctx}"
                                )
                                send_telegram_photo(img_path, caption)
                                add_alert(f"[CAM1] {caption}", "THREAT")
                                last_alert_time = time.time()

                        threading.Thread(target=check_gemini,
                            args=(snapshot, recent_customer, extra_context),
                            daemon=True).start()

                with frame_lock1:
                    cam1_frame = frame.copy()

            cap.release()

        except Exception as e:
            cam1_status = "Error"
            print(f"CAM1 error: {e}")
            time.sleep(10)

# ─── CAMERA 2 LOOP ────────────────────────────────────────────────────────────
def camera2_loop():
    global cam2_frame, cam2_status

    while True:
        try:
            cap = cv2.VideoCapture(CAM2_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            if not cap.isOpened():
                cam2_status = "Offline"
                time.sleep(10)
                continue

            cam2_status = "Online"
            print("CAM2 connected!")

            while True:
                ret, frame = cap.read()
                if not ret:
                    cam2_status = "Offline"
                    break

                h, w = frame.shape[:2]
                ts = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, f"CAM2 | {ts}", (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)

                with frame_lock2:
                    cam2_frame = frame.copy()

            cap.release()

        except Exception as e:
            cam2_status = "Error"
            print(f"CAM2 error: {e}")
            time.sleep(10)

# ─── STREAM GENERATORS ────────────────────────────────────────────────────────
def generate_cam1():
    while True:
        with frame_lock1:
            frame = cam1_frame.copy() if cam1_frame is not None else None
        if frame is not None:
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
        time.sleep(0.033)

def generate_cam2():
    while True:
        with frame_lock2:
            frame = cam2_frame.copy() if cam2_frame is not None else None
        if frame is not None:
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
        time.sleep(0.033)

# ─── DASHBOARD ────────────────────────────────────────────────────────────────
HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>Mamun Shop AI Security</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#080d0a;color:#e0e8e4;font-family:"Segoe UI",Arial,sans-serif;min-height:100vh}
.topbar{background:#0a140d;border-bottom:2px solid #1a4a2a;padding:10px 16px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
.brand{font-size:17px;font-weight:800;letter-spacing:1px}
.brand .g{color:#00ff88}.brand .w{color:#fff}
.pills{display:flex;gap:6px;flex-wrap:wrap}
.pill{padding:3px 10px;border-radius:20px;font-size:10px;font-weight:700}
.online{background:rgba(0,255,136,.12);color:#00ff88;border:1px solid rgba(0,255,136,.3)}
.offline{background:rgba(255,60,60,.12);color:#ff4444;border:1px solid rgba(255,60,60,.3)}
.content{padding:10px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}
.stat{background:#0d1a10;border:1px solid #1a3a22;border-radius:10px;padding:8px;text-align:center}
.stat-val{font-size:20px;font-weight:800;color:#00ff88;line-height:1}
.stat-lbl{font-size:8px;color:#2a6a3a;margin-top:3px;text-transform:uppercase;letter-spacing:1px}
.cameras{display:grid;grid-template-columns:1fr;gap:8px;margin-bottom:10px}
.cam-card{background:#0d1a10;border:1px solid #1a3a22;border-radius:10px;overflow:hidden}
.cam-head{padding:7px 10px;display:flex;align-items:center;justify-content:space-between;background:#0a1208}
.cam-title{font-size:11px;font-weight:700;color:#00ff88}
.rec{display:flex;align-items:center;gap:4px;font-size:9px;color:#ff4444;font-weight:700}
.rdot{width:6px;height:6px;background:#ff4444;border-radius:50%;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.cam-card img{width:100%;display:block;background:#050a06}
.cam-foot{padding:5px 10px;font-size:9px;color:#2a5a32;font-family:monospace;background:#080f09}
.alerts-box{background:#0d1a10;border:1px solid #1a3a22;border-radius:10px;overflow:hidden}
.alerts-head{padding:10px 14px;background:#0a1208;border-bottom:1px solid #1a3a22;display:flex;align-items:center;justify-content:space-between}
.alerts-title{font-size:13px;font-weight:800;color:#ff4444}
.alerts-count{background:rgba(255,68,68,.15);color:#ff4444;font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;border:1px solid rgba(255,68,68,.3)}
.alert-item{padding:10px 14px;border-left:3px solid #ff4444;margin:8px;background:#080f09;border-radius:0 8px 8px 0}
.alert-msg{font-size:11px;line-height:1.5}
.alert-time{font-size:9px;color:#2a5a32;margin-top:3px;font-family:monospace}
.badge{font-size:9px;color:#4488ff;margin-top:2px;font-weight:600}
.empty{padding:24px;text-align:center;color:#1a3a22;font-size:12px}
.models{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-bottom:10px}
.model-pill{background:#0d1a10;border:1px solid #1a3a22;border-radius:8px;padding:6px 10px;font-size:9px;font-family:monospace;color:#00ff88;display:flex;align-items:center;gap:5px}
.model-pill::before{content:"✓";color:#00ff88;font-weight:700}
</style>
</head>
<body>
<div class="topbar">
  <div class="brand"><span class="g">Mamun</span><span class="w">Shop</span> <span class="g" style="font-size:11px">AI v2</span></div>
  <div class="pills">
    <div class="pill {{ cam1_cls }}">CAM1: {{ cam1_status }}</div>
    <div class="pill {{ cam2_cls }}">CAM2: {{ cam2_status }}</div>
  </div>
</div>
<div class="content">
  <div class="stats">
    <div class="stat"><div class="stat-val" style="color:#ff4444" id="alertCount">{{ alert_count }}</div><div class="stat-lbl">Alerts</div></div>
    <div class="stat"><div class="stat-val">2</div><div class="stat-lbl">Cameras</div></div>
    <div class="stat"><div class="stat-val">4</div><div class="stat-lbl">AI Models</div></div>
    <div class="stat"><div class="stat-val" style="font-size:12px;padding-top:4px">24/7</div><div class="stat-lbl">Active</div></div>
  </div>
  <div class="models">
    <div class="model-pill">YOLOv10s Detection</div>
    <div class="model-pill">YOLOv8x-pose</div>
    <div class="model-pill">YOLOv8x Objects</div>
    <div class="model-pill">MediaPipe Hands</div>
  </div>
  <div class="cameras">
    <div class="cam-card">
      <div class="cam-head">
        <div class="cam-title">CAM1 — EZVIZ (Full AI)</div>
        <div class="rec"><div class="rdot"></div>LIVE AI</div>
      </div>
      <img src="/video1" alt="Camera 1">
      <div class="cam-foot">YOLOv10s + YOLOv8x-pose + YOLOv8x + MediaPipe + Gemini</div>
    </div>
    <div class="cam-card">
      <div class="cam-head">
        <div class="cam-title">CAM2 — NVR Ch8</div>
        <div class="rec"><div class="rdot"></div>LIVE</div>
      </div>
      <img src="/video2" alt="Camera 2">
      <div class="cam-foot">Live monitoring active</div>
    </div>
  </div>
  <div class="alerts-box" id="alertsBox">
    <div class="alerts-head">
      <div class="alerts-title">⚠ Alert Log</div>
      <div class="alerts-count" id="alertBadge">{{ alert_count }} total</div>
    </div>
    {% for alert in alerts %}
    <div class="alert-item">
      <div class="alert-msg">{{ alert.msg }}</div>
      <div class="alert-time">{{ alert.time }}</div>
      <div class="badge">✓ Verified by Gemini AI</div>
    </div>
    {% endfor %}
    {% if not alerts %}
    <div class="empty">✅ No suspicious activity detected</div>
    {% endif %}
  </div>
</div>
<script>
setInterval(()=>{
  fetch("/alerts").then(r=>r.json()).then(data=>{
    document.getElementById("alertCount").textContent = data.count;
    document.getElementById("alertBadge").textContent = data.count+" total";
    var box = document.getElementById("alertsBox");
    var head = box.querySelector(".alerts-head").outerHTML;
    var items = data.alerts.length ? data.alerts.map(a=>
      "<div class='alert-item'><div class='alert-msg'>"+a.msg+"</div><div class='alert-time'>"+a.time+"</div><div class='badge'>✓ Verified by Gemini AI</div></div>"
    ).join("") : "<div class='empty'>✅ No suspicious activity detected</div>";
    box.innerHTML = head + items;
  });
}, 10000);
</script>
</body>
</html>'''

@app.route('/')
def index():
    c1cls = 'online' if cam1_status == 'Online' else 'offline'
    c2cls = 'online' if cam2_status == 'Online' else 'offline'
    return render_template_string(HTML,
        cam1_status=cam1_status, cam2_status=cam2_status,
        cam1_cls=c1cls, cam2_cls=c2cls,
        alerts=alerts, alert_count=len(alerts))

@app.route('/alerts')
def get_alerts():
    with alert_lock:
        return jsonify({"alerts": alerts[:20], "count": len(alerts)})

@app.route('/video1')
def video1():
    return Response(generate_cam1(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video2')
def video2():
    return Response(generate_cam2(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Thread(target=camera1_loop, daemon=True).start()
    threading.Thread(target=camera2_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=8080, threaded=True)
