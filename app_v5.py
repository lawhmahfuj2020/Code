import cv2
import time
import requests
import threading
import base64
import numpy as np
from datetime import datetime
from collections import deque
from flask import Flask, Response, render_template_string, jsonify
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CAM1_URL       = "rtsp://admin:MTQUSN@146.196.49.41:554/ch1/main"
CAM2_URL       = "rtsp://admin:aaaa5555@146.196.49.41:5001/cam/realmonitor?channel=8&subtype=0"
BOT_TOKEN      = "8831097652:AAFluHl3A9c-mRFGg3yLBX2rQr-xBOMe8xc"
CHAT_ID        = "2052275350"
GEMINI_API_KEY = "AQ.Ab8RN6K8v8l1IoNUfHlTVXdXCoL4fS-iG6-3QrtBh68N4-j3RQ"
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

ALERT_COOLDOWN  = 120
GEMINI_COOLDOWN = 15
DETECT_EVERY    = 4

# ─── ZONES (NVR 960x1080) ─────────────────────────────────────────────────────
# Single cashier space — entire behind-counter area
CASHIER_SPACE = (0, 80, 390, 1080)   # cashier standing area
DRAWER_ZONE   = (0, 80,  390, 580)   # upper half = counter/drawer level
POCKET_ZONE   = (0, 500, 390, 1080)  # lower half = waist/pocket level

# Keypoint indices
YOLO_LEFT_WRIST  = 9
YOLO_RIGHT_WRIST = 10
MP_LEFT_WRIST    = 15
MP_RIGHT_WRIST   = 16

# ─── STATE ────────────────────────────────────────────────────────────────────
cam1_frame  = None
cam2_frame  = None
frame_lock1 = threading.Lock()
frame_lock2 = threading.Lock()
alert_lock  = threading.Lock()
alerts      = []
last_alert_time  = 0
last_gemini_time = 0

wrist_history = {
    "left":  deque(maxlen=40),
    "right": deque(maxlen=40),
}

system_status = {
    "cam1": "Connecting...",
    "cam2": "Connecting...",
    "persons": 0,
    "left_wrist":   "unknown",
    "right_wrist":  "unknown",
    "last_trigger": "None",
    "fps2": 0,
    "detection": "YOLO+MediaPipe",
}

app = Flask(__name__)

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
print("Loading YOLOv8n-pose...")
yolo_model = YOLO("yolov8n-pose.pt")
print("✓ YOLOv8n-pose loaded!")

print("Loading MediaPipe Pose...")
_mp_base = mp_python.BaseOptions(model_asset_path='/tmp/pose_landmarker.task')
_mp_opts  = mp_vision.PoseLandmarkerOptions(base_options=_mp_base)
mp_detector = mp_vision.PoseLandmarker.create_from_options(_mp_opts)
print("✓ MediaPipe Pose loaded!")

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def in_zone(x, y, zone):
    return zone[0] < x < zone[2] and zone[1] < y < zone[3]

def zone_name(x, y):
    if x == 0 and y == 0: return "UNKNOWN"
    if in_zone(x, y, DRAWER_ZONE):  return "DRAWER"
    if in_zone(x, y, POCKET_ZONE):  return "POCKET"
    if in_zone(x, y, CASHIER_SPACE): return "CASHIER"
    return "OTHER"

def encode_image(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')

def send_telegram(frame, caption):
    try:
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        requests.post(url,
            data={"chat_id": CHAT_ID, "caption": caption},
            files={"photo": ("alert.jpg", buf.tobytes(), "image/jpeg")},
            timeout=10)
        print(f"[TELEGRAM] Sent!")
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")

def add_alert(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with alert_lock:
        alerts.insert(0, {"time": ts, "msg": msg})
        if len(alerts) > 50: alerts.pop()

def ask_gemini(frame):
    try:
        img_b64 = encode_image(frame)
        prompt = (
            "This is a CCTV side-view of a shop cash counter. "
            "Is the cashier stealing money — taking cash from the drawer and putting it in their pocket or clothing? "
            "Reply ONLY: SUSPICIOUS or NORMAL"
        )
        payload = {"contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
        ]}]}
        r = requests.post(GEMINI_URL, json=payload, timeout=15)
        answer = r.json()['candidates'][0]['content']['parts'][0]['text'].strip().upper()
        print(f"[GEMINI] {answer}")
        return "SUSPICIOUS" in answer
    except Exception as e:
        print(f"[GEMINI ERROR] {e}")
        return False

def check_trajectory(history):
    """Check if wrist moved DRAWER → POCKET."""
    if len(history) < 8: return False
    positions = [zone_name(x, y) for x, y in history]
    had_drawer = False
    for pos in positions:
        if pos == "DRAWER": had_drawer = True
        if pos == "POCKET" and had_drawer: return True
    return False

def get_wrists_mediapipe(frame):
    """Get wrist positions using MediaPipe (more accurate)."""
    try:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = mp_detector.detect(mp_image)
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            lw = (lm[MP_LEFT_WRIST].x * w,  lm[MP_LEFT_WRIST].y * h)
            rw = (lm[MP_RIGHT_WRIST].x * w, lm[MP_RIGHT_WRIST].y * h)
            return lw, rw
    except: pass
    return None, None

def draw_overlay(frame, lw_zone, rw_zone, persons):
    h, w = frame.shape[:2]

    # Draw zones
    dcolor = (0, 200, 255)
    pcolor = (0, 120, 255)
    cv2.rectangle(frame, (DRAWER_ZONE[0], DRAWER_ZONE[1]),
                  (DRAWER_ZONE[2], DRAWER_ZONE[3]), dcolor, 2)
    cv2.putText(frame, "DRAWER", (DRAWER_ZONE[0]+5, DRAWER_ZONE[1]+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, dcolor, 2)
    cv2.rectangle(frame, (POCKET_ZONE[0], POCKET_ZONE[1]),
                  (POCKET_ZONE[2], POCKET_ZONE[3]), pcolor, 2)
    cv2.putText(frame, "POCKET", (POCKET_ZONE[0]+5, POCKET_ZONE[1]+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, pcolor, 2)

    # Status bar
    alert_active = lw_zone == "POCKET" or rw_zone == "POCKET"
    bar_color = (0, 0, 160) if alert_active else (0, 50, 0)
    cv2.rectangle(frame, (0, 0), (w, 40), bar_color, -1)
    ts = datetime.now().strftime("%H:%M:%S")
    txt = f"NVR Ch8 | {ts} | P:{persons} | LW:{lw_zone} RW:{rw_zone} | YOLO+MP"
    cv2.putText(frame, txt, (6, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    return frame

# ─── CAM1 LOOP ────────────────────────────────────────────────────────────────
def camera1_loop():
    global cam1_frame
    while True:
        try:
            cap = cv2.VideoCapture(CAM1_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                system_status["cam1"] = "Offline"
                time.sleep(10); continue
            system_status["cam1"] = "Online"
            print("[CAM1] Connected!")
            while True:
                ret, frame = cap.read()
                if not ret:
                    system_status["cam1"] = "Reconnecting..."
                    break
                ts = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, f"EZVIZ | {ts}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                with frame_lock1:
                    cam1_frame = frame.copy()
            cap.release()
        except Exception as e:
            system_status["cam1"] = "Error"
            print(f"[CAM1 ERROR] {e}")
            time.sleep(10)

# ─── CAM2 LOOP (AI Detection) ─────────────────────────────────────────────────
def camera2_loop():
    global cam2_frame, last_alert_time, last_gemini_time
    frame_count = 0
    t0 = time.time()

    while True:
        try:
            cap = cv2.VideoCapture(CAM2_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                system_status["cam2"] = "Offline"
                time.sleep(10); continue
            system_status["cam2"] = "Online"
            print("[CAM2] NVR Connected!")
            wrist_history["left"].clear()
            wrist_history["right"].clear()

            while True:
                ret, frame = cap.read()
                if not ret:
                    system_status["cam2"] = "Reconnecting..."
                    break

                frame_count += 1
                if frame_count % 30 == 0:
                    system_status["fps2"] = round(30 / (time.time() - t0), 1)
                    t0 = time.time()

                display = frame.copy()

                if frame_count % DETECT_EVERY == 0:
                    lw_x, lw_y = 0, 0
                    rw_x, rw_y = 0, 0
                    persons = 0
                    theft_detected = False
                    snap = frame.copy()

                    # ── Step 1: YOLO for person detection & filtering ──
                    yolo_results = yolo_model(frame, verbose=False, conf=0.35)
                    yr = yolo_results[0]

                    cashier_detected = False
                    if yr.boxes is not None and len(yr.boxes) > 0:
                        boxes = yr.boxes.xyxy.cpu().numpy()
                        for box in boxes:
                            bx1, by1, bx2, by2 = box
                            cx = (bx1 + bx2) / 2
                            # Cashier is on left side of frame
                            if cx < 450:
                                cashier_detected = True
                                persons += 1

                    system_status["persons"] = persons

                    if cashier_detected:
                        # ── Step 2: MediaPipe for accurate wrist positions ──
                        lw, rw = get_wrists_mediapipe(frame)

                        if lw: lw_x, lw_y = lw
                        if rw: rw_x, rw_y = rw

                        # Update history
                        if lw_x > 0: wrist_history["left"].append((lw_x, lw_y))
                        if rw_x > 0: wrist_history["right"].append((rw_x, rw_y))

                        lw_zone = zone_name(lw_x, lw_y)
                        rw_zone = zone_name(rw_x, rw_y)
                        system_status["left_wrist"]  = lw_zone
                        system_status["right_wrist"] = rw_zone

                        # Draw wrist dots
                        lcolor = (0,0,255) if lw_zone in ["POCKET","DRAWER"] else (0,255,0)
                        rcolor = (0,0,255) if rw_zone in ["POCKET","DRAWER"] else (0,255,0)
                        if lw_x > 0:
                            cv2.circle(display, (int(lw_x), int(lw_y)), 14, lcolor, -1)
                            cv2.putText(display, f"LW:{lw_zone}", (int(lw_x)+8, int(lw_y)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, lcolor, 2)
                        if rw_x > 0:
                            cv2.circle(display, (int(rw_x), int(rw_y)), 14, rcolor, -1)
                            cv2.putText(display, f"RW:{rw_zone}", (int(rw_x)+8, int(rw_y)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, rcolor, 2)

                        # Also draw YOLO skeleton
                        display = yolo_results[0].plot(img=display, boxes=False)

                        # ── Step 3: Check trajectory ──
                        if (check_trajectory(wrist_history["left"]) or
                            check_trajectory(wrist_history["right"]) or
                            lw_zone == "POCKET" or rw_zone == "POCKET"):
                            theft_detected = True

                    else:
                        system_status["left_wrist"]  = "unknown"
                        system_status["right_wrist"] = "unknown"

                    lw_zone = system_status["left_wrist"]
                    rw_zone = system_status["right_wrist"]
                    display = draw_overlay(display, lw_zone, rw_zone, persons)

                    # ── Alert logic ──
                    current_time = time.time()
                    if (theft_detected and
                        current_time - last_gemini_time > GEMINI_COOLDOWN):
                        last_gemini_time = current_time
                        system_status["last_trigger"] = datetime.now().strftime("%H:%M:%S")

                        def verify_and_alert(snapshot):
                            global last_alert_time
                            print("[DETECTION] Suspicious movement! Verifying with Gemini...")
                            is_suspicious = ask_gemini(snapshot)
                            if is_suspicious and time.time() - last_alert_time > ALERT_COOLDOWN:
                                last_alert_time = time.time()
                                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                caption = (
                                    f"🚨 THEFT ALERT!\n"
                                    f"Mamun Shop — {ts}\n"
                                    f"Hand moved: DRAWER → POCKET\n"
                                    f"Verified by Gemini AI ✓"
                                )
                                send_telegram(snapshot, caption)
                                add_alert(caption)
                                wrist_history["left"].clear()
                                wrist_history["right"].clear()

                        threading.Thread(target=verify_and_alert, args=(snap,), daemon=True).start()

                with frame_lock2:
                    cam2_frame = display.copy()

            cap.release()

        except Exception as e:
            system_status["cam2"] = "Error"
            print(f"[CAM2 ERROR] {e}")
            time.sleep(10)

# ─── FLASK ────────────────────────────────────────────────────────────────────
HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mamun Shop AI v5</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#080d0a;color:#e0e8e4;font-family:"Segoe UI",Arial,sans-serif}
.topbar{background:#0a140d;border-bottom:2px solid #1a4a2a;padding:10px 16px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
.brand{font-size:17px;font-weight:800}.brand .g{color:#00ff88}.brand .w{color:#fff}
.badges{display:flex;gap:6px}
.pill{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700}
.online{background:rgba(0,255,136,.12);color:#00ff88;border:1px solid rgba(0,255,136,.3)}
.offline{background:rgba(255,60,60,.12);color:#ff4444;border:1px solid rgba(255,60,60,.3)}
.content{padding:10px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}
.stat{background:#0d1a10;border:1px solid #1a3a22;border-radius:10px;padding:10px;text-align:center}
.stat-val{font-size:22px;font-weight:800;color:#00ff88;line-height:1}
.stat-lbl{font-size:9px;color:#2a6a3a;margin-top:4px;text-transform:uppercase;letter-spacing:1px}
.cam-card{background:#0d1a10;border:1px solid #1a3a22;border-radius:10px;overflow:hidden;margin-bottom:10px}
.cam-head{padding:8px 12px;display:flex;align-items:center;justify-content:space-between;background:#0a1208}
.cam-title{font-size:12px;font-weight:700;color:#00ff88}
.ai-badge{background:rgba(0,100,255,.2);color:#4488ff;border:1px solid rgba(0,100,255,.3);padding:2px 8px;border-radius:10px;font-size:10px;font-weight:700}
.rec{display:flex;align-items:center;gap:4px;font-size:9px;color:#ff4444;font-weight:700}
.rdot{width:7px;height:7px;background:#ff4444;border-radius:50%;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.cam-card img{width:100%;display:block}
.wrist-info{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:8px 12px;background:#080f09}
.wrist-box{background:#0d1a10;border-radius:6px;padding:6px;text-align:center}
.wrist-label{font-size:9px;color:#2a6a3a;text-transform:uppercase}
.wrist-val{font-size:12px;font-weight:800;margin-top:2px}
.drawer{color:#ffcc00}.pocket{color:#ff4444}.other{color:#888}.unknown{color:#555}
.cam-foot{padding:6px 12px;font-size:10px;color:#2a5a32;font-family:monospace;background:#080f09}
.alerts-box{background:#0d1a10;border:1px solid #1a3a22;border-radius:10px;overflow:hidden}
.alerts-head{padding:10px 14px;background:#0a1208;border-bottom:1px solid #1a3a22;display:flex;align-items:center;justify-content:space-between}
.alerts-title{font-size:13px;font-weight:800;color:#ff4444}
.alerts-count{background:rgba(255,68,68,.15);color:#ff4444;font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;border:1px solid rgba(255,68,68,.3)}
.alert-item{padding:10px 14px;border-left:3px solid #ff4444;margin:8px;background:#080f09;border-radius:0 8px 8px 0}
.alert-msg{font-size:11px;line-height:1.6;white-space:pre-line}
.alert-time{font-size:9px;color:#2a5a32;margin-top:3px;font-family:monospace}
.badge{font-size:9px;color:#4488ff;margin-top:2px;font-weight:600}
.empty{padding:24px;text-align:center;color:#1a3a22;font-size:12px}
</style>
</head>
<body>
<div class="topbar">
  <div class="brand"><span class="g">Mamun</span><span class="w">Shop</span> <span class="g" style="font-size:11px">AI v5</span></div>
  <div class="badges">
    <div class="pill {{cam1_cls}}" id="c1">CAM1: {{cam1}}</div>
    <div class="pill {{cam2_cls}}" id="c2">CAM2: {{cam2}}</div>
  </div>
</div>
<div class="content">
  <div class="stats">
    <div class="stat"><div class="stat-val" style="color:#ff4444" id="aC">{{alert_count}}</div><div class="stat-lbl">Alerts</div></div>
    <div class="stat"><div class="stat-val" id="pC">{{persons}}</div><div class="stat-lbl">Persons</div></div>
    <div class="stat"><div class="stat-val" id="fC" style="font-size:14px;padding-top:5px">{{fps}} fps</div><div class="stat-lbl">FPS</div></div>
    <div class="stat"><div class="stat-val" id="tC" style="font-size:11px;padding-top:5px">{{last_trigger}}</div><div class="stat-lbl">Last Trigger</div></div>
  </div>
  <div class="cam-card">
    <div class="cam-head">
      <div style="display:flex;align-items:center;gap:8px">
        <div class="cam-title">NVR Channel 8 — Side View</div>
        <div class="ai-badge">🤖 YOLO + MediaPipe</div>
      </div>
      <div class="rec"><div class="rdot"></div>LIVE</div>
    </div>
    <img src="/video2">
    <div class="wrist-info">
      <div class="wrist-box">
        <div class="wrist-label">Left Wrist</div>
        <div class="wrist-val {{lw_cls}}" id="lw">{{lw}}</div>
      </div>
      <div class="wrist-box">
        <div class="wrist-label">Right Wrist</div>
        <div class="wrist-val {{rw_cls}}" id="rw">{{rw}}</div>
      </div>
    </div>
    <div class="cam-foot">YOLOv8n-pose + MediaPipe Pose | Drawer→Pocket detection | Gemini AI verification</div>
  </div>
  <div class="cam-card">
    <div class="cam-head">
      <div class="cam-title">EZVIZ — Overhead View</div>
      <div class="rec"><div class="rdot"></div>LIVE</div>
    </div>
    <img src="/video1">
    <div class="cam-foot">Display only</div>
  </div>
  <div class="alerts-box" id="aBox">
    <div class="alerts-head">
      <div class="alerts-title">⚠ Alert Log</div>
      <div class="alerts-count" id="aBadge">{{alert_count}} total</div>
    </div>
    {%for a in alerts%}
    <div class="alert-item">
      <div class="alert-msg">{{a.msg}}</div>
      <div class="alert-time">{{a.time}}</div>
      <div class="badge">✓ Verified by Gemini AI</div>
    </div>
    {%endfor%}
    {%if not alerts%}<div class="empty">✅ No suspicious activity detected</div>{%endif%}
  </div>
</div>
<script>
function zc(z){if(!z)return'unknown';z=z.toUpperCase();if(z==='DRAWER')return'drawer';if(z==='POCKET')return'pocket';if(z==='UNKNOWN')return'unknown';return'other';}
setInterval(()=>{
  fetch('/status').then(r=>r.json()).then(d=>{
    document.getElementById('aC').textContent=d.alert_count;
    document.getElementById('aBadge').textContent=d.alert_count+' total';
    document.getElementById('pC').textContent=d.persons;
    document.getElementById('fC').textContent=d.fps+' fps';
    document.getElementById('tC').textContent=d.last_trigger;
    const lw=document.getElementById('lw');lw.textContent=d.left_wrist;lw.className='wrist-val '+zc(d.left_wrist);
    const rw=document.getElementById('rw');rw.textContent=d.right_wrist;rw.className='wrist-val '+zc(d.right_wrist);
    document.getElementById('c1').textContent='CAM1: '+d.cam1;document.getElementById('c1').className='pill '+(d.cam1==='Online'?'online':'offline');
    document.getElementById('c2').textContent='CAM2: '+d.cam2;document.getElementById('c2').className='pill '+(d.cam2==='Online'?'online':'offline');
    if(d.alerts&&d.alerts.length>0){
      const h=document.querySelector('.alerts-head').outerHTML;
      const items=d.alerts.map(a=>"<div class='alert-item'><div class='alert-msg'>"+a.msg+"</div><div class='alert-time'>"+a.time+"</div><div class='badge'>✓ Verified by Gemini AI</div></div>").join('');
      document.getElementById('aBox').innerHTML=h+items;
    }
  });
},2000);
</script>
</body>
</html>'''

cam1_ref = [None]
cam2_ref = [None]

def sync_frames():
    while True:
        with frame_lock1: cam1_ref[0] = cam1_frame
        with frame_lock2: cam2_ref[0] = cam2_frame
        time.sleep(0.03)

@app.route('/')
def index():
    with alert_lock: al = list(alerts)
    lw = system_status["left_wrist"]
    rw = system_status["right_wrist"]
    def cls(z):
        if z=="DRAWER": return "drawer"
        if z=="POCKET": return "pocket"
        if z=="unknown": return "unknown"
        return "other"
    return render_template_string(HTML,
        cam1=system_status["cam1"], cam2=system_status["cam2"],
        cam1_cls="online" if system_status["cam1"]=="Online" else "offline",
        cam2_cls="online" if system_status["cam2"]=="Online" else "offline",
        persons=system_status["persons"],
        fps=system_status["fps2"],
        last_trigger=system_status["last_trigger"],
        lw=lw, rw=rw, lw_cls=cls(lw), rw_cls=cls(rw),
        alerts=al, alert_count=len(al))

@app.route('/status')
def status():
    with alert_lock: al = list(alerts[:20])
    return jsonify({
        "cam1": system_status["cam1"], "cam2": system_status["cam2"],
        "persons": system_status["persons"],
        "left_wrist": system_status["left_wrist"],
        "right_wrist": system_status["right_wrist"],
        "last_trigger": system_status["last_trigger"],
        "fps": system_status["fps2"],
        "alerts": al, "alert_count": len(alerts),
    })

@app.route('/video1')
def video1():
    def gen():
        while True:
            f = cam1_ref[0]
            if f is not None:
                ret, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video2')
def video2():
    def gen():
        while True:
            f = cam2_ref[0]
            if f is not None:
                ret, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Thread(target=camera1_loop, daemon=True).start()
    threading.Thread(target=camera2_loop, daemon=True).start()
    threading.Thread(target=sync_frames,  daemon=True).start()
    app.run(host='0.0.0.0', port=8080, threaded=True)
