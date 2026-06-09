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

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CAM1_URL       = "rtsp://admin:MTQUSN@146.196.49.41:554/ch1/main"
CAM2_URL       = "rtsp://admin:aaaa5555@146.196.49.41:5001/cam/realmonitor?channel=8&subtype=0"
BOT_TOKEN      = "8831097652:AAFluHl3A9c-mRFGg3yLBX2rQr-xBOMe8xc"
CHAT_ID        = "2052275350"
GEMINI_API_KEY = "AQ.Ab8RN6K8v8l1IoNUfHlTVXdXCoL4fS-iG6-3QrtBh68N4-j3RQ"
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

ALERT_COOLDOWN  = 120   # seconds between alerts
GEMINI_COOLDOWN = 15    # seconds between Gemini calls
DETECT_EVERY    = 5     # run pose every N frames

# ─── ZONES for NVR Channel 8 (960x1080) ──────────────────────────────────────
# Based on pose_test.jpg analysis:
# Cashier stands on LEFT side of frame (x: 0-400)
# Drawer area is at counter level (y: 400-650)
# Pocket area is at waist/hip level (y: 600-900, x: 0-350)
# Customer zone is RIGHT side / background (x: 400-960)

DRAWER_ZONE  = (0,   350, 450, 620)   # where hands go to get cash
POCKET_ZONE  = (0,   580, 380, 950)   # where hand goes when pocketing
CASHIER_ZONE = (0,   200, 450, 1080)  # cashier standing area (left side)
CUSTOMER_ZONE= (380, 0,   960, 800)   # customer area (right side)

# Keypoint indices (YOLOv8-pose COCO format)
LEFT_WRIST  = 9
RIGHT_WRIST = 10
LEFT_HIP    = 11
RIGHT_HIP   = 12

# ─── STATE ────────────────────────────────────────────────────────────────────
cam1_frame   = None
cam2_frame   = None
cam1_status  = "Connecting..."
cam2_status  = "Connecting..."
alerts       = []
frame_lock1  = threading.Lock()
frame_lock2  = threading.Lock()
alert_lock   = threading.Lock()
last_alert_time  = 0
last_gemini_time = 0

# Wrist trajectory tracking — store last 30 positions per wrist
wrist_history = {
    "left":  deque(maxlen=30),
    "right": deque(maxlen=30),
}

system_status = {
    "cam1": "Connecting...",
    "cam2": "Connecting...",
    "persons": 0,
    "left_wrist":  "unknown",
    "right_wrist": "unknown",
    "last_trigger": "None",
    "fps2": 0,
}

app = Flask(__name__)

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
print("Loading YOLOv8n-pose...")
pose_model = YOLO("yolov8n-pose.pt")
print("✓ Model loaded!")

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def in_zone(x, y, zone):
    return zone[0] < x < zone[2] and zone[1] < y < zone[3]

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
        print(f"[TELEGRAM] Sent: {caption[:60]}")
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")

def add_alert(msg, level="THREAT"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with alert_lock:
        alerts.insert(0, {"time": ts, "msg": msg, "level": level})
        if len(alerts) > 50:
            alerts.pop()

def ask_gemini(frame):
    try:
        img_b64 = encode_image(frame)
        prompt = (
            "This is a CCTV side-view image of a shop cash counter. "
            "Look carefully at the cashier's hands. "
            "Is the cashier moving money from the cash drawer into their own pocket or clothing? "
            "Reply with ONLY one word: SUSPICIOUS or NORMAL"
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

def zone_name(x, y):
    if in_zone(x, y, DRAWER_ZONE):  return "DRAWER"
    if in_zone(x, y, POCKET_ZONE):  return "POCKET"
    if in_zone(x, y, CASHIER_ZONE): return "CASHIER_AREA"
    return "OTHER"

def check_theft_trajectory(history):
    """Check if wrist moved from DRAWER → POCKET in recent history."""
    if len(history) < 10:
        return False
    positions = [zone_name(x, y) for x, y in history]
    # Look for DRAWER appearing before POCKET
    had_drawer = False
    for pos in positions:
        if pos == "DRAWER":
            had_drawer = True
        if pos == "POCKET" and had_drawer:
            return True
    return False

def draw_zones(frame):
    """Draw detection zones on frame."""
    # Drawer zone - yellow
    cv2.rectangle(frame, (DRAWER_ZONE[0], DRAWER_ZONE[1]),
                  (DRAWER_ZONE[2], DRAWER_ZONE[3]), (0, 255, 255), 2)
    cv2.putText(frame, "DRAWER", (DRAWER_ZONE[0]+5, DRAWER_ZONE[1]+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Pocket zone - orange
    cv2.rectangle(frame, (POCKET_ZONE[0], POCKET_ZONE[1]),
                  (POCKET_ZONE[2], POCKET_ZONE[3]), (0, 165, 255), 2)
    cv2.putText(frame, "POCKET", (POCKET_ZONE[0]+5, POCKET_ZONE[1]+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    return frame

# ─── CAM1 LOOP (EZVIZ - display only) ────────────────────────────────────────
def camera1_loop():
    global cam1_frame, cam1_status
    while True:
        try:
            cap = cv2.VideoCapture(CAM1_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                cam1_status = "Offline"
                system_status["cam1"] = "Offline"
                time.sleep(10)
                continue
            cam1_status = "Online"
            system_status["cam1"] = "Online"
            print("[CAM1] Connected!")
            while True:
                ret, frame = cap.read()
                if not ret:
                    cam1_status = "Reconnecting..."
                    system_status["cam1"] = "Reconnecting..."
                    break
                # Add timestamp
                ts = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, f"CAM1 EZVIZ | {ts}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                with frame_lock1:
                    cam1_frame = frame.copy()
            cap.release()
        except Exception as e:
            system_status["cam1"] = "Error"
            print(f"[CAM1 ERROR] {e}")
            time.sleep(10)

# ─── CAM2 LOOP (NVR Channel 8 - AI Detection) ────────────────────────────────
def camera2_loop():
    global cam2_frame, cam2_status, last_alert_time, last_gemini_time

    frame_count = 0
    t0 = time.time()

    while True:
        try:
            cap = cv2.VideoCapture(CAM2_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                cam2_status = "Offline"
                system_status["cam2"] = "Offline"
                time.sleep(10)
                continue
            cam2_status = "Online"
            system_status["cam2"] = "Online"
            print("[CAM2] NVR Connected!")

            # Reset wrist history on reconnect
            wrist_history["left"].clear()
            wrist_history["right"].clear()

            while True:
                ret, frame = cap.read()
                if not ret:
                    cam2_status = "Reconnecting..."
                    system_status["cam2"] = "Reconnecting..."
                    break

                frame_count += 1

                # FPS tracking
                if frame_count % 30 == 0:
                    elapsed = time.time() - t0
                    system_status["fps2"] = round(30 / elapsed, 1)
                    t0 = time.time()

                display = frame.copy()
                display = draw_zones(display)

                # Run pose detection every DETECT_EVERY frames
                if frame_count % DETECT_EVERY == 0:
                    results = pose_model(frame, verbose=False, conf=0.35)
                    r = results[0]

                    persons = 0
                    theft_detected = False
                    alert_snap = frame.copy()

                    if r.keypoints is not None and len(r.keypoints.xy) > 0:
                        kpts_list = r.keypoints.xy.cpu().numpy()
                        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []

                        for idx, kpts in enumerate(kpts_list):
                            if len(kpts) < 17:
                                continue

                            # Get box center to identify cashier vs customer
                            if len(boxes) > idx:
                                bx1, by1, bx2, by2 = boxes[idx]
                                cx = (bx1 + bx2) / 2
                                # Only process cashier (left side of frame)
                                if cx > 450:
                                    continue  # skip customers on right side

                            persons += 1

                            lw_x, lw_y = kpts[LEFT_WRIST]
                            rw_x, rw_y = kpts[RIGHT_WRIST]

                            # Update wrist history (skip zero/undetected points)
                            if lw_x > 0 and lw_y > 0:
                                wrist_history["left"].append((lw_x, lw_y))
                            if rw_x > 0 and rw_y > 0:
                                wrist_history["right"].append((rw_x, rw_y))

                            # Get zone names
                            lw_zone = zone_name(lw_x, lw_y)
                            rw_zone = zone_name(rw_x, rw_y)
                            system_status["left_wrist"]  = lw_zone
                            system_status["right_wrist"] = rw_zone

                            # Draw wrists on display
                            lw_color = (0, 0, 255) if lw_zone in ["DRAWER","POCKET"] else (0, 255, 0)
                            rw_color = (0, 0, 255) if rw_zone in ["DRAWER","POCKET"] else (0, 255, 0)
                            if lw_x > 0:
                                cv2.circle(display, (int(lw_x), int(lw_y)), 12, lw_color, -1)
                                cv2.putText(display, f"LW:{lw_zone}", (int(lw_x)+5, int(lw_y)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, lw_color, 2)
                            if rw_x > 0:
                                cv2.circle(display, (int(rw_x), int(rw_y)), 12, rw_color, -1)
                                cv2.putText(display, f"RW:{rw_zone}", (int(rw_x)+5, int(rw_y)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, rw_color, 2)

                            # Draw skeleton
                            display = results[0].plot(img=display, boxes=False)

                            # Check trajectory for theft
                            if (check_theft_trajectory(wrist_history["left"]) or
                                check_theft_trajectory(wrist_history["right"])):
                                theft_detected = True

                    system_status["persons"] = persons

                    # ── Alert logic ──
                    current_time = time.time()
                    if (theft_detected and
                        current_time - last_gemini_time > GEMINI_COOLDOWN):

                        last_gemini_time = current_time
                        snap = alert_snap.copy()
                        system_status["last_trigger"] = datetime.now().strftime("%H:%M:%S")

                        def verify_and_alert(snapshot):
                            global last_alert_time
                            print("[DETECTION] DRAWER→POCKET trajectory detected! Verifying...")
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
                                add_alert(caption, "THREAT")
                                # Reset history after alert
                                wrist_history["left"].clear()
                                wrist_history["right"].clear()

                        threading.Thread(target=verify_and_alert, args=(snap,), daemon=True).start()

                # Status bar
                h, w = display.shape[:2]
                bar_color = (0, 0, 150) if system_status.get("last_trigger","None") != "None" else (0, 60, 0)
                cv2.rectangle(display, (0, 0), (w, 42), bar_color, -1)
                ts = datetime.now().strftime("%H:%M:%S")
                txt = f"NVR Ch8 AI | {ts} | Persons:{system_status['persons']} | LW:{system_status['left_wrist']} RW:{system_status['right_wrist']}"
                cv2.putText(display, txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                with frame_lock2:
                    cam2_frame = display.copy()

            cap.release()

        except Exception as e:
            system_status["cam2"] = "Error"
            print(f"[CAM2 ERROR] {e}")
            time.sleep(10)

# ─── FLASK STREAMS ────────────────────────────────────────────────────────────
def gen_stream(lock, frame_ref):
    while True:
        with lock:
            frame = frame_ref[0]
        if frame is not None:
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
        time.sleep(0.05)

# ─── DASHBOARD ───────────────────────────────────────────────────────────────
HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mamun Shop AI v4</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#080d0a;color:#e0e8e4;font-family:"Segoe UI",Arial,sans-serif}
.topbar{background:#0a140d;border-bottom:2px solid #1a4a2a;padding:10px 16px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
.brand{font-size:17px;font-weight:800;letter-spacing:1px}
.brand .g{color:#00ff88}.brand .w{color:#fff}
.badges{display:flex;gap:6px}
.pill{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700}
.online{background:rgba(0,255,136,.12);color:#00ff88;border:1px solid rgba(0,255,136,.3)}
.offline{background:rgba(255,60,60,.12);color:#ff4444;border:1px solid rgba(255,60,60,.3)}
.content{padding:10px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}
.stat{background:#0d1a10;border:1px solid #1a3a22;border-radius:10px;padding:10px;text-align:center}
.stat-val{font-size:22px;font-weight:800;color:#00ff88;line-height:1}
.stat-lbl{font-size:9px;color:#2a6a3a;margin-top:4px;text-transform:uppercase;letter-spacing:1px}
.cams{display:grid;grid-template-columns:1fr;gap:10px;margin-bottom:10px}
.cam-card{background:#0d1a10;border:1px solid #1a3a22;border-radius:10px;overflow:hidden}
.cam-head{padding:8px 12px;display:flex;align-items:center;justify-content:space-between;background:#0a1208}
.cam-title{font-size:12px;font-weight:700;color:#00ff88}
.cam-badge{font-size:10px;padding:2px 8px;border-radius:10px;font-weight:700}
.ai-badge{background:rgba(0,100,255,.2);color:#4488ff;border:1px solid rgba(0,100,255,.3)}
.rec{display:flex;align-items:center;gap:4px;font-size:9px;color:#ff4444;font-weight:700}
.rdot{width:7px;height:7px;background:#ff4444;border-radius:50%;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.cam-card img{width:100%;display:block}
.cam-foot{padding:6px 12px;font-size:10px;color:#2a5a32;font-family:monospace;background:#080f09}
.wrist-info{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:8px 12px;background:#080f09}
.wrist-box{background:#0d1a10;border-radius:6px;padding:6px;text-align:center}
.wrist-label{font-size:9px;color:#2a6a3a;text-transform:uppercase}
.wrist-val{font-size:12px;font-weight:800;margin-top:2px}
.drawer{color:#00ff88}.pocket{color:#ff4444}.other{color:#888}
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
  <div class="brand"><span class="g">Mamun</span><span class="w">Shop</span> <span class="g" style="font-size:11px">AI v4</span></div>
  <div class="badges">
    <div class="pill {{ cam1_cls }}" id="cam1Pill">CAM1: {{ cam1 }}</div>
    <div class="pill {{ cam2_cls }}" id="cam2Pill">CAM2: {{ cam2 }}</div>
  </div>
</div>
<div class="content">
  <div class="stats">
    <div class="stat"><div class="stat-val" style="color:#ff4444" id="alertCount">{{ alert_count }}</div><div class="stat-lbl">Alerts</div></div>
    <div class="stat"><div class="stat-val" id="personsVal">{{ persons }}</div><div class="stat-lbl">Persons</div></div>
    <div class="stat"><div class="stat-val" id="fpsVal" style="font-size:14px;padding-top:5px">{{ fps }}</div><div class="stat-lbl">FPS</div></div>
    <div class="stat"><div class="stat-val" id="triggerVal" style="font-size:11px;padding-top:5px">{{ last_trigger }}</div><div class="stat-lbl">Last Trigger</div></div>
  </div>

  <div class="cams">
    <div class="cam-card">
      <div class="cam-head">
        <div style="display:flex;align-items:center;gap:8px">
          <div class="cam-title">NVR Channel 8 — Side View</div>
          <div class="cam-badge ai-badge">🤖 AI ACTIVE</div>
        </div>
        <div class="rec"><div class="rdot"></div>LIVE</div>
      </div>
      <img src="/video2" alt="NVR Feed">
      <div class="wrist-info">
        <div class="wrist-box">
          <div class="wrist-label">Left Wrist</div>
          <div class="wrist-val {{ lw_cls }}" id="lwVal">{{ lw }}</div>
        </div>
        <div class="wrist-box">
          <div class="wrist-label">Right Wrist</div>
          <div class="wrist-val {{ rw_cls }}" id="rwVal">{{ rw }}</div>
        </div>
      </div>
      <div class="cam-foot">YOLOv8n-pose | Drawer→Pocket trajectory detection | Gemini AI verification</div>
    </div>

    <div class="cam-card">
      <div class="cam-head">
        <div class="cam-title">EZVIZ — Overhead View</div>
        <div class="rec"><div class="rdot"></div>LIVE</div>
      </div>
      <img src="/video1" alt="EZVIZ Feed">
      <div class="cam-foot">Display only</div>
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
function zoneClass(z) {
  if (!z) return 'other';
  z = z.toUpperCase();
  if (z === 'DRAWER') return 'drawer';
  if (z === 'POCKET') return 'pocket';
  return 'other';
}
setInterval(() => {
  fetch("/status").then(r => r.json()).then(d => {
    document.getElementById("alertCount").textContent  = d.alert_count;
    document.getElementById("alertBadge").textContent  = d.alert_count + " total";
    document.getElementById("personsVal").textContent  = d.persons;
    document.getElementById("fpsVal").textContent      = d.fps + " fps";
    document.getElementById("triggerVal").textContent  = d.last_trigger;
    const lw = document.getElementById("lwVal");
    lw.textContent = d.left_wrist;
    lw.className = "wrist-val " + zoneClass(d.left_wrist);
    const rw = document.getElementById("rwVal");
    rw.textContent = d.right_wrist;
    rw.className = "wrist-val " + zoneClass(d.right_wrist);
    document.getElementById("cam1Pill").textContent = "CAM1: " + d.cam1;
    document.getElementById("cam1Pill").className = "pill " + (d.cam1==="Online"?"online":"offline");
    document.getElementById("cam2Pill").textContent = "CAM2: " + d.cam2;
    document.getElementById("cam2Pill").className = "pill " + (d.cam2==="Online"?"online":"offline");
    if (d.alerts && d.alerts.length > 0) {
      const head = document.querySelector(".alerts-head").outerHTML;
      const items = d.alerts.map(a =>
        "<div class='alert-item'><div class='alert-msg'>"+a.msg+
        "</div><div class='alert-time'>"+a.time+
        "</div><div class='badge'>✓ Verified by Gemini AI</div></div>"
      ).join("");
      document.getElementById("alertsBox").innerHTML = head + items;
    }
  });
}, 2000);
</script>
</body>
</html>'''

# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    with alert_lock:
        al = list(alerts)
    lw = system_status["left_wrist"]
    rw = system_status["right_wrist"]
    def cls(z):
        if z == "DRAWER": return "drawer"
        if z == "POCKET": return "pocket"
        return "other"
    return render_template_string(HTML,
        cam1=system_status["cam1"],
        cam2=system_status["cam2"],
        cam1_cls="online" if system_status["cam1"]=="Online" else "offline",
        cam2_cls="online" if system_status["cam2"]=="Online" else "offline",
        persons=system_status["persons"],
        fps=f'{system_status["fps2"]}',
        last_trigger=system_status["last_trigger"],
        lw=lw, rw=rw,
        lw_cls=cls(lw), rw_cls=cls(rw),
        alerts=al,
        alert_count=len(al))

@app.route('/status')
def status():
    with alert_lock:
        al = list(alerts[:20])
    return jsonify({
        "cam1":         system_status["cam1"],
        "cam2":         system_status["cam2"],
        "persons":      system_status["persons"],
        "left_wrist":   system_status["left_wrist"],
        "right_wrist":  system_status["right_wrist"],
        "last_trigger": system_status["last_trigger"],
        "fps":          system_status["fps2"],
        "alerts":       al,
        "alert_count":  len(alerts),
    })

# Use mutable containers so gen_stream can reference latest frame
cam1_ref = [None]
cam2_ref = [None]

def sync_frames():
    global cam1_ref, cam2_ref
    while True:
        with frame_lock1:
            cam1_ref[0] = cam1_frame
        with frame_lock2:
            cam2_ref[0] = cam2_frame
        time.sleep(0.03)

@app.route('/video1')
def video1():
    def gen():
        while True:
            f = cam1_ref[0]
            if f is not None:
                ret, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                           buf.tobytes() + b'\r\n')
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
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                           buf.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Thread(target=camera1_loop, daemon=True).start()
    threading.Thread(target=camera2_loop, daemon=True).start()
    threading.Thread(target=sync_frames,  daemon=True).start()
    app.run(host='0.0.0.0', port=8080, threaded=True)
