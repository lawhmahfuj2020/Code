import cv2
import time
import requests
import threading
import base64
import numpy as np
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CAM_URL        = "rtsp://admin:MTQUSN@146.196.49.41:554/ch1/main"
BOT_TOKEN      = "8831097652:AAFluHl3A9c-mRFGg3yLBX2rQr-xBOMe8xc"
CHAT_ID        = "2052275350"
GEMINI_API_KEY = "AQ.Ab8RN6K8v8l1IoNUfHlTVXdXCoL4fS-iG6-3QrtBh68N4-j3RQ"
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

ALERT_COOLDOWN  = 120   # seconds between alerts
GEMINI_COOLDOWN = 20    # seconds between Gemini calls
MOTION_THRESHOLD = 800  # pixel area to count as real motion in drawer zone

# ─── ZONES (1920x1080) — calibrated from actual footage ───────────────────────
# The cash drawer open area (where money is visible from overhead)
DRAWER_ZONE = (370, 510, 870, 760)

# Cashier standing area — used to distinguish cashier vs intruder
# If a person centroid is here, they are the cashier (behind counter)
CASHIER_ZONE = (0, 400, 900, 1080)

# Customer zone — in front of counter
CUSTOMER_ZONE = (200, 50, 1400, 400)

# ─── STATE ────────────────────────────────────────────────────────────────────
current_frame   = None
frame_lock      = threading.Lock()
alerts          = []
alert_lock      = threading.Lock()
last_alert_time = 0
last_gemini_time = 0
system_status   = {
    "cam": "Connecting...",
    "motion_score": 0,
    "drawer_open": False,
    "last_trigger": "None",
    "fps": 0,
}

app = Flask(__name__)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def encode_image(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')

def send_telegram_photo(frame, caption):
    try:
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        requests.post(url,
            data={"chat_id": CHAT_ID, "caption": caption},
            files={"photo": ("alert.jpg", buf.tobytes(), "image/jpeg")},
            timeout=10)
        print(f"[TELEGRAM] Alert sent: {caption[:60]}")
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")

def add_alert(msg, level="THREAT"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with alert_lock:
        alerts.insert(0, {"time": ts, "msg": msg, "level": level})
        if len(alerts) > 50:
            alerts.pop()

def ask_gemini(frame):
    """Ask Gemini if this is suspicious cash handling. Returns True if suspicious."""
    try:
        img_b64 = encode_image(frame)
        prompt = (
            "This is a top-down CCTV image of a shop cash counter. "
            "The cash drawer is open and there is activity near the money. "
            "Look carefully: is someone taking/stealing cash from the drawer without a customer present, "
            "or pocketing money? "
            "Reply with ONLY one word: SUSPICIOUS or NORMAL"
        )
        payload = {"contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
        ]}]}
        r = requests.post(GEMINI_URL, json=payload, timeout=15)
        answer = r.json()['candidates'][0]['content']['parts'][0]['text'].strip().upper()
        print(f"[GEMINI] Response: {answer}")
        return "SUSPICIOUS" in answer
    except Exception as e:
        print(f"[GEMINI ERROR] {e}")
        return False  # Don't alert if Gemini fails

def draw_overlay(frame, motion_score, drawer_active, customer_present):
    """Draw zones and status overlay on frame."""
    h, w = frame.shape[:2]

    # Drawer zone — red when active, green when idle
    dcolor = (0, 0, 255) if drawer_active else (0, 200, 0)
    cv2.rectangle(frame,
        (DRAWER_ZONE[0], DRAWER_ZONE[1]),
        (DRAWER_ZONE[2], DRAWER_ZONE[3]),
        dcolor, 3)
    label = "⚠ DRAWER ACTIVE" if drawer_active else "DRAWER ZONE"
    cv2.putText(frame, label,
        (DRAWER_ZONE[0]+5, DRAWER_ZONE[1]-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, dcolor, 2)

    # Customer zone
    ccolor = (0, 255, 255) if customer_present else (80, 80, 80)
    cv2.rectangle(frame,
        (CUSTOMER_ZONE[0], CUSTOMER_ZONE[1]),
        (CUSTOMER_ZONE[2], CUSTOMER_ZONE[3]),
        ccolor, 1)
    cv2.putText(frame, "CUSTOMER ZONE" if customer_present else "NO CUSTOMER",
        (CUSTOMER_ZONE[0]+5, CUSTOMER_ZONE[1]+22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ccolor, 2)

    # Status bar at top
    bar_color = (0, 0, 180) if drawer_active else (0, 80, 0)
    cv2.rectangle(frame, (0, 0), (w, 40), bar_color, -1)
    ts = datetime.now().strftime("%H:%M:%S")
    status_text = f"MAMUN SHOP AI  |  {ts}  |  Motion: {motion_score}  |  {'⚠ DRAWER ACTIVE' if drawer_active else 'Normal'}"
    cv2.putText(frame, status_text, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return frame

# ─── MAIN CAMERA + DETECTION LOOP ─────────────────────────────────────────────
def camera_loop():
    global current_frame, last_alert_time, last_gemini_time

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=200,        # frames to build background model
        varThreshold=50,    # sensitivity — lower = more sensitive
        detectShadows=False
    )

    frame_count = 0
    t0 = time.time()

    while True:
        try:
            cap = cv2.VideoCapture(CAM_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimal buffer = less lag

            if not cap.isOpened():
                system_status["cam"] = "Offline"
                print("[CAM] Cannot connect, retrying in 10s...")
                time.sleep(10)
                continue

            system_status["cam"] = "Online"
            print("[CAM] Connected!")

            # Reset background model on reconnect
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=50, detectShadows=False)

            while True:
                ret, frame = cap.read()
                if not ret:
                    system_status["cam"] = "Reconnecting..."
                    print("[CAM] Frame lost, reconnecting...")
                    break

                frame_count += 1

                # ── FPS tracking ──
                if frame_count % 30 == 0:
                    elapsed = time.time() - t0
                    system_status["fps"] = round(30 / elapsed, 1)
                    t0 = time.time()

                # ── Background subtraction on full frame ──
                fg_mask = bg_subtractor.apply(frame)

                # ── Crop mask to DRAWER ZONE only ──
                dx1, dy1, dx2, dy2 = DRAWER_ZONE
                drawer_mask = fg_mask[dy1:dy2, dx1:dx2]

                # Clean up noise with morphology
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                drawer_mask = cv2.morphologyEx(drawer_mask, cv2.MORPH_OPEN, kernel)
                drawer_mask = cv2.dilate(drawer_mask, kernel, iterations=2)

                # Motion score = white pixel count in drawer zone
                motion_score = int(np.sum(drawer_mask > 128))
                system_status["motion_score"] = motion_score

                drawer_active = motion_score > MOTION_THRESHOLD
                system_status["drawer_open"] = drawer_active

                # ── Customer presence: simple motion in customer zone ──
                customer_mask = fg_mask[CUSTOMER_ZONE[1]:CUSTOMER_ZONE[3],
                                        CUSTOMER_ZONE[0]:CUSTOMER_ZONE[2]]
                customer_motion = int(np.sum(customer_mask > 128))
                customer_present = customer_motion > 3000

                # ── Draw overlay ──
                display = draw_overlay(frame.copy(), motion_score, drawer_active, customer_present)

                # Highlight motion area in drawer zone
                if drawer_active:
                    # Tint the drawer zone red on display frame
                    roi = display[dy1:dy2, dx1:dx2]
                    red_tint = roi.copy()
                    red_tint[:, :, 0] = 0   # remove blue
                    red_tint[:, :, 1] = 0   # remove green
                    display[dy1:dy2, dx1:dx2] = cv2.addWeighted(roi, 0.6, red_tint, 0.4, 0)

                with frame_lock:
                    current_frame = display.copy()

                # ── Alert logic ──
                current_time = time.time()

                if (drawer_active and
                    not customer_present and  # no customer = suspicious
                    current_time - last_gemini_time > GEMINI_COOLDOWN):

                    last_gemini_time = current_time
                    snap = frame.copy()  # clean frame for Gemini
                    system_status["last_trigger"] = datetime.now().strftime("%H:%M:%S")

                    def verify_and_alert(snapshot):
                        global last_alert_time
                        print("[DETECTION] Drawer motion without customer — checking Gemini...")
                        is_suspicious = ask_gemini(snapshot)
                        if is_suspicious and time.time() - last_alert_time > ALERT_COOLDOWN:
                            last_alert_time = time.time()
                            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            caption = (
                                f"🚨 THEFT ALERT!\n"
                                f"Mamun Shop — {ts}\n"
                                f"Motion in cash drawer (no customer present)\n"
                                f"Verified by Gemini AI"
                            )
                            send_telegram_photo(snapshot, caption)
                            add_alert(caption, "THREAT")

                    threading.Thread(target=verify_and_alert, args=(snap,), daemon=True).start()

            cap.release()

        except Exception as e:
            system_status["cam"] = "Error"
            print(f"[CAM ERROR] {e}")
            time.sleep(10)

# ─── FLASK STREAM ─────────────────────────────────────────────────────────────
def generate_stream():
    while True:
        with frame_lock:
            frame = current_frame.copy() if current_frame is not None else None
        if frame is not None:
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
        time.sleep(0.05)  # ~20fps stream

# ─── DASHBOARD HTML ───────────────────────────────────────────────────────────
HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mamun Shop AI Security</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#080d0a;color:#e0e8e4;font-family:"Segoe UI",Arial,sans-serif}
.topbar{background:#0a140d;border-bottom:2px solid #1a4a2a;padding:10px 16px;
        display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
.brand{font-size:17px;font-weight:800;letter-spacing:1px}
.brand .g{color:#00ff88}.brand .w{color:#fff}
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
.rec{display:flex;align-items:center;gap:4px;font-size:9px;color:#ff4444;font-weight:700}
.rdot{width:7px;height:7px;background:#ff4444;border-radius:50%;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.cam-card img{width:100%;display:block}
.cam-foot{padding:6px 12px;font-size:10px;color:#2a5a32;font-family:monospace;background:#080f09}
.alerts-box{background:#0d1a10;border:1px solid #1a3a22;border-radius:10px;overflow:hidden}
.alerts-head{padding:10px 14px;background:#0a1208;border-bottom:1px solid #1a3a22;
             display:flex;align-items:center;justify-content:space-between}
.alerts-title{font-size:13px;font-weight:800;color:#ff4444}
.alerts-count{background:rgba(255,68,68,.15);color:#ff4444;font-size:10px;font-weight:700;
              padding:2px 8px;border-radius:10px;border:1px solid rgba(255,68,68,.3)}
.alert-item{padding:10px 14px;border-left:3px solid #ff4444;margin:8px;
            background:#080f09;border-radius:0 8px 8px 0}
.alert-msg{font-size:11px;line-height:1.6;white-space:pre-line}
.alert-time{font-size:9px;color:#2a5a32;margin-top:3px;font-family:monospace}
.badge{font-size:9px;color:#4488ff;margin-top:2px;font-weight:600}
.empty{padding:24px;text-align:center;color:#1a3a22;font-size:12px}
.motion-bar{height:6px;background:#1a3a22;border-radius:3px;margin:6px 12px}
.motion-fill{height:100%;border-radius:3px;background:#00ff88;transition:width .3s}
</style>
</head>
<body>
<div class="topbar">
  <div class="brand"><span class="g">Mamun</span><span class="w">Shop</span> <span class="g" style="font-size:11px">AI v3</span></div>
  <div class="pill {{ cam_cls }}" id="camPill">CAM: {{ cam_status }}</div>
</div>
<div class="content">
  <div class="stats">
    <div class="stat"><div class="stat-val" style="color:#ff4444" id="alertCount">{{ alert_count }}</div><div class="stat-lbl">Alerts</div></div>
    <div class="stat"><div class="stat-val" id="motionScore">{{ motion }}</div><div class="stat-lbl">Motion</div></div>
    <div class="stat"><div class="stat-val" id="drawerStatus" style="font-size:12px;padding-top:5px">{{ drawer }}</div><div class="stat-lbl">Drawer</div></div>
    <div class="stat"><div class="stat-val" id="fpsVal" style="font-size:14px;padding-top:5px">{{ fps }}</div><div class="stat-lbl">FPS</div></div>
  </div>
  <div class="cam-card">
    <div class="cam-head">
      <div class="cam-title">EZVIZ — Cash Counter (AI Active)</div>
      <div class="rec"><div class="rdot"></div>LIVE</div>
    </div>
    <img src="/video" alt="Camera Feed">
    <div class="motion-bar"><div class="motion-fill" id="motionBar" style="width:0%"></div></div>
    <div class="cam-foot">Motion detection + Gemini AI verification | Last trigger: <span id="lastTrigger">{{ last_trigger }}</span></div>
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
    <div class="empty" id="noAlerts">✅ No suspicious activity detected</div>
    {% endif %}
  </div>
</div>
<script>
const MAX_MOTION = 15000;
setInterval(() => {
  fetch("/status").then(r => r.json()).then(d => {
    document.getElementById("alertCount").textContent  = d.alert_count;
    document.getElementById("alertBadge").textContent  = d.alert_count + " total";
    document.getElementById("motionScore").textContent = d.motion_score;
    document.getElementById("fpsVal").textContent      = d.fps + " fps";
    document.getElementById("lastTrigger").textContent = d.last_trigger;
    document.getElementById("drawerStatus").textContent= d.drawer_open ? "⚠ OPEN" : "Closed";
    document.getElementById("drawerStatus").style.color= d.drawer_open ? "#ff4444" : "#00ff88";
    const pct = Math.min(100, (d.motion_score / MAX_MOTION) * 100);
    document.getElementById("motionBar").style.width = pct + "%";
    document.getElementById("motionBar").style.background = d.drawer_open ? "#ff4444" : "#00ff88";
    const pill = document.getElementById("camPill");
    pill.textContent = "CAM: " + d.cam;
    pill.className   = "pill " + (d.cam === "Online" ? "online" : "offline");
    if (d.alerts && d.alerts.length > 0) {
      const head = document.querySelector(".alerts-head").outerHTML;
      const items = d.alerts.map(a =>
        "<div class='alert-item'><div class='alert-msg'>" + a.msg +
        "</div><div class='alert-time'>" + a.time +
        "</div><div class='badge'>✓ Verified by Gemini AI</div></div>"
      ).join("");
      document.getElementById("alertsBox").innerHTML = head + items;
    }
  });
}, 2000);
</script>
</body>
</html>'''

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    cls = 'online' if system_status["cam"] == "Online" else 'offline'
    with alert_lock:
        al = list(alerts)
    return render_template_string(HTML,
        cam_status=system_status["cam"],
        cam_cls=cls,
        motion=system_status["motion_score"],
        drawer="⚠ OPEN" if system_status["drawer_open"] else "Closed",
        fps=f'{system_status["fps"]} fps',
        last_trigger=system_status["last_trigger"],
        alerts=al,
        alert_count=len(al))

@app.route('/status')
def status():
    with alert_lock:
        al = list(alerts[:20])
    return jsonify({
        "cam":          system_status["cam"],
        "motion_score": system_status["motion_score"],
        "drawer_open":  system_status["drawer_open"],
        "last_trigger": system_status["last_trigger"],
        "fps":          system_status["fps"],
        "alerts":       al,
        "alert_count":  len(alerts),
    })

@app.route('/video')
def video():
    return Response(generate_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=8080, threaded=True)
