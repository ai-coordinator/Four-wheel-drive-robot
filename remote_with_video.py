# -*- coding: utf-8 -*-
"""
Robot Remote + Gamepad + (ZED RGB/Depth or USB) + Jetson Telemetry Dashboard
- Remote control via browser (touch/keyboard) with keepalive
- Local gamepad via /dev/input/js0 as fallback
- Serial bridge to Arduino UNO (/dev/ttyACM0)
- Video: ZED SDK RGB + Depth colormap (or USB fallback)
- Jetson telemetry: tegrastats (if available) + /proc fallback
"""

import time
import threading
import serial
import os
import struct
import select
import subprocess
import re

import numpy as np
import cv2
from flask import Flask, request, jsonify, Response

# ========= Serial (UNO bridge) =========
SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200
SEND_HZ = 50

# ========= Gamepad (Linux joystick) =========
JS_DEV = "/dev/input/js0"
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS   = 0x02
JS_EVENT_INIT   = 0x80

DEADZONE = 0.18
MAX_OUT = 127

# axis/button mapping (adjust if your controller differs)
AX_TURN = 0
AX_THROTTLE = 1
BTN_STOP = 0

# keepalive / neutral threshold
NEUTRAL = 0.06

# ========= Arbitration / Safety =========
REMOTE_TIMEOUT = 1.0
GAMEPAD_TIMEOUT = 0.25
FAILSAFE_STOP = 1.2

# ========= ZED / Video tuning =========
USE_ZED = True  # True: ZED SDK, False: fallback USB cam
VIDEO_FPS = 15
RGB_W, RGB_H = 640, 360
JPEG_QUALITY_RGB = 70
JPEG_QUALITY_DEPTH = 70

# Depth visualization range (meters)
DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 5.0

# ========= Shared state =========
lock = threading.Lock()
state = {
    "remote_L": 0, "remote_R": 0, "remote_ts": 0.0,
    "pad_L": 0, "pad_R": 0, "pad_ts": 0.0,
    "active": "none",
    "last_sent_L": 0, "last_sent_R": 0,
    "last_write_ts": 0.0,
    "remote_last_cmd_ts": 0.0,     # last time /api/cmd called (for UI ping)
}

# Latest frames (JPEG) - “latest only” to reduce latency
frame_lock = threading.Lock()
latest_rgb_jpeg = None
latest_depth_jpeg = None
video_status = {"src": "none", "ok": False, "detail": "", "fps_est": 0.0}

# ========= Jetson telemetry =========
tele_lock = threading.Lock()
telemetry = {
    "ok": False,
    "src": "none",
    "ts": 0.0,
    "data": {},
    "detail": ""
}

# ==========================
# Utils
# ==========================
def apply_deadzone(x, dz=DEADZONE):
    if abs(x) < dz:
        return 0.0
    sign = 1.0 if x >= 0 else -1.0
    x = (abs(x) - dz) / (1.0 - dz)
    return sign * max(0.0, min(1.0, x))

def clamp_int(v, lo=-MAX_OUT, hi=MAX_OUT):
    return max(lo, min(hi, int(v)))

def to_int127(x):
    return clamp_int(round(x * MAX_OUT))

def curve(x):
    return x * abs(x)

def mix_to_lr(throttle, turn):
    # Differential drive mix:
    # turn > 0 => left increases, right decreases => right turn
    left = throttle + turn
    right = throttle - turn
    m = max(1.0, abs(left), abs(right))
    left /= m
    right /= m
    return to_int127(left), to_int127(right)

def encode_jpeg(bgr, quality=70):
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", bgr, params)
    return buf.tobytes() if ok else None

def now_s():
    return time.time()

def human_uptime(sec: int):
    if sec is None:
        return "n/a"
    d = sec // 86400
    h = (sec % 86400) // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if d > 0:
        return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"

# ==========================
# Gamepad loop (read /dev/input/js0 directly)
# ==========================
def gamepad_loop_js():
    if not os.path.exists(JS_DEV):
        print("[GAMEPAD] js0 not found:", JS_DEV)
        return

    fd = os.open(JS_DEV, os.O_RDONLY | os.O_NONBLOCK)
    print("[GAMEPAD] opened:", JS_DEV)

    turn_raw = 0.0
    throttle_raw = 0.0
    stop_pressed = False

    tick_hz = 50
    tick_dt = 1.0 / tick_hz

    while True:
        now = now_s()

        # Read any pending events (non-blocking)
        r, _, _ = select.select([fd], [], [], tick_dt)
        if r:
            while True:
                try:
                    data = os.read(fd, 8)
                    if len(data) < 8:
                        break
                    _, value, etype, num = struct.unpack("IhBB", data)
                    etype &= ~JS_EVENT_INIT

                    if etype == JS_EVENT_AXIS:
                        v = max(-1.0, min(1.0, value / 32767.0))
                        if num == AX_TURN:
                            turn_raw = v
                        elif num == AX_THROTTLE:
                            throttle_raw = v

                    elif etype == JS_EVENT_BUTTON:
                        if num == BTN_STOP:
                            stop_pressed = (value == 1)

                except BlockingIOError:
                    break
                except OSError as ex:
                    print("[GAMEPAD] read error:", ex)
                    return

        # Keepalive: if engaged, update pad_ts every tick (even if value unchanged)
        engaged = (abs(turn_raw) > NEUTRAL) or (abs(throttle_raw) > NEUTRAL) or stop_pressed

        if engaged:
            throttle = curve(apply_deadzone(-throttle_raw))  # up => forward
            turn = curve(apply_deadzone(-turn_raw))          # invert if needed
            L, R = mix_to_lr(throttle, turn)
            if stop_pressed:
                L, R = 0, 0

            with lock:
                state["pad_L"] = L
                state["pad_R"] = R
                state["pad_ts"] = now
        else:
            # neutral: don't touch pad_ts (so remote can take priority)
            with lock:
                state["pad_L"] = 0
                state["pad_R"] = 0

# ==========================
# Serial sender loop
# ==========================
def sender_loop():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.01)
    except Exception as e:
        print("[SERIAL] open failed:", e)
        return

    time.sleep(2.0)  # UNO reset wait
    dt = 1.0 / SEND_HZ

    while True:
        now = now_s()
        with lock:
            remote_ok = (now - state["remote_ts"]) <= REMOTE_TIMEOUT
            pad_ok = (now - state["pad_ts"]) <= GAMEPAD_TIMEOUT

            if remote_ok:
                L, R = state["remote_L"], state["remote_R"]
                state["active"] = "remote"
            elif pad_ok:
                L, R = state["pad_L"], state["pad_R"]
                state["active"] = "gamepad"
            else:
                if (now - max(state["remote_ts"], state["pad_ts"])) > FAILSAFE_STOP:
                    L, R = 0, 0
                    state["active"] = "failsafe"
                else:
                    L, R = 0, 0
                    state["active"] = "none"

            state["last_sent_L"], state["last_sent_R"] = L, R
            state["last_write_ts"] = now

        # Always send (UNO failsafe)
        try:
            ser.write(f"{L},{R}\n".encode("ascii"))
        except Exception as e:
            print("[SERIAL] write error:", e)
            time.sleep(0.5)
        time.sleep(dt)

# ==========================
# ZED RGB + Depth loop
# ==========================
def depth_to_colormap(depth_m: np.ndarray, dmin=DEPTH_MIN_M, dmax=DEPTH_MAX_M):
    # invalid => 0
    valid = np.isfinite(depth_m) & (depth_m > 0)
    depth = np.clip(depth_m, dmin, dmax)

    # normalize
    norm = (depth - dmin) / (dmax - dmin)  # 0..1
    norm[~valid] = 0.0
    img8 = (norm * 255).astype(np.uint8)

    # invert so near is "hot"
    img8 = 255 - img8
    cm = cv2.applyColorMap(img8, cv2.COLORMAP_JET)
    return cm

def zed_video_loop():
    global latest_rgb_jpeg, latest_depth_jpeg

    try:
        import pyzed.sl as sl
    except Exception as e:
        video_status.update({"src": "ZED", "ok": False, "detail": f"pyzed import failed: {e}"})
        print("[ZED] pyzed import failed:", e)
        return

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = VIDEO_FPS
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER

    st = zed.open(init)
    if st != sl.ERROR_CODE.SUCCESS:
        video_status.update({"src": "ZED", "ok": False, "detail": f"open failed: {st}"})
        print("[ZED] open failed:", st)
        return

    video_status.update({"src": "ZED", "ok": True, "detail": "ZED SDK OK"})
    print("[ZED] started")

    image = sl.Mat()
    depth = sl.Mat()

    dt = 1.0 / max(1, VIDEO_FPS)
    last_t = time.time()
    fps_smooth = 0.0

    while True:
        # grab latest
        if zed.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.01)
            continue

        zed.retrieve_image(image, sl.VIEW.LEFT)
        bgr = image.get_data()  # usually BGRA
        if bgr is None:
            time.sleep(0.01)
            continue

        # Convert BGRA->BGR if needed
        if bgr.ndim == 3 and bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

        bgr = cv2.resize(bgr, (RGB_W, RGB_H), interpolation=cv2.INTER_AREA)

        # Depth (meters)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        depth_m = depth.get_data()
        if depth_m is None:
            time.sleep(0.01)
            continue
        if depth_m.ndim == 3:
            depth_m = depth_m[:, :, 0]

        depth_vis = cv2.resize(
            depth_to_colormap(depth_m),
            (RGB_W, RGB_H),
            interpolation=cv2.INTER_NEAREST
        )

        # Overlay timestamp (debug latency)
        ts = f"{time.time():.3f}"
        cv2.putText(bgr, ts, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245,245,245), 2, cv2.LINE_AA)
        cv2.putText(depth_vis, ts, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245,245,245), 2, cv2.LINE_AA)

        rgb_jpg = encode_jpeg(bgr, JPEG_QUALITY_RGB)
        dep_jpg = encode_jpeg(depth_vis, JPEG_QUALITY_DEPTH)
        if rgb_jpg and dep_jpg:
            with frame_lock:
                latest_rgb_jpeg = rgb_jpg
                latest_depth_jpeg = dep_jpg

        # FPS estimate
        t = time.time()
        inst = 1.0 / max(1e-6, (t - last_t))
        last_t = t
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst
        video_status["fps_est"] = round(fps_smooth, 1)

        time.sleep(dt)

# ==========================
# USB fallback loop (if ZED not available)
# ==========================
def usb_video_loop():
    global latest_rgb_jpeg, latest_depth_jpeg

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        video_status.update({"src": "USB", "ok": False, "detail": "open failed"})
        return

    # Reduce latency
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RGB_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RGB_H)
    cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)

    video_status.update({"src": "USB", "ok": True, "detail": "USB cam OK (no depth)"})
    dt = 1.0 / max(1, VIDEO_FPS)

    last_t = time.time()
    fps_smooth = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        # If stereo side-by-side, show left only
        h, w = frame.shape[:2]
        if w >= h * 2:
            frame = frame[:, :w // 2]

        frame = cv2.resize(frame, (RGB_W, RGB_H), interpolation=cv2.INTER_AREA)
        ts = f"{time.time():.3f}"
        cv2.putText(frame, ts, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245,245,245), 2, cv2.LINE_AA)

        rgb_jpg = encode_jpeg(frame, JPEG_QUALITY_RGB)
        if rgb_jpg:
            with frame_lock:
                latest_rgb_jpeg = rgb_jpg
                latest_depth_jpeg = None

        t = time.time()
        inst = 1.0 / max(1e-6, (t - last_t))
        last_t = t
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst
        video_status["fps_est"] = round(fps_smooth, 1)

        time.sleep(dt)

# ==========================
# Jetson Telemetry
# ==========================
def read_proc_fallback():
    d = {}

    # uptime
    try:
        with open("/proc/uptime", "r") as f:
            up = float(f.read().split()[0])
        d["uptime_s"] = int(up)
        d["uptime_h"] = human_uptime(int(up))
    except Exception:
        pass

    # loadavg
    try:
        with open("/proc/loadavg", "r") as f:
            a, b, c = f.read().split()[:3]
        d["loadavg"] = [float(a), float(b), float(c)]
    except Exception:
        pass

    # meminfo
    try:
        mem = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                k = line.split(":")[0]
                v = line.split(":")[1].strip().split()[0]
                mem[k] = int(v)  # kB
        if "MemTotal" in mem and "MemAvailable" in mem:
            total = mem["MemTotal"] // 1024
            avail = mem["MemAvailable"] // 1024
            used = max(0, total - avail)
            d["ram_total_mb"] = int(total)
            d["ram_used_mb"] = int(used)
            d["ram_util_pct"] = round(100.0 * used / max(1, total), 1)
    except Exception:
        pass

    # disk /
    try:
        stv = os.statvfs("/")
        total = (stv.f_blocks * stv.f_frsize) / (1024**3)
        free  = (stv.f_bfree  * stv.f_frsize) / (1024**3)
        used = total - free
        d["disk_total_gb"] = round(total, 1)
        d["disk_free_gb"]  = round(free, 1)
        d["disk_used_gb"]  = round(used, 1)
        d["disk_util_pct"] = round(100.0 * used / max(1e-6, total), 1)
    except Exception:
        pass

    return d

def parse_tegrastats_line(line: str):
    d = {}
    s = line.strip()

    # RAM x/y
    m = re.search(r"RAM (\d+)\/(\d+)MB", s)
    if m:
        d["ram_used_mb"] = int(m.group(1))
        d["ram_total_mb"] = int(m.group(2))
        d["ram_util_pct"] = round(100.0 * d["ram_used_mb"] / max(1, d["ram_total_mb"]), 1)

    # SWAP x/y
    m = re.search(r"SWAP (\d+)\/(\d+)MB", s)
    if m:
        d["swap_used_mb"] = int(m.group(1))
        d["swap_total_mb"] = int(m.group(2))
        d["swap_util_pct"] = round(100.0 * d["swap_used_mb"] / max(1, d["swap_total_mb"]), 1)

    # GPU util
    m = re.search(r"GR3D_FREQ (\d+)%", s)
    if m:
        d["gpu_util_pct"] = int(m.group(1))

    # EMC util
    m = re.search(r"EMC_FREQ (\d+)%", s)
    if m:
        d["emc_util_pct"] = int(m.group(1))

    # CPU avg
    m = re.search(r"CPU \[(.+?)\]", s)
    if m:
        parts = m.group(1).split(",")
        vals = []
        for p in parts:
            m2 = re.search(r"(\d+)%", p)
            if m2:
                vals.append(int(m2.group(1)))
        if vals:
            d["cpu_util_pct"] = round(sum(vals) / len(vals), 1)

    # Temps
    temps = {}
    for name, val in re.findall(r"([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C", s):
        # tegrastats sometimes yields "GPU@44C" etc.
        temps[name] = float(val)
    if temps:
        d["temps_c"] = temps
        # pick "max_temp"
        d["temp_max_c"] = max(temps.values()) if temps else None

    d["raw"] = s
    return d

def telemetry_loop():
    # Prefer tegrastats if available
    cmd = ["tegrastats", "--interval", "1000"]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        with tele_lock:
            telemetry.update({"ok": True, "src": "tegrastats", "detail": "running"})
        while True:
            line = p.stdout.readline() if p.stdout else ""
            if not line:
                time.sleep(0.2)
                continue
            d = parse_tegrastats_line(line)
            d.update(read_proc_fallback())
            with tele_lock:
                telemetry["ok"] = True
                telemetry["ts"] = time.time()
                telemetry["data"] = d
    except Exception as e:
        with tele_lock:
            telemetry.update({"ok": False, "src": "fallback", "detail": f"tegrastats failed: {e}"})
        while True:
            d = read_proc_fallback()
            with tele_lock:
                telemetry["ok"] = True
                telemetry["ts"] = time.time()
                telemetry["data"] = d
            time.sleep(1.0)

# ==========================
# Flask server
# ==========================
app = Flask(__name__)

INDEX_HTML = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot Console</title>
<style>
:root {{
  --bg:#0b0f17;
  --panel:rgba(255,255,255,.06);
  --panel2:rgba(255,255,255,.08);
  --border:rgba(255,255,255,.12);
  --text:rgba(255,255,255,.92);
  --muted:rgba(255,255,255,.62);
  --good:#2be4a7;
  --warn:#ffd166;
  --bad:#ff4d6d;
  --accent:#7aa2ff;
  --shadow:0 10px 30px rgba(0,0,0,.45);
  --radius:16px;
}}

*{{box-sizing:border-box}}
body{{
  margin:0; padding:0;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial;
  color:var(--text);
  background:
    radial-gradient(1200px 600px at 10% 0%, rgba(122,162,255,.16), transparent 60%),
    radial-gradient(900px 500px at 100% 30%, rgba(43,228,167,.10), transparent 55%),
    radial-gradient(900px 500px at 40% 120%, rgba(255,77,109,.10), transparent 60%),
    var(--bg);
}}

header{{
  position:sticky; top:0; z-index:10;
  backdrop-filter: blur(14px);
  background: rgba(10,14,22,.72);
  border-bottom:1px solid var(--border);
}}
.topbar{{
  max-width: 1400px;
  margin:0 auto;
  padding:14px 16px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
}}
.brand{{
  display:flex; align-items:center; gap:10px;
  font-weight:800; letter-spacing:.2px;
}}
.badge{{
  display:inline-flex; align-items:center; gap:8px;
  padding:8px 10px;
  border-radius:999px;
  background: var(--panel);
  border:1px solid var(--border);
  box-shadow: var(--shadow);
  font-size:13px;
}}
.dot{{
  width:10px; height:10px; border-radius:50%;
  background: var(--muted);
  box-shadow: 0 0 0 4px rgba(255,255,255,.06);
}}
.dot.good{{ background: var(--good); box-shadow:0 0 0 4px rgba(43,228,167,.14);}}
.dot.warn{{ background: var(--warn); box-shadow:0 0 0 4px rgba(255,209,102,.14);}}
.dot.bad{{  background: var(--bad);  box-shadow:0 0 0 4px rgba(255,77,109,.14);}}

main{{
  max-width:1400px;
  margin:0 auto;
  padding:16px;
}}
.grid{{
  display:grid;
  grid-template-columns: 320px 1fr 360px;
  gap:14px;
  align-items:start;
}}
@media (max-width: 1180px){{
  .grid{{ grid-template-columns: 1fr; }}
}}

.card{{
  background: var(--panel);
  border:1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  overflow:hidden;
}}
.card .hd{{
  padding:12px 12px 10px 12px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  border-bottom:1px solid rgba(255,255,255,.08);
  background: rgba(255,255,255,.03);
}}
.card .hd b{{ font-size:13px; letter-spacing:.3px; }}
.card .bd{{ padding:12px; }}
.small{{ color:var(--muted); font-size:12px; }}

.stream{{
  width:100%;
  max-width:100%;
  border-radius:14px;
  display:block;
  background: rgba(0,0,0,.25);
}}

.padWrap{{
  display:flex;
  flex-direction:column;
  gap:12px;
}}
#pad{{
  width:100%;
  aspect-ratio:1/1;
  border:1px solid rgba(255,255,255,.14);
  border-radius: 22px;
  background:
    radial-gradient(circle at 50% 50%, rgba(122,162,255,.20), rgba(255,255,255,.02) 60%),
    rgba(0,0,0,.18);
  position:relative;
  touch-action:none;
  user-select:none;
  overflow:hidden;
}}
#pad:before{{
  content:"";
  position:absolute; inset:0;
  background:
    radial-gradient(circle at 50% 50%, rgba(255,255,255,.12), transparent 55%),
    radial-gradient(circle at 50% 50%, transparent 62%, rgba(255,255,255,.08) 63%, transparent 64%);
  opacity:.8;
}}
#dot{{
  width:22px;height:22px;
  border-radius:50%;
  position:absolute;
  left:50%; top:50%;
  transform: translate(-50%,-50%);
  background: rgba(255,255,255,.86);
  box-shadow: 0 0 0 6px rgba(122,162,255,.22), 0 10px 22px rgba(0,0,0,.35);
}}

.controls{{
  display:flex;
  gap:10px;
}}
.btn{{
  flex:1;
  border:none;
  padding:12px 12px;
  border-radius:14px;
  background: rgba(255,255,255,.10);
  color:var(--text);
  font-weight:700;
  cursor:pointer;
  border:1px solid rgba(255,255,255,.14);
}}
.btn:hover{{ background: rgba(255,255,255,.14); }}
.btnStop{{
  background: rgba(255,77,109,.14);
  border:1px solid rgba(255,77,109,.35);
}}
.btnStop:hover{{ background: rgba(255,77,109,.22); }}

.kbd{{
  display:flex;
  gap:8px;
  flex-wrap:wrap;
}}
.key{{
  padding:7px 9px;
  border-radius:10px;
  border:1px solid rgba(255,255,255,.12);
  background: rgba(255,255,255,.06);
  font-size:12px;
  color: var(--muted);
}}

.statGrid{{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap:10px;
}}
.metric{{
  padding:10px;
  border-radius:14px;
  border:1px solid rgba(255,255,255,.10);
  background: rgba(0,0,0,.12);
}}
.metric .lbl{{ font-size:12px; color:var(--muted); }}
.metric .val{{ font-size:18px; font-weight:800; margin-top:3px; }}
.bar{{
  height:10px;
  border-radius:999px;
  background: rgba(255,255,255,.08);
  overflow:hidden;
  margin-top:8px;
  border:1px solid rgba(255,255,255,.08);
}}
.fill{{
  height:100%;
  width:0%;
  background: linear-gradient(90deg, rgba(43,228,167,.95), rgba(122,162,255,.95));
}}
.fill.bad{{
  background: linear-gradient(90deg, rgba(255,209,102,.95), rgba(255,77,109,.95));
}}

.log{{
  height:180px;
  overflow:auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size:12px;
  color: rgba(255,255,255,.78);
  background: rgba(0,0,0,.18);
  border:1px solid rgba(255,255,255,.10);
  border-radius:14px;
  padding:10px;
  line-height:1.4;
}}
</style>
</head>
<body>
<header>
  <div class="topbar">
    <div class="brand">
      <div style="width:10px;height:10px;border-radius:3px;background:var(--accent); box-shadow:0 0 0 4px rgba(122,162,255,.18)"></div>
      <div>Robot Console</div>
      <span class="badge"><span id="connDot" class="dot"></span><span id="connText">connecting...</span></span>
      <span class="badge">Video: <span id="videoText">...</span></span>
      <span class="badge">Jetson: <span id="jetsonText">...</span></span>
    </div>
    <div class="badge">
      <span class="small">Remote age</span>&nbsp;<b id="ageRemote">-</b>s
      &nbsp;|&nbsp;<span class="small">Pad age</span>&nbsp;<b id="agePad">-</b>s
      &nbsp;|&nbsp;<span class="small">Video FPS</span>&nbsp;<b id="fpsVideo">-</b>
    </div>
  </div>
</header>

<main>
  <div class="grid">

    <!-- LEFT: controls -->
    <section class="card">
      <div class="hd"><b>CONTROL</b><span class="small">touch / keyboard / gamepad</span></div>
      <div class="bd padWrap">
        <div id="pad"><div id="dot"></div></div>
        <div class="controls">
          <button class="btn btnStop" onclick="stop()">STOP</button>
          <button class="btn" onclick="center()">CENTER</button>
        </div>
        <div class="kbd">
          <span class="key">W forward</span>
          <span class="key">S back</span>
          <span class="key">A left</span>
          <span class="key">D right</span>
          <span class="key">Space stop</span>
        </div>
        <div class="small">Tip: joystick keepalive sends even if you hold the dot still.</div>
      </div>
    </section>

    <!-- CENTER: video -->
    <section class="card">
      <div class="hd">
        <b>CAMERA</b>
        <span class="small">{RGB_W}×{RGB_H} @ {VIDEO_FPS}fps</span>
      </div>
      <div class="bd" style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
        <div class="card" style="background:var(--panel2); box-shadow:none;">
          <div class="hd"><b>RGB (Left)</b><span class="small" id="rgbInfo">/video/rgb</span></div>
          <div class="bd"><img class="stream" src="/video/rgb" alt="rgb"></div>
        </div>
        <div class="card" style="background:var(--panel2); box-shadow:none;">
          <div class="hd"><b>Depth</b><span class="small">{DEPTH_MIN_M}m - {DEPTH_MAX_M}m</span></div>
          <div class="bd"><img class="stream" src="/video/depth" alt="depth"></div>
        </div>
      </div>
    </section>

    <!-- RIGHT: telemetry -->
    <section class="card">
      <div class="hd"><b>JETSON TELEMETRY</b><span class="small" id="teleTs">-</span></div>
      <div class="bd">
        <div class="statGrid">
          <div class="metric">
            <div class="lbl">CPU util</div>
            <div class="val"><span id="cpuUtil">-</span>%</div>
            <div class="bar"><div id="cpuBar" class="fill"></div></div>
          </div>
          <div class="metric">
            <div class="lbl">GPU util (GR3D)</div>
            <div class="val"><span id="gpuUtil">-</span>%</div>
            <div class="bar"><div id="gpuBar" class="fill"></div></div>
          </div>
          <div class="metric">
            <div class="lbl">RAM</div>
            <div class="val"><span id="ramUsed">-</span>/<span id="ramTotal">-</span>MB</div>
            <div class="bar"><div id="ramBar" class="fill"></div></div>
          </div>
          <div class="metric">
            <div class="lbl">Disk (/)</div>
            <div class="val"><span id="diskUsed">-</span>/<span id="diskTotal">-</span>GB</div>
            <div class="bar"><div id="diskBar" class="fill"></div></div>
          </div>
        </div>

        <div style="margin-top:12px; display:flex; gap:10px; flex-wrap:wrap;">
          <span class="badge">Uptime: <b id="uptime">-</b></span>
          <span class="badge">Load: <b id="loadavg">-</b></span>
          <span class="badge">Max temp: <b id="tempMax">-</b>°C</span>
          <span class="badge">Src: <b id="teleSrc">-</b></span>
        </div>

        <div style="margin-top:12px;">
          <div class="small" style="margin-bottom:6px;">Latest tegrastats raw</div>
          <div id="rawLine" class="log"></div>
        </div>
      </div>
    </section>

  </div>

  <div style="margin-top:14px;" class="card">
    <div class="hd"><b>STATUS</b><span class="small">active source / last sent</span></div>
    <div class="bd">
      <div class="statGrid">
        <div class="metric">
          <div class="lbl">Active</div>
          <div class="val" id="activeSrc">-</div>
        </div>
        <div class="metric">
          <div class="lbl">Last sent L/R</div>
          <div class="val"><span id="lastL">-</span> / <span id="lastR">-</span></div>
        </div>
      </div>
      <div style="margin-top:12px;">
        <div class="small" style="margin-bottom:6px;">Event log</div>
        <div id="elog" class="log"></div>
      </div>
    </div>
  </div>

</main>

<script>
const pad=document.getElementById("pad");
const dot=document.getElementById("dot");
const elog=document.getElementById("elog");
let x=0,y=0,keys={{w:false,a:false,s:false,d:false}},dragging=false;

const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));
function updateDot(){{
  const rect=pad.getBoundingClientRect();
  const r=Math.min(rect.width/2, rect.height/2)-28;
  dot.style.left=(rect.width/2 + x*r)+"px";
  dot.style.top =(rect.height/2 - y*r)+"px";
}}

function log(line){{
  const t=new Date().toLocaleTimeString();
  elog.textContent = `[${{t}}] ${{line}}\\n` + elog.textContent;
}}

function send(){{
  fetch("/api/cmd", {{
    method:"POST",
    headers:{{"Content-Type":"application/json"}},
    body:JSON.stringify({{turn:x, throttle:y}})
  }}).catch(()=>{{}});
}}

function stop(){{ x=0;y=0; updateDot(); send(); log("STOP"); }}
function center(){{ x=0;y=0; updateDot(); }}

function pointerToXY(ev){{
  const rect=pad.getBoundingClientRect();
  const cx=rect.left+rect.width/2, cy=rect.top+rect.height/2;
  x=clamp((ev.clientX-cx)/(rect.width/2),-1,1);
  y=clamp(-(ev.clientY-cy)/(rect.height/2),-1,1);
  updateDot();
}}

pad.addEventListener("pointerdown",ev=>{{dragging=true;pad.setPointerCapture(ev.pointerId);pointerToXY(ev);send();}});
pad.addEventListener("pointermove",ev=>{{if(!dragging)return;pointerToXY(ev);send();}});
pad.addEventListener("pointerup",()=>{{dragging=false; stop();}});
pad.addEventListener("pointercancel",()=>{{dragging=false; stop();}});

window.addEventListener("keydown",e=>{{
  if(e.repeat) return;
  if(e.key==="w") keys.w=true;
  if(e.key==="a") keys.a=true;
  if(e.key==="s") keys.s=true;
  if(e.key==="d") keys.d=true;
  if(e.key===" ") stop();
}});
window.addEventListener("keyup",e=>{{
  if(e.key==="w") keys.w=false;
  if(e.key==="a") keys.a=false;
  if(e.key==="s") keys.s=false;
  if(e.key==="d") keys.d=false;
}});

setInterval(()=>{{
  let t=0,r=0;
  if(keys.w)t+=1; if(keys.s)t-=1; if(keys.d)r+=1; if(keys.a)r-=1;
  if(t!==0||r!==0){{
    x=0.7*r; y=0.7*t; updateDot(); send();
  }}
}},50);

// Browser keepalive: keep sending even if held
setInterval(()=>{{
  const nonZero = (Math.abs(x)>0.02)||(Math.abs(y)>0.02);
  const activeKeys = keys.w||keys.a||keys.s||keys.d;
  if(dragging||activeKeys||nonZero) send();
}},100);

function setBar(el, pct){{
  pct = Math.max(0, Math.min(100, pct||0));
  el.style.width = pct + "%";
  if(pct >= 85) el.classList.add("bad");
  else el.classList.remove("bad");
}}

function setConn(active){{
  const dot=document.getElementById("connDot");
  const text=document.getElementById("connText");
  dot.className="dot";
  if(active==="remote"){{ dot.classList.add("good"); text.textContent="REMOTE"; }}
  else if(active==="gamepad"){{ dot.classList.add("warn"); text.textContent="GAMEPAD"; }}
  else if(active==="failsafe"){{ dot.classList.add("bad"); text.textContent="FAILSAFE"; }}
  else {{ text.textContent="IDLE"; }}
}}

async function poll(){{
  try{{
    const res = await fetch("/api/status");
    const s = await res.json();

    // header
    document.getElementById("ageRemote").textContent = s.remote.age_s;
    document.getElementById("agePad").textContent = s.gamepad.age_s;
    document.getElementById("fpsVideo").textContent = s.video.fps_est || "-";
    document.getElementById("videoText").textContent = (s.video.src||"-") + " / " + (s.video.ok ? "OK" : "NG");
    setConn(s.active);

    // status card
    document.getElementById("activeSrc").textContent = s.active;
    document.getElementById("lastL").textContent = s.last_sent.L;
    document.getElementById("lastR").textContent = s.last_sent.R;

    // telemetry
    const j = s.jetson || {{}};
    document.getElementById("jetsonText").textContent = (j.ok ? "OK" : "NG") + " / " + (j.src || "-");
    document.getElementById("teleSrc").textContent = j.src || "-";
    document.getElementById("teleTs").textContent = j.ts ? new Date(j.ts*1000).toLocaleTimeString() : "-";

    const d = j.data || {{}};
    const cpu = d.cpu_util_pct ?? 0;
    const gpu = d.gpu_util_pct ?? 0;
    const ramU = d.ram_used_mb ?? 0;
    const ramT = d.ram_total_mb ?? 0;
    const ramP = d.ram_util_pct ?? (ramT? (100*ramU/ramT) : 0);
    const diskU = d.disk_used_gb ?? 0;
    const diskT = d.disk_total_gb ?? 0;
    const diskP = d.disk_util_pct ?? (diskT? (100*diskU/diskT) : 0);

    document.getElementById("cpuUtil").textContent = cpu;
    document.getElementById("gpuUtil").textContent = gpu;
    document.getElementById("ramUsed").textContent = ramU;
    document.getElementById("ramTotal").textContent = ramT;
    document.getElementById("diskUsed").textContent = diskU;
    document.getElementById("diskTotal").textContent = diskT;

    setBar(document.getElementById("cpuBar"), cpu);
    setBar(document.getElementById("gpuBar"), gpu);
    setBar(document.getElementById("ramBar"), ramP);
    setBar(document.getElementById("diskBar"), diskP);

    document.getElementById("uptime").textContent = d.uptime_h || "-";
    document.getElementById("loadavg").textContent = d.loadavg ? d.loadavg.join(", ") : "-";
    document.getElementById("tempMax").textContent = (d.temp_max_c ?? "-");

    const raw = d.raw || "";
    const rawEl = document.getElementById("rawLine");
    if(raw) rawEl.textContent = raw + "\\n" + rawEl.textContent.split("\\n").slice(0,50).join("\\n");
  }} catch(e) {{
    // silent
  }}
}}

setInterval(poll, 300);
updateDot();
log("Console ready");
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")

@app.get("/api/status")
def api_status():
    with lock:
        s = {
            "active": state["active"],
            "remote": {"L": state["remote_L"], "R": state["remote_R"], "age_s": round(time.time() - state["remote_ts"], 3)},
            "gamepad": {"L": state["pad_L"], "R": state["pad_R"], "age_s": round(time.time() - state["pad_ts"], 3)},
            "last_sent": {"L": state["last_sent_L"], "R": state["last_sent_R"], "ts": state["last_write_ts"]},
            "js": {"dev": JS_DEV, "AX_TURN": AX_TURN, "AX_THROTTLE": AX_THROTTLE, "BTN_STOP": BTN_STOP, "NEUTRAL": NEUTRAL},
            "video": dict(video_status),
            "video_cfg": {
                "fps": VIDEO_FPS, "rgb": [RGB_W, RGB_H],
                "jpeg_rgb": JPEG_QUALITY_RGB, "jpeg_depth": JPEG_QUALITY_DEPTH,
                "depth_range_m": [DEPTH_MIN_M, DEPTH_MAX_M],
                "use_zed": USE_ZED
            },
        }
    with tele_lock:
        s["jetson"] = dict(telemetry)
    return jsonify(s)

@app.post("/api/cmd")
def api_cmd():
    data = request.get_json(force=True, silent=True) or {}
    turn = float(data.get("turn", 0.0))
    throttle = float(data.get("throttle", 0.0))

    turn = max(-1.0, min(1.0, turn))
    throttle = max(-1.0, min(1.0, throttle))

    throttle = curve(apply_deadzone(throttle))
    turn = curve(apply_deadzone(-turn))  # browser left/right invert (remove "-" if you want)

    L, R = mix_to_lr(throttle, turn)

    with lock:
        state["remote_L"] = L
        state["remote_R"] = R
        state["remote_ts"] = time.time()
        state["remote_last_cmd_ts"] = time.time()

    return jsonify({"ok": True, "L": L, "R": R})

def mjpeg_stream(which="rgb"):
    boundary = b"--frame"
    while True:
        with frame_lock:
            jpg = latest_rgb_jpeg if which == "rgb" else latest_depth_jpeg

        if jpg is None:
            time.sleep(0.05)
            continue

        yield (boundary + b"\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
               jpg + b"\r\n")

@app.get("/video/rgb")
def video_rgb():
    return Response(mjpeg_stream("rgb"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/video/depth")
def video_depth():
    return Response(mjpeg_stream("depth"), mimetype="multipart/x-mixed-replace; boundary=frame")

# ==========================
# Main
# ==========================
def main():
    threading.Thread(target=sender_loop, daemon=True).start()
    threading.Thread(target=gamepad_loop_js, daemon=True).start()
    threading.Thread(target=telemetry_loop, daemon=True).start()

    if USE_ZED:
        threading.Thread(target=zed_video_loop, daemon=True).start()
    else:
        threading.Thread(target=usb_video_loop, daemon=True).start()

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()
