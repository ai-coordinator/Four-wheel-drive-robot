# -*- coding: utf-8 -*-
"""
Robot Remote + Gamepad + (ZED RGB/Depth or USB) + Jetson Telemetry Dashboard

変更点（今回）:
- ★MJPEG機能を完全削除（/video/* を削除。Snapshot /frame/*.jpg のみ）
- Camera: RGBを領域いっぱい表示、Depthを左下PiP表示＋表示/非表示（Snapshotのみ）
- Jetson Telemetry: gauge/チップ表示、右側が切れないようレイアウト修正
- Control: ジョイスティックをより“カッコよく”
- Status: Last sent L/R を“レーダー”表示（左右反転バグ修正、レーダー風デザイン）
"""

import time
import threading
import serial
import os
import struct
import select
import subprocess
import re
import shutil
import glob

import numpy as np
import cv2
from flask import Flask, request, jsonify, Response, render_template, make_response

import io
import qrcode


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

AX_TURN = 0
AX_THROTTLE = 1
BTN_STOP = 0

NEUTRAL = 0.06

# ========= Arbitration / Safety =========
REMOTE_TIMEOUT = 0.30
GAMEPAD_TIMEOUT = 0.25
FAILSAFE_STOP = 1.2

# ========= Video tuning =========
USE_ZED = True  # True: ZED SDK, False: USB cam fallback

VIDEO_FPS = 10
RGB_W, RGB_H = 640, 360
JPEG_QUALITY_RGB = 70

DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 5.0
DEPTH_FPS = 3
JPEG_QUALITY_DEPTH = 60

# ========= Shared state =========
lock = threading.Lock()
state = {
    "remote_L": 0, "remote_R": 0, "remote_ts": 0.0,
    "pad_L": 0, "pad_R": 0, "pad_ts": 0.0,
    "active": "none",
    "last_sent_L": 0, "last_sent_R": 0,
    "last_write_ts": 0.0,
    "remote_last_cmd_ts": 0.0,
    "remote_seq": 0,
    "remote_client_t": 0.0,
    "remote_sid": "",
}

frame_lock = threading.Lock()
latest_rgb_jpeg = None
latest_depth_jpeg = None
latest_rgb_ts = 0.0
latest_depth_ts = 0.0

video_status = {"src": "none", "ok": False, "detail": "", "fps_est": 0.0, "fps_depth_est": 0.0}

depth_lock = threading.Lock()
depth_enabled = True  # UIから切替

tele_lock = threading.Lock()
telemetry = {"ok": False, "src": "none", "ts": 0.0, "data": {}, "detail": ""}


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

def fmt_age_s(ts, now):
    if not ts:
        return None
    return round(now - ts, 3)

def run_forever(name: str, fn):
    """Run fn() forever; if it crashes, restart after a short delay."""
    while True:
        try:
            fn()
        except Exception as e:
            print(f"[{name}] crashed: {e}")
            time.sleep(1.0)


# ==========================
# Gamepad loop
# ==========================
def gamepad_loop_js():
    """Read Linux joystick /dev/input/js0 with auto-reconnect."""
    turn_raw = 0.0
    throttle_raw = 0.0
    stop_pressed = False

    tick_hz = 50
    tick_dt = 1.0 / tick_hz

    fd = None
    while True:
        try:
            # (re)open if needed
            if fd is None:
                if not os.path.exists(JS_DEV):
                    # keep state neutral while device absent
                    with lock:
                        state["pad_L"] = 0
                        state["pad_R"] = 0
                    time.sleep(0.5)
                    continue

                fd = os.open(JS_DEV, os.O_RDONLY | os.O_NONBLOCK)
                print("[GAMEPAD] opened:", JS_DEV)

            now = time.time()

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

            engaged = (abs(turn_raw) > NEUTRAL) or (abs(throttle_raw) > NEUTRAL) or stop_pressed

            if engaged:
                throttle = curve(apply_deadzone(-throttle_raw))
                turn = curve(apply_deadzone(-turn_raw))
                L, R = mix_to_lr(throttle, turn)
                if stop_pressed:
                    L, R = 0, 0
                with lock:
                    state["pad_L"] = L
                    state["pad_R"] = R
                    state["pad_ts"] = now
            else:
                with lock:
                    state["pad_L"] = 0
                    state["pad_R"] = 0

        except (OSError, ValueError) as ex:
            # device unplugged or fd invalid -> reopen
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
                fd = None
            print("[GAMEPAD] error, retrying:", ex)
            time.sleep(0.5)
        except Exception as ex:
            # unexpected; keep loop alive
            print("[GAMEPAD] unexpected error:", ex)
            time.sleep(0.5)



# ==========================
# Serial sender loop
# ==========================
def sender_loop():
    """Send L/R to Arduino over serial with auto-reconnect + failsafe."""
    dt = 1.0 / SEND_HZ
    ser = None
    while True:
        try:
            if ser is None or (not ser.is_open):
                try:
                    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.01)
                    time.sleep(2.0)  # Arduino reset
                    print("[SERIAL] opened:", SERIAL_PORT)
                except Exception as e:
                    print("[SERIAL] open failed:", e)
                    ser = None
                    time.sleep(1.0)
                    continue

            now = time.time()
            with lock:
                remote_ok = (now - state["remote_ts"]) <= REMOTE_TIMEOUT if state["remote_ts"] else False
                pad_ok = (now - state["pad_ts"]) <= GAMEPAD_TIMEOUT if state["pad_ts"] else False

                if remote_ok:
                    L, R = state["remote_L"], state["remote_R"]
                    state["active"] = "remote"
                elif pad_ok:
                    L, R = state["pad_L"], state["pad_R"]
                    state["active"] = "gamepad"
                else:
                    # failsafe if both stale for too long
                    if (now - max(state["remote_ts"], state["pad_ts"])) > FAILSAFE_STOP:
                        L, R = 0, 0
                        state["active"] = "failsafe"
                    else:
                        L, R = 0, 0
                        state["active"] = "none"

                state["last_sent_L"], state["last_sent_R"] = L, R
                state["last_write_ts"] = now

            try:
                ser.write(f"{L},{R}\n".encode("ascii"))
            except Exception as e:
                print("[SERIAL] write error:", e)
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                time.sleep(0.5)

            time.sleep(dt)

        except Exception as ex:
            print("[SENDER] unexpected error:", ex)
            try:
                if ser:
                    ser.close()
            except Exception:
                pass
            ser = None
            time.sleep(0.5)



# ==========================
# Depth colormap
# ==========================
def depth_to_colormap(depth_m: np.ndarray, dmin=DEPTH_MIN_M, dmax=DEPTH_MAX_M):
    valid = np.isfinite(depth_m) & (depth_m > 0)
    depth = np.clip(depth_m, dmin, dmax)

    norm = (depth - dmin) / (dmax - dmin)
    norm[~valid] = 0.0
    img8 = (norm * 255).astype(np.uint8)

    img8 = 255 - img8
    cm = cv2.applyColorMap(img8, cv2.COLORMAP_JET)
    return cm


# ==========================
# ZED loop
# ==========================
def zed_video_loop():
    global latest_rgb_jpeg, latest_depth_jpeg, latest_rgb_ts, latest_depth_ts

    try:
        import pyzed.sl as sl
    except Exception as e:
        video_status.update({"src": "ZED", "ok": False, "detail": f"pyzed import failed: {e}"})
        print("[ZED] import failed:", e)
        return

    zed = sl.Camera()

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.VGA
    init.camera_fps = max(1, int(VIDEO_FPS))
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

    # out resolution hints (some SDK versions need sl.Resolution)
    try:
        out_res = sl.Resolution(RGB_W, RGB_H)
    except Exception:
        out_res = None

    last_rgb_t = time.time()
    fps_rgb_smooth = 0.0

    last_depth_gen_t = 0.0
    last_depth_tick = time.time()
    fps_depth_smooth = 0.0

    while True:
        if zed.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.005)
            continue

        # ---- RGB ----
        try:
            if out_res is not None:
                zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, out_res)
            else:
                zed.retrieve_image(image, sl.VIEW.LEFT)
        except Exception:
            zed.retrieve_image(image, sl.VIEW.LEFT)

        bgr = image.get_data()
        if bgr is None:
            continue

        if bgr.ndim == 3 and bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

        if bgr.shape[1] != RGB_W or bgr.shape[0] != RGB_H:
            bgr = cv2.resize(bgr, (RGB_W, RGB_H), interpolation=cv2.INTER_AREA)

        ts = f"{time.time():.3f}"
        cv2.putText(bgr, ts, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)

        rgb_jpg = encode_jpeg(bgr, JPEG_QUALITY_RGB)
        if rgb_jpg:
            tnow = time.time()
            with frame_lock:
                latest_rgb_jpeg = rgb_jpg
                latest_rgb_ts = tnow

        t = time.time()
        inst = 1.0 / max(1e-6, (t - last_rgb_t))
        last_rgb_t = t
        fps_rgb_smooth = 0.9 * fps_rgb_smooth + 0.1 * inst
        video_status["fps_est"] = round(fps_rgb_smooth, 1)

        # ---- Depth ----
        with depth_lock:
            de = depth_enabled

        if de and (t - last_depth_gen_t) >= (1.0 / max(1, DEPTH_FPS)):
            last_depth_gen_t = t
            try:
                if out_res is not None:
                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU, out_res)
                else:
                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            except Exception:
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            depth_m = depth.get_data()
            if depth_m is not None:
                if depth_m.ndim == 3:
                    depth_m = depth_m[:, :, 0]
                if depth_m.shape[1] != RGB_W or depth_m.shape[0] != RGB_H:
                    depth_m = cv2.resize(depth_m, (RGB_W, RGB_H), interpolation=cv2.INTER_NEAREST)

                depth_vis = depth_to_colormap(depth_m)
                cv2.putText(depth_vis, ts, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)

                dep_jpg = encode_jpeg(depth_vis, JPEG_QUALITY_DEPTH)
                if dep_jpg:
                    tnow = time.time()
                    with frame_lock:
                        latest_depth_jpeg = dep_jpg
                        latest_depth_ts = tnow

                tt = time.time()
                instd = 1.0 / max(1e-6, (tt - last_depth_tick))
                last_depth_tick = tt
                fps_depth_smooth = 0.9 * fps_depth_smooth + 0.1 * instd
                video_status["fps_depth_est"] = round(fps_depth_smooth, 1)

        time.sleep(0.001)


# ==========================
# USB fallback
# ==========================
def usb_video_loop():
    global latest_rgb_jpeg, latest_depth_jpeg, latest_rgb_ts

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        video_status.update({"src": "USB", "ok": False, "detail": "open failed"})
        return

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RGB_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RGB_H)
    cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)

    video_status.update({"src": "USB", "ok": True, "detail": "USB cam OK (no depth)"})

    last_t = time.time()
    fps_smooth = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue

        h, w = frame.shape[:2]
        if w >= h * 2:
            frame = frame[:, :w // 2]

        if frame.shape[1] != RGB_W or frame.shape[0] != RGB_H:
            frame = cv2.resize(frame, (RGB_W, RGB_H), interpolation=cv2.INTER_AREA)

        ts = f"{time.time():.3f}"
        cv2.putText(frame, ts, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)

        rgb_jpg = encode_jpeg(frame, JPEG_QUALITY_RGB)
        if rgb_jpg:
            tnow = time.time()
            with frame_lock:
                latest_rgb_jpeg = rgb_jpg
                latest_rgb_ts = tnow
                latest_depth_jpeg = None

        t = time.time()
        inst = 1.0 / max(1e-6, (t - last_t))
        last_t = t
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst
        video_status["fps_est"] = round(fps_smooth, 1)

        time.sleep(0.001)


# ==========================
# Jetson Telemetry
# ==========================

def _read_first_line(path: str):
    try:
        with open(path, "r") as f:
            return f.readline().strip()
    except Exception:
        return None

_cpu_prev = {"total": None, "idle": None, "ts": None}

def _cpu_percent_sample():
    """Return overall CPU utilization percent using /proc/stat deltas."""
    try:
        line = _read_first_line("/proc/stat")
        if not line or not line.startswith("cpu "):
            return None
        parts = [int(x) for x in line.split()[1:]]
        # user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice
        total = sum(parts)
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        now = time.time()
        if _cpu_prev["total"] is None:
            _cpu_prev.update({"total": total, "idle": idle, "ts": now})
            return None
        dt_total = total - _cpu_prev["total"]
        dt_idle = idle - _cpu_prev["idle"]
        _cpu_prev.update({"total": total, "idle": idle, "ts": now})
        if dt_total <= 0:
            return None
        util = 100.0 * (dt_total - dt_idle) / dt_total
        return round(max(0.0, min(100.0, util)), 1)
    except Exception:
        return None

def _cpu_freq_avg_mhz():
    """Average current CPU freq across cores (MHz) using sysfs."""
    freqs = []
    for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"):
        v = _read_first_line(p)
        if not v:
            continue
        try:
            khz = int(v)
            freqs.append(khz / 1000.0)
        except Exception:
            pass
    if freqs:
        return int(round(sum(freqs) / len(freqs)))
    return None

def _gpu_util_pct_sysfs():
    """Jetson GPU load percent from /sys/devices/gpu.0/load (0-255)."""
    v = _read_first_line("/sys/devices/gpu.0/load")
    if not v:
        return None
    try:
        load = int(v)
        return int(round(max(0, min(255, load)) * 100.0 / 255.0))
    except Exception:
        return None

def _temps_from_thermal_zones():
    temps = {}
    try:
        for tz in glob.glob("/sys/class/thermal/thermal_zone*"):
            t = _read_first_line(os.path.join(tz, "type")) or "tz"
            v = _read_first_line(os.path.join(tz, "temp"))
            if not v:
                continue
            try:
                mv = float(v)
                # many zones are in milliC
                c = mv / 1000.0 if mv > 200 else mv
                # filter obviously bogus
                if -20 <= c <= 140:
                    temps[t] = round(c, 1)
            except Exception:
                pass
    except Exception:
        pass

    if not temps:
        return None

    hot_name = max(temps, key=lambda k: temps[k])
    hot_val = temps[hot_name]
    return {"temps_c": temps, "temp_hot_name": hot_name, "temp_hot_c": hot_val, "temp_max_c": hot_val}

def read_proc_fallback():
    """Best-effort telemetry without tegrastats.
    Provides CPU/RAM/Disk/Uptime/Load + (if available) CPU freq, GPU load, temps.
    """
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

    # cpu util / freq
    cpu_pct = _cpu_percent_sample()
    if cpu_pct is not None:
        d["cpu_util_pct"] = cpu_pct
    cpu_f = _cpu_freq_avg_mhz()
    if cpu_f is not None:
        d["cpu_freq_avg_mhz"] = cpu_f

    # gpu util
    gpu_pct = _gpu_util_pct_sysfs()
    if gpu_pct is not None:
        d["gpu_util_pct"] = gpu_pct

    # temps
    tinfo = _temps_from_thermal_zones()
    if tinfo:
        d.update(tinfo)

    # meminfo
    try:
        mem = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                k = line.split(":")[0]
                v = line.split(":")[1].strip().split()[0]
                mem[k] = int(v)
        if "MemTotal" in mem and "MemAvailable" in mem:
            total = mem["MemTotal"] // 1024
            avail = mem["MemAvailable"] // 1024
            used = max(0, total - avail)
            d["ram_total_mb"] = int(total)
            d["ram_used_mb"] = int(used)
            d["ram_util_pct"] = round(100.0 * used / max(1, total), 1)
    except Exception:
        pass

    # swap
    try:
        mem = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("SwapTotal") or line.startswith("SwapFree"):
                    k = line.split(":")[0]
                    v = line.split(":")[1].strip().split()[0]
                    mem[k] = int(v)
        if "SwapTotal" in mem and "SwapFree" in mem:
            total = mem["SwapTotal"] // 1024
            free = mem["SwapFree"] // 1024
            used = max(0, total - free)
            d["swap_total_mb"] = int(total)
            d["swap_used_mb"] = int(used)
            d["swap_util_pct"] = round(100.0 * used / max(1, total), 1)
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

def _read_net_bytes():
    # returns (rx_bytes, tx_bytes) total across non-lo, non-docker-ish
    rx = 0
    tx = 0
    try:
        with open("/proc/net/dev", "r") as f:
            lines = f.read().splitlines()
        for line in lines[2:]:
            if ":" not in line:
                continue
            iface, rest = line.split(":", 1)
            iface = iface.strip()
            if iface == "lo":
                continue
            # ignore some virtuals (optional)
            if iface.startswith("docker") or iface.startswith("br-") or iface.startswith("veth"):
                continue
            cols = rest.split()
            if len(cols) >= 16:
                rx += int(cols[0])
                tx += int(cols[8])
    except Exception:
        return None
    return rx, tx

def parse_tegrastats_line(line: str):
    d = {}
    s = line.strip()

    # RAM
    m = re.search(r"RAM (\d+)\/(\d+)MB", s)
    if m:
        d["ram_used_mb"] = int(m.group(1))
        d["ram_total_mb"] = int(m.group(2))
        d["ram_util_pct"] = round(100.0 * d["ram_used_mb"] / max(1, d["ram_total_mb"]), 1)

    # SWAP
    m = re.search(r"SWAP (\d+)\/(\d+)MB", s)
    if m:
        d["swap_used_mb"] = int(m.group(1))
        d["swap_total_mb"] = int(m.group(2))
        d["swap_util_pct"] = round(100.0 * d["swap_used_mb"] / max(1, d["swap_total_mb"]), 1)

    # GPU util (optional @freq)
    m = re.search(r"GR3D_FREQ (\d+)%(@(\d+))?", s)
    if m:
        d["gpu_util_pct"] = int(m.group(1))
        if m.group(3):
            d["gpu_freq_mhz"] = int(m.group(3))

    # EMC util (optional @freq)
    m = re.search(r"EMC_FREQ (\d+)%(@(\d+))?", s)
    if m:
        d["emc_util_pct"] = int(m.group(1))
        if m.group(3):
            d["emc_freq_mhz"] = int(m.group(3))

    # CPU [x%@freq,...]
    m = re.search(r"CPU \[(.+?)\]", s)
    if m:
        parts = m.group(1).split(",")
        utils = []
        freqs = []
        for p in parts:
            m2 = re.search(r"(\d+)%@(\d+)", p)
            if m2:
                utils.append(int(m2.group(1)))
                freqs.append(int(m2.group(2)))
            else:
                m3 = re.search(r"(\d+)%", p)
                if m3:
                    utils.append(int(m3.group(1)))
        if utils:
            d["cpu_util_pct"] = round(sum(utils) / len(utils), 1)
        if freqs:
            d["cpu_freq_avg_mhz"] = int(round(sum(freqs) / len(freqs)))

    # Temps
    temps = {}
    for name, val in re.findall(r"([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C", s):
        temps[name] = float(val)
    if temps:
        d["temps_c"] = temps
        d["temp_max_c"] = max(temps.values()) if temps else None
        # "hot" sensor name
        hot_name = None
        hot_val = None
        for k, v in temps.items():
            if (hot_val is None) or (v > hot_val):
                hot_val = v
                hot_name = k
        d["temp_hot_name"] = hot_name
        d["temp_hot_c"] = hot_val

    d["raw"] = s
    return d


def _find_tegrastats():
    # systemd sometimes has a minimal PATH; try explicit locations too
    p = shutil.which("tegrastats")
    if p and os.path.exists(p):
        return p
    for cand in ["/usr/bin/tegrastats", "/usr/sbin/tegrastats", "/bin/tegrastats", "/sbin/tegrastats"]:
        if os.path.exists(cand):
            return cand
    return None

def telemetry_loop():
    """Continuously refresh global telemetry dict.

    Priority:
      1) tegrastats (most complete: CPU/GPU/EMC/temps)
      2) sysfs/proc fallback (CPU/GPU/temps/mem/disk/load/uptime)
    """
    last_net = _read_net_bytes()
    last_net_t = time.time()

    tegra = _find_tegrastats()
    if tegra:
        cmd = [tegra, "--interval", "1000"]
        try:
            # use line-buffered text mode
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            with tele_lock:
                telemetry.update({"ok": True, "src": "tegrastats", "detail": f"running: {tegra}"})

            while True:
                line = p.stdout.readline() if p.stdout else ""
                if not line:
                    # if process died, break and fall back
                    if p.poll() is not None:
                        raise RuntimeError("tegrastats exited")
                    time.sleep(0.2)
                    continue

                d = parse_tegrastats_line(line)
                # fill missing fields with proc/sysfs where possible
                d.update(read_proc_fallback())

                # net Mbps
                now = time.time()
                cur = _read_net_bytes()
                if cur and last_net:
                    dt = max(1e-6, now - last_net_t)
                    rx_bps = (cur[0] - last_net[0]) * 8.0 / dt
                    tx_bps = (cur[1] - last_net[1]) * 8.0 / dt
                    d["net_down_mbps"] = round(rx_bps / 1e6, 2)
                    d["net_up_mbps"] = round(tx_bps / 1e6, 2)
                last_net = cur or last_net
                last_net_t = now

                with tele_lock:
                    telemetry["ok"] = True
                    telemetry["ts"] = time.time()
                    telemetry["data"] = d

        except Exception as e:
            with tele_lock:
                telemetry.update({"ok": True, "src": "sysfs", "detail": f"tegrastats failed: {e}"})
    else:
        with tele_lock:
            telemetry.update({"ok": True, "src": "sysfs", "detail": "tegrastats not found"})

    # fallback loop
    while True:
        d = read_proc_fallback()
        now = time.time()
        cur = _read_net_bytes()
        if cur and last_net:
            dt = max(1e-6, now - last_net_t)
            rx_bps = (cur[0] - last_net[0]) * 8.0 / dt
            tx_bps = (cur[1] - last_net[1]) * 8.0 / dt
            d["net_down_mbps"] = round(rx_bps / 1e6, 2)
            d["net_up_mbps"] = round(tx_bps / 1e6, 2)
        last_net = cur or last_net
        last_net_t = now

        with tele_lock:
            telemetry["ok"] = True
            telemetry["ts"] = time.time()
            telemetry["data"] = d
        time.sleep(1.0)


# ==========================
# Flask server
# ==========================
app = Flask(__name__)

# ==========================
# System control (reboot / shutdown)
# ==========================
# Security note:
# - These endpoints execute reboot/shutdown on the host.
# - Recommended: protect with a key via env var ROBOT_CONSOLE_KEY and/or reverse-proxy auth.
ROBOT_CONSOLE_KEY = os.environ.get("ROBOT_CONSOLE_KEY", "")

def _auth_ok(req) -> bool:
    if not ROBOT_CONSOLE_KEY:
        return True
    # Accept header or query param for simplicity
    k = req.headers.get("X-Console-Key", "") or req.args.get("key", "")
    return k == ROBOT_CONSOLE_KEY

def _run_system_cmd(cmd: list):
    """Run a privileged system command via sudo -n. Returns (ok, detail)."""
    try:
        if os.geteuid() == 0:
            subprocess.Popen(cmd)
            return True, "started as root"
        # non-interactive sudo (requires NOPASSWD for this command)
        subprocess.Popen(["sudo", "-n"] + cmd)
        return True, "started via sudo -n"
    except Exception as e:
        return False, str(e)

INDEX_HTML = None  # moved to templates/index.html

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}



def _get_lan_ip_prefer_192() -> str:
    # Best-effort LAN IPv4 detection. Prefer 192.168.* if available.
    # 1) Ask kernel which source IP it would use to reach the internet
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        if ip:
            return ip
    except Exception:
        pass

    # 2) Parse `ip -4 addr` output and prefer 192.168.*
    try:
        import subprocess
        out = subprocess.check_output(['ip', '-4', 'addr'], text=True, stderr=subprocess.DEVNULL)
        ips = re.findall(r"inet\s+(\d+\.\d+\.\d+\.\d+)/\d+", out)
        for cand in ips:
            if cand.startswith('192.168.'):
                return cand
        # fallback to other private ranges
        for cand in ips:
            if cand.startswith('10.') or cand.startswith('172.'):
                return cand
        if ips:
            return ips[0]
    except Exception:
        pass

    return '127.0.0.1'

@app.get("/")
def index():
    resp = make_response(render_template("index.html",
        rgb_w=RGB_W, rgb_h=RGB_H,
        video_fps=VIDEO_FPS, depth_fps=DEPTH_FPS,
        depth_min=DEPTH_MIN_M, depth_max=DEPTH_MAX_M
    ))
    resp.headers.update(NO_CACHE_HEADERS)
    return resp
@app.post("/api/prefs")
def api_prefs():
    global depth_enabled, latest_depth_jpeg, latest_depth_ts
    data = request.get_json(force=True, silent=True) or {}
    v = bool(data.get("depth", True))
    with depth_lock:
        depth_enabled = v
    if not v:
        with frame_lock:
            latest_depth_jpeg = None
            latest_depth_ts = 0.0
    return jsonify({"ok": True, "depth": v})

@app.get("/api/status")
def api_status():
    now = time.time()
    with lock:
        s = {
            "active": state["active"],
            "remote": {"L": state["remote_L"], "R": state["remote_R"], "age_s": fmt_age_s(state["remote_ts"], now)},
            "gamepad": {"L": state["pad_L"], "R": state["pad_R"], "age_s": fmt_age_s(state["pad_ts"], now)},
            "last_sent": {"L": state["last_sent_L"], "R": state["last_sent_R"], "ts": state["last_write_ts"]},
            "video": dict(video_status),
        }
    with frame_lock:
        rgb_age_ms = int(1000 * (now - latest_rgb_ts)) if latest_rgb_ts else None
        s["video"]["rgb_age_ms"] = rgb_age_ms
    with tele_lock:
        s["jetson"] = dict(telemetry)
    with depth_lock:
        s["prefs"] = {"depth_enabled": depth_enabled}
    return jsonify(s)

@app.post("/api/cmd")
def api_cmd():
    data = request.get_json(force=True, silent=True) or {}

    # --- Anti-queue / reload-safe fields (browser -> server) ---
    # sid: page session id (changes on reload). Used to reset seq.
    # seq: increasing integer within the page session (optional)
    # t:   client ms epoch (Date.now()) for "late" detection
    sid = data.get("sid", "")
    try:
        sid = str(sid)[:64]
    except Exception:
        sid = ""

    seq = data.get("seq", 0)
    try:
        seq = int(seq)
    except Exception:
        seq = 0

    ct_ms = data.get("t", 0)
    try:
        ct_ms = float(ct_ms)
    except Exception:
        ct_ms = 0.0
    ct = ct_ms / 1000.0 if ct_ms else 0.0

    turn = float(data.get("turn", 0.0))
    throttle = float(data.get("throttle", 0.0))


    turn = max(-1.0, min(1.0, turn))
    throttle = max(-1.0, min(1.0, throttle))

    throttle = curve(apply_deadzone(throttle))
    turn = curve(apply_deadzone(-turn))  # 左右反転したいなら "-" を外す

    L, R = mix_to_lr(throttle, turn)

    now = time.time()
    with lock:
        # Reset sequence when browser reloads (sid changes)
        if sid and sid != state.get("remote_sid", ""):
            state["remote_sid"] = sid
            state["remote_seq"] = 0
            state["remote_client_t"] = 0.0

        # Drop commands that arrive "too late" (request queue / network stall).
        # This prevents "moving after you stop" due to a backlog.
        # NOTE: If stop packet is also delayed, REMOTE_TIMEOUT will stop the robot quickly.
        if ct:
            late_s = now - ct
            if late_s > 0.35:
                return jsonify({"ok": True, "ignored": True, "reason": "late", "late_s": round(late_s, 3)})

        # Drop stale seq within the same sid (optional safeguard)
        if seq and seq <= state.get("remote_seq", 0):
            return jsonify({"ok": True, "ignored": True, "reason": "stale_seq", "seq": seq, "last": state.get("remote_seq", 0)})

        # Drop stale client timestamps (out-of-order)
        if ct and ct < state.get("remote_client_t", 0.0):
            return jsonify({"ok": True, "ignored": True, "reason": "stale_time", "t": ct_ms})

        if sid:
            state["remote_sid"] = sid
        if seq:
            state["remote_seq"] = seq
        if ct:
            state["remote_client_t"] = ct

        state["remote_L"] = L
        state["remote_R"] = R
        state["remote_ts"] = now
        state["remote_last_cmd_ts"] = now

    return jsonify({"ok": True, "L": L, "R": R})

@app.post("/api/system/reboot")
def api_system_reboot():
    if not _auth_ok(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    ok, detail = _run_system_cmd(["/sbin/reboot"])
    return jsonify({"ok": ok, "detail": detail})

@app.post("/api/system/shutdown")
def api_system_shutdown():
    if not _auth_ok(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    ok, detail = _run_system_cmd(["/sbin/shutdown", "-h", "now"])
    return jsonify({"ok": ok, "detail": detail})




# ---- Snapshot endpoints only ----
@app.get("/frame/rgb.jpg")
def frame_rgb():
    with frame_lock:
        jpg = latest_rgb_jpeg
    if jpg is None:
        return Response(status=204, headers=NO_CACHE_HEADERS)
    return Response(jpg, mimetype="image/jpeg", headers=NO_CACHE_HEADERS)

@app.get("/frame/depth.jpg")
def frame_depth():
    with frame_lock:
        jpg = latest_depth_jpeg
    if jpg is None:
        return Response(status=204, headers=NO_CACHE_HEADERS)
    return Response(jpg, mimetype="image/jpeg", headers=NO_CACHE_HEADERS)


# ==========================
# Main
# ==========================
def main():
    # Run worker threads with auto-restart on unexpected exceptions.
    threading.Thread(target=lambda: run_forever("SENDER", sender_loop), daemon=True).start()
    threading.Thread(target=lambda: run_forever("GAMEPAD", gamepad_loop_js), daemon=True).start()
    threading.Thread(target=lambda: run_forever("TELEMETRY", telemetry_loop), daemon=True).start()

    if USE_ZED:
        threading.Thread(target=lambda: run_forever("VIDEO_ZED", zed_video_loop), daemon=True).start()
    else:
        threading.Thread(target=lambda: run_forever("VIDEO_USB", usb_video_loop), daemon=True).start()

    # Flask itself handles per-request exceptions; keep debug off for stability.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


@app.get("/qr.png")
def qr_png():
    # If the console is opened as "localhost", QR would become unreachable from phones.
    # So we generate a LAN-reachable URL (prefer 192.168.*) while keeping the same port.
    host = request.host  # e.g. "localhost:5000" or "192.168.1.92:5000"
    if ":" in host:
        base_host, port = host.rsplit(":", 1)
        port_part = ":" + port
    else:
        base_host, port_part = host, ""
    if base_host in ("localhost", "127.0.0.1", "0.0.0.0"):
        base_host = _get_lan_ip_prefer_192()
    url = f"http://{base_host}{port_part}/"

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=6,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    return Response(data, mimetype="image/png", headers=NO_CACHE_HEADERS)
if __name__ == "__main__":
    main()
