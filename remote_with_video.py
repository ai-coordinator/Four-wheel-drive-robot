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

INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot Console</title>
<style>
:root{
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
}

*{box-sizing:border-box}
html, body { height:100%; }
body{
  margin:0; padding:0;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  color:var(--text);
  background:
    radial-gradient(1200px 600px at 10% 0%, rgba(122,162,255,.16), transparent 60%),
    radial-gradient(900px 500px at 100% 30%, rgba(43,228,167,.10), transparent 55%),
    radial-gradient(900px 500px at 40% 120%, rgba(255,77,109,.10), transparent 60%),
    var(--bg);
}

header{
  position:sticky; top:0; z-index:10;
  backdrop-filter: blur(14px);
  background: rgba(10,14,22,.72);
  border-bottom:1px solid var(--border);
}
.topbar{
  max-width: 1400px;
  margin:0 auto;
  padding:14px 16px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
}
.brand{
  display:flex; align-items:center; gap:10px;
  font-weight:800; letter-spacing:.2px;
  flex-wrap:wrap;
}
.badge{
  display:inline-flex; align-items:center; gap:8px;
  padding:8px 10px;
  border-radius:999px;
  background: var(--panel);
  border:1px solid var(--border);
  box-shadow: var(--shadow);
  font-size:13px;
  white-space:nowrap;
}

.badgeWide{
  width:100%;
  justify-content:center;
  text-align:center;
  white-space:normal;
}

@media (max-width: 520px){
  .camStage{ position:relative; width:100%; height:auto; min-height: 420px; overflow:hidden; aspect-ratio: __RGB_W__/__RGB_H__; max-height: 70vh; }
  .pip{ width: 38%; min-width:140px; }
}
.sysBtns{ display:inline-flex; gap:8px; align-items:center; }
.sysBtn{
  border:none;
  padding:8px 10px;
  border-radius:999px;
  background: rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.14);
  color: var(--text);
  font-weight:900;
  cursor:pointer;
  font-size:13px;
  white-space:nowrap;
}
.sysBtn:hover{ background: rgba(255,255,255,.14); }
.sysBtn.danger{
  background: rgba(255,77,109,.12);
  border-color: rgba(255,77,109,.30);
}
.sysBtn.danger:hover{ background: rgba(255,77,109,.20); }


.dot{
  width:10px; height:10px; border-radius:50%;
  background: var(--muted);
  box-shadow:0 0 0 4px rgba(255,255,255,.06);
}
.dot.good{ background: var(--good); box-shadow:0 0 0 4px rgba(43,228,167,.14); }
.dot.warn{ background: var(--warn); box-shadow:0 0 0 4px rgba(255,209,102,.14); }
.dot.bad { background: var(--bad);  box-shadow:0 0 0 4px rgba(255,77,109,.14); }

main{ max-width:1400px; margin:0 auto; padding:16px; }
.grid{
  display:grid;
  grid-template-columns: minmax(280px, 320px) minmax(0, 1fr) minmax(320px, 360px);
  grid-template-areas:
    "control camera tele"
    "status  status  status";
  gap:14px;
  align-items: stretch;
}
.cardControl{ grid-area: control; }
.cardCamera{ grid-area: camera; }
.cardTele{ grid-area: tele; }
.cardStatus{ grid-area: status; }

/* ============================
   Responsive (SIMPLIFIED)
   - Desktop: 3 columns (control / camera / tele) + status full width
   - Tablet/Mid: camera full width, then (control | tele), then status
   - Phone: 1 column stack
   ============================ */

/* Tablet / mid screens (this fixes "1050x911 -> only status" type breakage) */
@media (max-width: 1199px){
  .grid{
    grid-template-columns: minmax(280px, 1fr) minmax(280px, 1fr);
    grid-template-areas:
      "camera  camera"
      "control tele"
      "status  status";
    align-items: start;
  }

  /* Keep joystick usable and square */
  #pad{
    width:100%;
    aspect-ratio: 1 / 1 !important;
    height: auto !important;
    max-width: min(92vw, 420px);
    margin-inline: auto;
  }
}

/* Phone */
@media (max-width: 719px){
  main{ padding:12px; }
  .grid{
    grid-template-columns: 1fr;
    grid-template-areas:
      "camera"
      "control"
      "tele"
      "status";
    gap:12px;
  }

  /* Avoid camera stage being too tall on small screens */
  .camStage{ min-height: unset; max-height: 44vh; }
  .teleGrid{ grid-template-columns: 1fr; }
  .radar{ width:190px; height:190px; flex:0 0 190px; }
}

.card{
  background: var(--panel);
  border:1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  overflow:hidden;
  display:flex;
  flex-direction:column;
  min-width:0; /* ★これがないと右側が切れやすい */
}
.card .hd{
  padding:12px 12px 10px 12px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  border-bottom:1px solid rgba(255,255,255,.08);
  background: rgba(255,255,255,.03);
  gap:12px;
  flex-wrap:wrap;
}
.card .hd b{ font-size:13px; letter-spacing:.3px; }
.card .bd{ padding:12px; flex:1; min-height:0; min-width:0; }
.small{ color:var(--muted); font-size:12px; }

/* ===== CONTROL (more stylish pad) ===== */
.padWrap{ display:flex; flex-direction:column; gap:12px; }
#pad{
  width:100%;
  aspect-ratio:1/1;
  border-radius: 26px;
  position:relative;
  touch-action:none;
  user-select:none;
  overflow:hidden;

  border:1px solid rgba(255,255,255,.14);
  background:
    /* subtle noise/tint */
    radial-gradient(900px 450px at 20% 10%, rgba(122,162,255,.26), transparent 55%),
    radial-gradient(700px 380px at 90% 60%, rgba(43,228,167,.12), transparent 55%),
    rgba(0,0,0,.18);
  box-shadow:
    inset 0 0 0 1px rgba(255,255,255,.06),
    inset 0 0 40px rgba(0,0,0,.35);
}
#pad:before{
  content:"";
  position:absolute; inset:0;
  background:
    /* grid rings */
    repeating-radial-gradient(circle at 50% 50%,
      rgba(255,255,255,.10) 0 1px,
      transparent 1px 18px
    ),
    /* crosshair */
    linear-gradient(to right, transparent 49.7%, rgba(255,255,255,.10) 50%, transparent 50.3%),
    linear-gradient(to bottom, transparent 49.7%, rgba(255,255,255,.10) 50%, transparent 50.3%);
  opacity:.35;
  pointer-events:none;
}
#pad:after{
  content:"";
  position:absolute; inset:-40%;
  background:
    conic-gradient(from 0deg,
      rgba(122,162,255,.00) 0deg,
      rgba(122,162,255,.12) 40deg,
      rgba(122,162,255,.00) 90deg,
      rgba(122,162,255,.00) 360deg
    );
  animation: padSpin 4.0s linear infinite;
  opacity:.55;
  pointer-events:none;
  filter: blur(.2px);
}

#dot{
  width:22px;height:22px;
  border-radius:50%;
  position:absolute;
  left:50%; top:50%;
  transform: translate(-50%,-50%);
  background: rgba(255,255,255,.92);
  box-shadow:
    0 0 0 10px rgba(122,162,255,.18),
    0 0 28px rgba(122,162,255,.25),
    0 10px 22px rgba(0,0,0,.35);
}
.padCenter{
  position:absolute; left:50%; top:50%;
  width:10px; height:10px;
  border-radius:50%;
  transform: translate(-50%,-50%);
  background: rgba(255,255,255,.25);
  box-shadow: 0 0 0 8px rgba(255,255,255,.04);
  pointer-events:none;
}

@keyframes padSpin{ to { transform: rotate(360deg); } }

.controls{ display:flex; gap:10px; }
.btn{
  flex:1;
  border:none;
  padding:12px 12px;
  border-radius:14px;
  background: rgba(255,255,255,.10);
  color:var(--text);
  font-weight:800;
  cursor:pointer;
  border:1px solid rgba(255,255,255,.14);
}
.btn:hover{ background: rgba(255,255,255,.14); }
.btnStop{
  background: rgba(255,77,109,.14);
  border:1px solid rgba(255,77,109,.35);
}
.btnStop:hover{ background: rgba(255,77,109,.22); }

.kbd{ display:flex; gap:8px; flex-wrap:wrap; }
.key{
  padding:7px 9px;
  border-radius:10px;
  border:1px solid rgba(255,255,255,.12);
  background: rgba(255,255,255,.06);
  font-size:12px;
  color: var(--muted);
}

/* ===== CAMERA ===== */
.camTools{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.toggle{
  display:inline-flex; align-items:center; gap:8px;
  padding:8px 10px; border-radius:999px; background: var(--panel);
  border:1px solid var(--border); font-size:13px;
}
.toggle input{ transform: scale(1.1); }

.camStage{ position:relative; width:100%; height:auto; min-height: 420px; overflow:hidden; aspect-ratio: __RGB_W__/__RGB_H__; max-height: 70vh; }
.streamMain{
  width:100%; height:100%;
  /* Keep ZED aspect ratio (no cropping) */
  object-fit: contain;
  border-radius:14px;
  display:block;
  background: rgba(0,0,0,.25);
}
.pip{
  position:absolute;
  z-index:5;
  left:12px; bottom:12px;
  width: 32%;
  max-width:260px;
  min-width:160px;
  border-radius:14px;
  overflow:hidden;
  border:1px solid rgba(255,255,255,.16);
  box-shadow: 0 12px 26px rgba(0,0,0,.45);
  background: rgba(0,0,0,.25);
}
.pip img{ width:100%; height:100%; display:block; object-fit: contain; }


/* ===== Mobile Depth PiP harden (overlay fix) =====
   iOS/Safari で PiP が重ならず “下に落ちる” ケース対策。
   機能は触らず、レイアウト(CSS)のみで強制的に重ねる。
*/
@media (max-width: 700px){
  /* iOS/Safari で PiP が“下に落ちる”ケース対策: 位置と積層を強制 */
  .camStage{ position:relative; width:100%; height:auto; min-height: 240px; overflow:hidden; aspect-ratio: __RGB_W__/__RGB_H__; max-height: 44vh; }
  #rgbImg.streamMain{ display:block; width:100% !important; height:100% !important; object-fit:contain !important; }
  #depthPip.pip{ position:absolute !important; left:12px !important; bottom:12px !important; z-index:30 !important; display:block; }
  #depthPip.pip img{ width:100% !important; height:100% !important; object-fit:contain !important; display:block; }
}


/* ===== TELEMETRY ===== */
.teleGrid{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap:10px;
}
@media (max-width: 1180px){
  .teleGrid{ grid-template-columns: 1fr 1fr; }
}
@media (max-width: 520px){
  .teleGrid{ grid-template-columns: 1fr; }
}

.tile{
  border-radius:14px;
  border:1px solid rgba(255,255,255,.10);
  background: rgba(0,0,0,.12);
  padding:12px;
  display:flex;
  gap:12px;
  align-items:center;
  min-width:0;
}
.gauge{
  width:60px; height:60px;
  border-radius:50%;
  background:
    radial-gradient(circle at 50% 50%, rgba(255,255,255,.07) 0 45%, transparent 46%),
    conic-gradient(from -90deg, rgba(122,162,255,.95) 0deg, rgba(43,228,167,.95) var(--ang, 0deg), rgba(255,255,255,.10) var(--ang, 0deg) 360deg);
  position:relative;
  flex:0 0 60px;
  overflow:hidden;
  border:1px solid rgba(255,255,255,.10);
  box-shadow: inset 0 0 0 6px rgba(0,0,0,.22);
}
.gauge:before{
  content:"";
  position:absolute; inset:8px;
  border-radius:50%;
  background: rgba(10,14,22,.85);
  border:1px solid rgba(255,255,255,.08);
}
.gaugeVal{
  position:absolute; left:50%; top:50%;
  transform: translate(-50%,-50%);
  font-weight:900;
  font-size:13px;
  color: rgba(255,255,255,.88);
}
.tile .meta{ min-width:0; }
.tile .lbl{ font-size:12px; color: var(--muted); font-weight:800; letter-spacing:.08em; }
.tile .big{ font-size:22px; font-weight:900; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.tile .sub{ font-size:12px; color: rgba(255,255,255,.72); font-weight:800; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

.pills{
  margin-top:12px;
  display:grid;
  grid-template-columns: 1fr 1fr;
  grid-auto-rows: minmax(44px, auto);
  gap:10px;
  align-items:stretch;
}
.pills .pill--span2{ grid-column: 1 / -1; }
@media (max-width: 560px){
  .pills{ grid-template-columns: 1fr; }
  .pills .pill--span2{ grid-column: auto; }
}

.pill{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  padding:10px 14px;
  border-radius:999px;
  background: rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,.10);
  color: rgba(255,255,255,.88);
  font-weight:900;
  min-width:0;
}
.pill .k{ color: var(--muted); font-weight:900; white-space:nowrap; }
.pill .v{
  display:flex;
  flex-wrap:wrap;
  justify-content:flex-end;
  column-gap:10px;
  row-gap:2px;
  text-align:right;
  min-width:0;
  overflow-wrap:anywhere;
  font-variant-numeric: tabular-nums;
}
.pill.hot .v{ overflow-wrap:anywhere; }
.pill.good{ border-color: rgba(43,228,167,.18); }
.pill.warn{ border-color: rgba(255,209,102,.18); }
.pill.bad{ border-color: rgba(255,77,109,.22); }

.tempChips{
  margin-top:10px;
  display:grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap:10px;
}
@media (max-width: 560px){
  .tempChips{ grid-template-columns: 1fr; }
}
.tempRow{
  padding:10px 12px;
  border-radius:14px;
  background: rgba(0,0,0,.12);
  border:1px solid rgba(255,255,255,.10);
  display:flex;
  flex-direction:column;
  gap:8px;
}
.tempTop{
  display:flex;
  align-items:baseline;
  justify-content:space-between;
  gap:10px;
}
.tempName{
  flex:1;
  font-weight:900;
  font-size:12px;
  color: rgba(255,255,255,.86);
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}
.tempVal{
  font-weight:900;
  font-size:12px;
  color: rgba(255,255,255,.92);
  text-align:right;
  white-space:nowrap;
}
.tempBar{
  position:relative;
  height:10px;
  border-radius:999px;
  background: rgba(255,255,255,.10);
  overflow:hidden;
}
.tempFill{
  position:absolute;
  left:0; top:0; bottom:0;
  width:0%;
  border-radius:999px;
  background: rgba(122,162,255,.85);
}
.tempRow.good .tempFill{ background: rgba(43,228,167,.90); }
.tempRow.warn .tempFill{ background: rgba(255,209,102,.90); }
.tempRow.bad  .tempFill{ background: rgba(255,77,109,.92); }
/* ===== STATUS ===== */
.statGrid{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap:10px;
}
@media (max-width: 520px){
  .statGrid{ grid-template-columns: 1fr; }
}
.metric{
  padding:10px;
  border-radius:14px;
  border:1px solid rgba(255,255,255,.10);
  background: rgba(0,0,0,.12);
  min-width:0;
}
.metric .lbl{ font-size:12px; color:var(--muted); font-weight:800; }
.metric .val{ font-size:18px; font-weight:900; margin-top:3px; }

/* Radar more radar-like */
.radarWrap{
  margin-top:12px;
  display:flex;
  gap:14px;
  align-items:flex-start;
  flex-wrap:wrap;
}
.radar{
  width:220px; height:220px; flex:0 0 220px;
  border-radius:999px;
  border:1px solid rgba(130,255,180,.22);
  background:
    radial-gradient(circle at 20% 30%, rgba(120,255,180,.08), transparent 35%),
    radial-gradient(circle at 80% 70%, rgba(120,255,180,.06), transparent 40%),
    repeating-radial-gradient(circle at 50% 50%,
      rgba(120,255,180,.12) 0 1px,
      transparent 1px 22px
    ),
    linear-gradient(to right,
      transparent 49.6%, rgba(120,255,180,.12) 50%, transparent 50.4%
    ),
    linear-gradient(to bottom,
      transparent 49.6%, rgba(120,255,180,.12) 50%, transparent 50.4%
    ),
    radial-gradient(circle at 50% 50%, rgba(20,40,28,.85), rgba(0,0,0,.55) 65%, rgba(0,0,0,.35)),
    rgba(0,0,0,.20);
  position:relative;
  overflow:hidden;
  box-shadow:
    inset 0 0 0 10px rgba(120,255,180,.03),
    0 14px 28px rgba(0,0,0,.35);
}
.radar:before{
  content:"";
  position:absolute; inset:-60%;
  background:
    conic-gradient(from 0deg,
      rgba(120,255,180,.00) 0deg,
      rgba(120,255,180,.20) 18deg,
      rgba(120,255,180,.00) 38deg,
      rgba(120,255,180,.00) 360deg
    );
  animation: radarSpin 1.8s linear infinite;
  filter: blur(0.2px);
  opacity:.85;
}
.radar:after{
  content:"";
  position:absolute; inset:0;
  background:
    repeating-linear-gradient(to bottom,
      rgba(255,255,255,.03) 0 1px,
      transparent 1px 4px
    );
  opacity:.20;
  mix-blend-mode: overlay;
  pointer-events:none;
}
.radarGlow{
  position:absolute; inset:10px;
  border-radius:999px;
  border:1px solid rgba(120,255,180,.10);
  box-shadow: inset 0 0 18px rgba(120,255,180,.08);
  pointer-events:none;
  z-index:2;
}
.radarCenter{
  position:absolute; left:50%; top:50%;
  width:6px;height:6px;border-radius:50%;
  transform: translate(-50%,-50%);
  background: rgba(180,255,210,.45);
  box-shadow: 0 0 0 6px rgba(120,255,180,.06);
  z-index:3;
}
.radarDot{
  position:absolute;
  width:12px; height:12px;
  border-radius:50%;
  left:50%; top:50%;
  transform: translate(-50%,-50%);
  background: rgba(140,255,190,.95);
  box-shadow:
    0 0 0 7px rgba(120,255,180,.16),
    0 0 22px rgba(120,255,180,.45);
  z-index:4;
  animation: blip 1.2s ease-in-out infinite;
}
.radarLabelN, .radarLabelE, .radarLabelS, .radarLabelW{
  position:absolute;
  font-size:11px;
  color: rgba(160,255,210,.62);
  font-weight:900;
  letter-spacing:.12em;
  z-index:5;
  user-select:none;
}
.radarLabelN{ left:50%; top:10px; transform:translateX(-50%); }
.radarLabelS{ left:50%; bottom:10px; transform:translateX(-50%); }
.radarLabelW{ left:12px; top:50%; transform:translateY(-50%); }
.radarLabelE{ right:12px; top:50%; transform:translateY(-50%); }

@keyframes radarSpin{ to{ transform: rotate(360deg);} }
@keyframes blip{
  0%,100%{ transform: translate(-50%,-50%) scale(1); opacity:.95; }
  50%{ transform: translate(-50%,-50%) scale(1.25); opacity:.70; }
}

.log{
  height:180px;
  overflow:auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size:12px;
  color: rgba(255,255,255,.78);
  background: rgba(0,0,0,.18);
  border:1px solid rgba(255,255,255,.10);
  border-radius:14px;
  padding:10px;
  line-height:1.4;
}


/* keep Depth PiP overlay in every mode */
#depthPip{ position:absolute !important; left:12px !important; bottom:12px !important; z-index:50 !important;
  width: clamp(140px, 30%, 260px) !important; max-width: 42% !important; min-width: 140px !important;
}

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
      <span class="badge sysBtns">
        <button class="sysBtn" onclick="doReboot()">REBOOT</button>
        <button class="sysBtn danger" onclick="doShutdown()">SHUTDOWN</button>
      </span>
    </div>

  </div>
</header>

<main>
  <div class="grid">

    <!-- LEFT: CONTROL -->
    <section class="card cardControl">
      <div class="hd"><b>CONTROL</b><span class="small">touch / keyboard / gamepad</span></div>
      <div class="bd padWrap">
        <div id="pad">
          <div class="padCenter"></div>
          <div id="dot"></div>
        </div>

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

        </div>
    </section>

    <!-- MID: CAMERA -->
    <section class="card cardCamera">
      <div class="hd">
        <b>CAMERA</b>
        <div class="camTools">
          <span class="small">__RGB_W__×__RGB_H__ | RGB __VIDEO_FPS__fps / Depth __DEPTH_FPS__fps</span>
          <label class="toggle">
            <input id="toggleDepth" type="checkbox" checked>
            Depth
          </label>
        </div>
      </div>

      <div class="bd">
        <div class="camStage">
          <img id="rgbImg" class="streamMain" alt="rgb">
          <div id="depthPip" class="pip">
            <img id="depthImg" alt="depth">
          </div>
        </div>
        <div class="small" style="margin-top:10px;">Depth range: __DEPTH_MIN_M__m - __DEPTH_MAX_M__m</div>
      </div>
    </section>

    <!-- RIGHT: JETSON -->
    <section class="card cardTele">
      <div class="hd"><b>JETSON TELEMETRY</b><span class="small" id="teleTs">-</span></div>
      <div class="bd">
        <div class="teleGrid">
          <div class="tile">
            <div class="gauge"><div id="gCpu" class="gaugeVal">-</div></div>
            <div class="meta">
              <div class="lbl">CPU</div>
              <div class="big"><span id="cpuUtil">-</span>%</div>
              <div class="sub">avg freq <span id="cpuFreq">-</span> MHz</div>
            </div>
          </div>

          <div class="tile">
            <div class="gauge"><div id="gGpu" class="gaugeVal">-</div></div>
            <div class="meta">
              <div class="lbl">GPU</div>
              <div class="big"><span id="gpuUtil">-</span>%</div>
              <div class="sub">EMC <span id="emcUtil">-</span>%</div>
            </div>
          </div>

          <div class="tile">
            <div class="gauge"><div id="gRam" class="gaugeVal">-</div></div>
            <div class="meta">
              <div class="lbl">RAM</div>
              <div class="big"><span id="ramUsed">-</span>/<span id="ramTotal">-</span>MB</div>
              <div class="sub">SWAP <span id="swapUsed">-</span>/<span id="swapTotal">-</span>MB</div>
            </div>
          </div>

          <div class="tile">
            <div class="gauge"><div id="gDisk" class="gaugeVal">-</div></div>
            <div class="meta">
              <div class="lbl">Disk (/)</div>
              <div class="big"><span id="diskUsed">-</span>/<span id="diskTotal">-</span>GB</div>
              <div class="sub">free <span id="diskFree">-</span>GB</div>
            </div>
          </div>
        </div>

        <div class="pills">
          <div class="pill good"><span class="k">Uptime:</span><span class="v"><span id="uptime">-</span></span></div>
          <div class="pill"><span class="k">Load:</span><span class="v"><span id="loadavg">-</span></span></div>

          <div class="pill warn pill--span2 hot"><span class="k">Hot:</span><span class="v"><span id="hotSpot">-</span></span></div>

          <div class="pill"><span class="k">NET:</span><span class="v"><span>↓ <span id="netDown">-</span></span><span>↑ <span id="netUp">-</span> Mbps</span></span></div>
          <div class="pill"><span class="k">Src:</span><span class="v"><span id="teleSrc">-</span></span></div>
        </div>

        <div class="tempChips" id="tempChips"></div>
      </div>
    </section>


    <!-- STATUS -->
    <section class="card cardStatus">
    <div class="hd"><b>STATUS</b><span class="small">active source / last sent</span></div>
    <div class="bd">
      <div class="badge badgeWide">
                <span class="small">Remote</span>&nbsp;<b id="ageRemote">-</b>s
                &nbsp;|&nbsp;<span class="small">Pad</span>&nbsp;<b id="agePad">-</b>s
                &nbsp;|&nbsp;<span class="small">RGB FPS</span>&nbsp;<b id="fpsVideo">-</b>
                &nbsp;|&nbsp;<span class="small">Depth FPS</span>&nbsp;<b id="fpsDepth">-</b>
                &nbsp;|&nbsp;<span class="small">RGB age</span>&nbsp;<b id="rgbAge">-</b>ms
              </div>

      <div class="statGrid">
        <div class="metric">
          <div class="lbl">Active</div>
          <div class="val" id="activeSrc">-</div>
        </div>
        <div class="metric">
          <div class="lbl">Last sent L / R</div>
          <div class="val"><span id="lastL">-</span> / <span id="lastR">-</span></div>
        </div>
      </div>

      <div class="radarWrap">
        <div class="radar">
          <div class="radarGlow"></div>
          <div class="radarLabelN">N</div>
          <div class="radarLabelE">E</div>
          <div class="radarLabelS">S</div>
          <div class="radarLabelW">W</div>
          <div class="radarCenter"></div>
          <div id="radarDot" class="radarDot"></div>
        </div>

        <div style="flex:1; min-width:260px;">
          <div class="metric" style="margin-bottom:10px;">
            <div class="lbl">Radar meaning</div>
            <div class="val" style="font-size:14px; font-weight:900;">
              X = turn (right +), Y = throttle (forward +)
              <span class="small" style="display:block; margin-top:6px;">
                * Left/Right inversion fixed for visual sense
              </span>
            </div>
          </div>

          <div class="metric">
            <div class="lbl">Last sent (normalized)</div>
            <div class="val"><span id="stTurn">-</span> / <span id="stThr">-</span></div>
          </div>
        </div>
      </div>

      <div style="margin-top:12px;">
        <div class="small" style="margin-bottom:6px;">Event log</div>
        <div id="elog" class="log"></div>
      </div>
    </div>
  
    </section>

  </div>

  </main>

<script>
const pad=document.getElementById("pad");
const dot=document.getElementById("dot");
const elog=document.getElementById("elog");
let x=0,y=0,keys={w:false,a:false,s:false,d:false},dragging=false;
let _lastPollErr=0;
let _lastJsErr=0;
window.addEventListener('error', (ev)=>{const now=Date.now(); if(now-_lastJsErr>2000){_lastJsErr=now; try{log('JS error: '+(ev.message||ev.type));}catch(e){}}});

const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));
function updateDot(){
  const rect = pad.getBoundingClientRect();

  // Make full use of the pad area even if it becomes vertically long on phones.
  // We allow an ellipse: X uses pad width, Y uses pad height.
  const dotRect = dot.getBoundingClientRect();
  const dotR = Math.max(10, dotRect.width / 2); // fallback if not laid out yet
  const margin = 6;

  const rx = Math.max(0, rect.width  / 2 - (dotR + margin));
  const ry = Math.max(0, rect.height / 2 - (dotR + margin));

  dot.style.left = (rect.width/2  + x*rx) + "px";
  dot.style.top  = (rect.height/2 - y*ry) + "px";
}

function log(line){
  const t=new Date().toLocaleTimeString();
  elog.textContent = `[${t}] ${line}\n` + elog.textContent;
}

const SID = (Date.now().toString(36) + "_" + Math.random().toString(36).slice(2,10));
let cmdSeq = 0;

// --- Anti-queue + smooth control ---
// We never abort in-flight requests (that causes jerk). Instead we COALESCE:
// at most one request in flight, and we only keep the latest command pending.
let inFlight = false;
let pendingPayload = null;
let pendingForce = false;

// simple low-pass filter to reduce jitter from touch/trackpad
let fx = 0, fy = 0;
const SMOOTH = 0.35;      // 0..1 (higher = more responsive)
const SEND_MS = 25;       // ~40Hz when changing
const KEEPALIVE_MS = 50;  // ~20Hz when holding same
let lastSentAt = 0;
let lastSentX = 0, lastSentY = 0;

function pumpSend(){
  if(inFlight || !pendingPayload) return;
  inFlight = true;

  const payload = pendingPayload;
  const force = pendingForce;
  pendingPayload = null;
  pendingForce = false;

  lastSentAt = performance.now();
  lastSentX = payload.turn;
  lastSentY = payload.throttle;

  cmdSeq++;

  fetch("/api/cmd", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({turn:payload.turn, throttle:payload.throttle, sid:SID, seq:cmdSeq, t:Date.now()})
  }).catch(()=>{}).finally(()=>{
    inFlight = false;
    // if something arrived while sending, send it immediately (still coalesced)
    if(pendingPayload) pumpSend();
  });
}

function send(force=false){
  // filter input a bit (prevents tiny oscillations from feeling like jerk)
  fx = fx + (x - fx) * SMOOTH;
  fy = fy + (y - fy) * SMOOTH;

  const now = performance.now();
  const same = (Math.abs(fx-lastSentX) < 0.005 && Math.abs(fy-lastSentY) < 0.005);

  if(!force){
    const minDt = same ? KEEPALIVE_MS : SEND_MS;
    if(now - lastSentAt < minDt){
      // just remember latest; pump will send when allowed / when inFlight clears
      pendingPayload = {turn: fx, throttle: fy};
      return;
    }
  }

  pendingPayload = {turn: fx, throttle: fy};
  pendingForce = pendingForce || force;
  pumpSend();
}

// Keepalive loop: ensures "hold" keeps moving smoothly without creating a queue.
setInterval(()=>{
  if(dragging || Math.abs(x) > 0.01 || Math.abs(y) > 0.01) send(false);
}, 33);

function stop(){ x=0;y=0; updateDot(); send(true); log("STOP"); }
function center(){ x=0;y=0; updateDot(); }

function pointerToXY(ev){
  const rect=pad.getBoundingClientRect();
  const cx=rect.left+rect.width/2, cy=rect.top+rect.height/2;

  const dotRect = dot.getBoundingClientRect();
  const dotR = Math.max(10, dotRect.width / 2);
  const margin = 6;

  const rx = Math.max(1, rect.width  / 2 - (dotR + margin));
  const ry = Math.max(1, rect.height / 2 - (dotR + margin));

  x = clamp((ev.clientX - cx) / rx, -1, 1);
  y = clamp(-(ev.clientY - cy) / ry, -1, 1);

  updateDot();
}

pad.addEventListener("pointerdown",ev=>{dragging=true;pad.setPointerCapture(ev.pointerId);pointerToXY(ev);send();});
pad.addEventListener("pointermove",ev=>{if(!dragging)return;pointerToXY(ev);send();});
pad.addEventListener("pointerup",()=>{dragging=false; stop();});
pad.addEventListener("pointercancel",()=>{dragging=false; stop();});

window.addEventListener("blur", ()=>{ try{ stop(); }catch(e){} });
window.addEventListener("keydown",e=>{
  if(e.repeat) return;
  if(e.key==="w") keys.w=true;
  if(e.key==="a") keys.a=true;
  if(e.key==="s") keys.s=true;
  if(e.key==="d") keys.d=true;
  if(e.key===" ") stop();
});
window.addEventListener("keyup",e=>{
  if(e.key==="w") keys.w=false;
  if(e.key==="a") keys.a=false;
  if(e.key==="s") keys.s=false;
  if(e.key==="d") keys.d=false;
});

setInterval(()=>{
  let t=0,r=0;
  if(keys.w)t+=1; if(keys.s)t-=1; if(keys.d)r+=1; if(keys.a)r-=1;
  if(t!==0||r!==0){
    x=0.7*r; y=0.7*t; updateDot(); send();
  }
},50);

setInterval(()=>{
  const nonZero = (Math.abs(x)>0.02)||(Math.abs(y)>0.02);
  const activeKeys = keys.w||keys.a||keys.s||keys.d;
  if(dragging||activeKeys||nonZero) send();
},100);

function setConn(active){
  const dot=document.getElementById("connDot");
  const text=document.getElementById("connText");
  dot.className="dot";
  if(active==="remote"){ dot.classList.add("good"); text.textContent="REMOTE"; }
  else if(active==="gamepad"){ dot.classList.add("warn"); text.textContent="GAMEPAD"; }
  else if(active==="failsafe"){ dot.classList.add("bad"); text.textContent="FAILSAFE"; }
  else { text.textContent="IDLE"; }
}

async function postSystem(url, label){
  if(!confirm(`${label} を実行します。よろしいですか？`)) return;
  try{
    log(label + " requested");
    const res = await fetch(url, {method:"POST"});
    const j = await res.json().catch(()=> ({}));
    if(!res.ok || !j.ok){
      log(label + " failed: " + (j.error || j.detail || ("HTTP "+res.status)));
      alert(label + " 失敗: " + (j.error || j.detail || ("HTTP "+res.status)));
      return;
    }
    log(label + " started: " + (j.detail || "ok"));
    alert(label + " を開始しました");
  }catch(e){
    log(label + " error: " + e);
    alert(label + " エラー: " + e);
  }
}
function doReboot(){ postSystem("/api/system/reboot", "Reboot"); }
function doShutdown(){ postSystem("/api/system/shutdown", "Shutdown"); }

/* ===== Camera: Snapshot only ===== */
const rgbImg = document.getElementById("rgbImg");
const depthImg = document.getElementById("depthImg");
const depthPip = document.getElementById("depthPip");
const toggleDepth = document.getElementById("toggleDepth");

let depthOn = (localStorage.getItem("depthVisible") ?? "1") === "1";
toggleDepth.checked = depthOn;

function applyDepthVisible(v){
  depthOn = !!v;
  depthPip.style.display = depthOn ? "block" : "none";
  localStorage.setItem("depthVisible", depthOn ? "1" : "0");
}
async function setDepthServer(v){
  try{
    await fetch("/api/prefs", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({depth: !!v})
    });
  }catch(e){}
}
applyDepthVisible(depthOn);
setDepthServer(depthOn);

toggleDepth.addEventListener("change", ()=>{
  applyDepthVisible(toggleDepth.checked);
  setDepthServer(toggleDepth.checked);
  if(!depthOn){
    depthImg.removeAttribute("src");
  }
});

let rgbInFlight=false;
let depInFlight=false;
let rgbUrl=null;
let depUrl=null;

async function fetchBlob(url){
  const res = await fetch(url, {cache:"no-store"});
  if(!res.ok) throw new Error("HTTP "+res.status);
  return await res.blob();
}

async function tickSnapshot(){
  if(!rgbInFlight){
    rgbInFlight=true;
    fetchBlob("/frame/rgb.jpg?t="+Date.now())
      .then(b=>{
        if(rgbUrl) URL.revokeObjectURL(rgbUrl);
        rgbUrl = URL.createObjectURL(b);
        rgbImg.src = rgbUrl;
      })
      .catch(()=>{})
      .finally(()=>{rgbInFlight=false;});
  }

  if(depthOn && !depInFlight){
    depInFlight=true;
    fetchBlob("/frame/depth.jpg?t="+Date.now())
      .then(b=>{
        if(depUrl) URL.revokeObjectURL(depUrl);
        depUrl = URL.createObjectURL(b);
        depthImg.src = depUrl;
      })
      .catch(()=>{})
      .finally(()=>{depInFlight=false;});
  }
}
// 30fps相当で“常に最新だけ”取りに行く
setInterval(tickSnapshot, 33);

/* ===== Gauges ===== */
function setGauge(elId, pct){
  const el = document.getElementById(elId);
  if(!el) return;
  pct = Math.max(0, Math.min(100, (pct===0?0:(pct||0))));
  el.textContent = Math.round(pct);
  // conic-gradient の終端角をCSS変数で更新（0-100% -> 0-360deg）
  const ang = (pct * 3.6).toFixed(1) + "deg";
  const ring = el.parentElement; // .gauge
  if(ring) ring.style.setProperty("--ang", ang);
}

/* ===== Radar ===== */
function setRadarFromLR(L, R){
  // throttle=(L+R)/2, turn=(R-L)/2 ←左右反転修正済み
  const thr = ((L + R) / 2) / 127.0;
  const trn = ((R - L) / 2) / 127.0;

  const x = Math.max(-1, Math.min(1, trn));
  const y = Math.max(-1, Math.min(1, thr));

  document.getElementById("stThr").textContent = y.toFixed(2);
  document.getElementById("stTurn").textContent = x.toFixed(2);

  const radar = document.querySelector(".radar");
  const dot = document.getElementById("radarDot");
  const rect = radar.getBoundingClientRect();
  const cx = rect.width/2;
  const cy = rect.height/2;
  const r = Math.min(cx, cy) - 18;

  const px = cx + x * r;
  const py = cy - y * r;

  dot.style.left = px + "px";
  dot.style.top  = py + "px";
}

function renderTemps(temps){
  const box = document.getElementById("tempChips");
  box.innerHTML = "";
  if(!temps) return;

  // temps: object {name: valueC}
  const keys = Object.keys(temps);

  // sort by temp desc (unknown/NaN last)
  keys.sort((a,b)=>{
    const av = Number(temps[a]); const bv = Number(temps[b]);
    if(!isFinite(av) && !isFinite(bv)) return 0;
    if(!isFinite(av)) return 1;
    if(!isFinite(bv)) return -1;
    return bv - av;
  });

  // temperature-to-bar mapping range (visual only)
  const T_MIN = 20;
  const T_MAX = 90;

  keys.slice(0, 12).forEach(k=>{
    const v = Number(temps[k]);
    if(!isFinite(v)) return;

    const row = document.createElement("div");
    row.className = "tempRow";

    // color bucket
    if(v >= 70) row.classList.add("bad");
    else if(v >= 55) row.classList.add("warn");
    else row.classList.add("good");

    const top = document.createElement("div");
    top.className = "tempTop";

    const name = document.createElement("div");
    name.className = "tempName";
    name.title = k;
    name.textContent = k;

    const val = document.createElement("div");
    val.className = "tempVal";
    val.textContent = v.toFixed(1) + "°C";

    top.appendChild(name);
    top.appendChild(val);

    const bar = document.createElement("div");
    bar.className = "tempBar";

    const fill = document.createElement("div");
    fill.className = "tempFill";
    const pct = Math.max(0, Math.min(1, (v - T_MIN) / (T_MAX - T_MIN))) * 100;
    fill.style.width = pct.toFixed(1) + "%";
    bar.appendChild(fill);

    row.appendChild(top);
    row.appendChild(bar);

    box.appendChild(row);
  });
}

async function poll(){
  try{
    const res = await fetch("/api/status", {cache:"no-store"});
    const s = await res.json();

    document.getElementById("ageRemote").textContent = (s.remote.age_s ?? "-");
    document.getElementById("agePad").textContent = (s.gamepad.age_s ?? "-");

    document.getElementById("fpsVideo").textContent = s.video.fps_est || "-";
    document.getElementById("fpsDepth").textContent = s.video.fps_depth_est || "-";
    document.getElementById("videoText").textContent = (s.video.src||"-") + " / " + (s.video.ok ? "OK" : "NG");
    setConn(s.active);

    document.getElementById("activeSrc").textContent = s.active;
    document.getElementById("lastL").textContent = s.last_sent.L;
    document.getElementById("lastR").textContent = s.last_sent.R;

    setRadarFromLR(s.last_sent.L, s.last_sent.R);

    document.getElementById("rgbAge").textContent = s.video.rgb_age_ms ?? "-";

    const j = s.jetson || {};
    document.getElementById("jetsonText").textContent = (j.ok ? "OK" : "NG") + " / " + (j.src || "-");
    document.getElementById("teleSrc").textContent = j.src || "-";
    document.getElementById("teleTs").textContent = j.ts ? new Date(j.ts*1000).toLocaleTimeString() : "-";

    const d = j.data || {};
    const cpu = d.cpu_util_pct ?? 0;
    const gpu = d.gpu_util_pct ?? 0;
    const emc = d.emc_util_pct ?? 0;

    const ramU = d.ram_used_mb ?? 0;
    const ramT = d.ram_total_mb ?? 0;
    const ramP = d.ram_util_pct ?? (ramT? (100*ramU/ramT) : 0);

    const swU = d.swap_used_mb ?? 0;
    const swT = d.swap_total_mb ?? 0;

    const diskU = d.disk_used_gb ?? 0;
    const diskT = d.disk_total_gb ?? 0;
    const diskF = d.disk_free_gb ?? 0;
    const diskP = d.disk_util_pct ?? (diskT? (100*diskU/diskT) : 0);

    document.getElementById("cpuUtil").textContent = cpu;
    document.getElementById("gpuUtil").textContent = gpu;
    document.getElementById("emcUtil").textContent = emc;

    document.getElementById("ramUsed").textContent = ramU;
    document.getElementById("ramTotal").textContent = ramT;
    document.getElementById("swapUsed").textContent = swU;
    document.getElementById("swapTotal").textContent = swT;

    document.getElementById("diskUsed").textContent = diskU;
    document.getElementById("diskTotal").textContent = diskT;
    document.getElementById("diskFree").textContent = diskF;

    document.getElementById("cpuFreq").textContent = d.cpu_freq_avg_mhz ?? "-";

    setGauge("gCpu", cpu);
    setGauge("gGpu", gpu);
    setGauge("gRam", ramP);
    setGauge("gDisk", diskP);

    document.getElementById("uptime").textContent = d.uptime_h || "-";
    document.getElementById("loadavg").textContent = d.loadavg ? d.loadavg.join(", ") : "-";

    const hotName = d.temp_hot_name ?? "-";
    const hotVal = (d.temp_hot_c ?? d.temp_max_c);
    document.getElementById("hotSpot").textContent = (hotVal != null) ? `${hotName} ${hotVal}°C` : "-";

    document.getElementById("netDown").textContent = (d.net_down_mbps ?? "-");
    document.getElementById("netUp").textContent = (d.net_up_mbps ?? "-");

    renderTemps(d.temps_c);

  }catch(e){const now=Date.now(); if(now-_lastPollErr>2000){_lastPollErr=now; try{log('poll error: '+e);}catch(_){}}}
}

setInterval(poll, 300);
updateDot();
log("Console ready");
function sendStopBeacon(){
  try{
    const payload = JSON.stringify({turn:0, throttle:0, sid:SID, seq:cmdSeq+1, t:Date.now()});
    if(navigator.sendBeacon){
      navigator.sendBeacon("/api/cmd", new Blob([payload], {type:"application/json"}));
    }else{
      fetch("/api/cmd",{method:"POST",headers:{"Content-Type":"application/json"},body:payload,keepalive:true}).catch(()=>{});
    }
  }catch(e){}
}

window.addEventListener("pagehide", ()=>{ sendStopBeacon(); });
document.addEventListener("visibilitychange", ()=>{ if(document.hidden) sendStopBeacon(); });

</script>
</body>
</html>
"""

INDEX_HTML = (INDEX_HTML
    .replace("__RGB_W__", str(RGB_W))
    .replace("__RGB_H__", str(RGB_H))
    .replace("__VIDEO_FPS__", str(VIDEO_FPS))
    .replace("__DEPTH_FPS__", str(DEPTH_FPS))
    .replace("__DEPTH_MIN_M__", str(DEPTH_MIN_M))
    .replace("__DEPTH_MAX_M__", str(DEPTH_MAX_M))
)

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}

@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html", headers=NO_CACHE_HEADERS)

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

if __name__ == "__main__":
    main()
