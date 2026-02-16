# -*- coding: utf-8 -*-
import time
import threading
import serial
import pygame
from flask import Flask, request, jsonify, Response

# ========= Serial (UNO bridge) =========
SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200
SEND_HZ = 50  # ★Arduino failsafe対策：同値でも周期送信する

# ========= Gamepad =========
DEADZONE = 0.18
MAX_OUT = 127

# ========= Arbitration / Safety =========
REMOTE_TIMEOUT = 1.0     # リモート更新がこれ以上止まったら無効
GAMEPAD_TIMEOUT = 0.25   # ゲームパッド更新が止まったら無効
FAILSAFE_STOP = 1.2      # どちらも止まったら停止

lock = threading.Lock()
state = {
    "remote_L": 0, "remote_R": 0, "remote_ts": 0.0,
    "pad_L": 0, "pad_R": 0, "pad_ts": 0.0,
    "active": "none",
    "last_sent_L": 0, "last_sent_R": 0,
    "last_write_ts": 0.0,
}

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
    return x * abs(x)  # 2乗カーブ（低速が扱いやすい）

def mix_to_lr(throttle, turn):
    left = throttle + turn
    right = throttle - turn
    m = max(1.0, abs(left), abs(right))
    left /= m
    right /= m
    return to_int127(left), to_int127(right)

# ---------------------------
# Gamepad thread
# ---------------------------
def gamepad_loop():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("[GAMEPAD] not found (remote only).")
        return

    js = pygame.joystick.Joystick(0)
    js.init()
    print("[GAMEPAD] found:", js.get_name())

    while True:
        pygame.event.pump()

        turn_raw = -js.get_axis(0)
        throttle_raw = -js.get_axis(1)

        throttle = curve(apply_deadzone(throttle_raw))
        turn = curve(apply_deadzone(turn_raw))

        L, R = mix_to_lr(throttle, turn)

        if js.get_button(0):  # Aで即停止
            L, R = 0, 0

        now = time.time()
        with lock:
            state["pad_L"] = L
            state["pad_R"] = R
            state["pad_ts"] = now

        time.sleep(0.005)

# ---------------------------
# Serial sender loop (single writer)
# ★ここが修正点：同じ値でも SEND_HZ で必ず送り続ける
# ---------------------------
def sender_loop():
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.01)
    time.sleep(2.0)  # UNO reset wait

    dt = 1.0 / SEND_HZ

    while True:
        now = time.time()
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

        # ★必ず送る（Arduino側 failsafe を満たす）
        ser.write(f"{L},{R}\n".encode("ascii"))
        time.sleep(dt)

# ---------------------------
# Flask server
# ---------------------------
app = Flask(__name__)

INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot Remote</title>
<style>
body { font-family: sans-serif; margin: 16px; }
.row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
#pad { width: 280px; height: 280px; border: 2px solid #333; border-radius: 12px;
       touch-action: none; display:flex; align-items:center; justify-content:center; user-select:none; }
#dot { width: 18px; height: 18px; border-radius: 50%; background:#333; position: relative; }
small { color:#555; }
button { padding: 10px 14px; font-size: 16px; }
pre { background:#f5f5f5; padding: 10px; border-radius: 8px; }
</style>
</head>
<body>
<h2>Robot Remote</h2>
<div class="row">
  <div id="pad"><div id="dot"></div></div>
  <div>
    <button onclick="stop()">STOP</button><br><br>
    <small>
      Touch: drag inside pad (hold)<br>
      Keyboard: W/A/S/D (hold)<br>
      Remote has priority while active
    </small>
    <pre id="status">...</pre>
  </div>
</div>

<script>
const pad = document.getElementById("pad");
const dot = document.getElementById("dot");
const st = document.getElementById("status");

let x=0, y=0;            // -1..1 (x=turn, y=throttle)
let keys = {w:false,a:false,s:false,d:false};
let dragging=false;

function clamp(v,min,max){ return Math.max(min, Math.min(max,v)); }

function updateDot(){
  const size = pad.getBoundingClientRect();
  const cx = size.width/2, cy = size.height/2;
  const r = Math.min(cx, cy) - 20;
  dot.style.left = (x*r) + "px";
  dot.style.top  = (-y*r) + "px";
}

function send(){
  fetch("/api/cmd", {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({turn:x, throttle:y})
  }).catch(()=>{});
}

function stop(){
  x=0; y=0;
  updateDot();
  send();
}

function pointerToXY(ev){
  const rect = pad.getBoundingClientRect();
  const cx = rect.left + rect.width/2;
  const cy = rect.top + rect.height/2;
  const rx = (ev.clientX - cx) / (rect.width/2);
  const ry = (ev.clientY - cy) / (rect.height/2);
  x = clamp(rx, -1, 1);
  y = clamp(-ry, -1, 1);
  updateDot();
}

pad.addEventListener("pointerdown", (ev)=>{
  dragging=true;
  pad.setPointerCapture(ev.pointerId);
  pointerToXY(ev);
  send();
});

pad.addEventListener("pointermove", (ev)=>{
  if(!dragging) return;
  pointerToXY(ev);
  send();
});

pad.addEventListener("pointerup", ()=>{
  dragging=false;
  stop();
});
pad.addEventListener("pointercancel", ()=>{
  dragging=false;
  stop();
});

window.addEventListener("keydown",(e)=>{
  if(e.repeat) return;
  if(e.key==="w") keys.w=true;
  if(e.key==="a") keys.a=true;
  if(e.key==="s") keys.s=true;
  if(e.key==="d") keys.d=true;
});
window.addEventListener("keyup",(e)=>{
  if(e.key==="w") keys.w=false;
  if(e.key==="a") keys.a=false;
  if(e.key==="s") keys.s=false;
  if(e.key==="d") keys.d=false;
});

// キーボードは50msで更新
setInterval(()=>{
  let t = 0, r = 0;
  if(keys.w) t += 1;
  if(keys.s) t -= 1;
  if(keys.d) r += 1;
  if(keys.a) r -= 1;

  if(t!==0 || r!==0){
    x = 0.7*r;
    y = 0.7*t;
    updateDot();
    send();
  }
}, 50);

// ★操作中は keepalive（同じ位置でもremote_ts更新）
setInterval(()=>{
  const active = dragging || keys.w || keys.a || keys.s || keys.d;
  if(active){
    send();
  }
}, 80);

setInterval(async ()=>{
  try{
    const res = await fetch("/api/status");
    const j = await res.json();
    st.textContent = JSON.stringify(j, null, 2);
  }catch(e){
    st.textContent = "status error";
  }
}, 300);

updateDot();
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")

@app.get("/api/status")
def status():
    with lock:
        return jsonify({
            "active": state["active"],
            "remote": {"L": state["remote_L"], "R": state["remote_R"], "age_s": round(time.time()-state["remote_ts"], 3)},
            "gamepad": {"L": state["pad_L"], "R": state["pad_R"], "age_s": round(time.time()-state["pad_ts"], 3)},
            "last_sent": {"L": state["last_sent_L"], "R": state["last_sent_R"]},
            "last_write_age_s": round(time.time()-state["last_write_ts"], 3),
        })

@app.post("/api/cmd")
def cmd():
    data = request.get_json(force=True, silent=True) or {}
    turn = float(data.get("turn", 0.0))
    throttle = float(data.get("throttle", 0.0))

    turn = -turn  # ★ここ追加
    
    turn = max(-1.0, min(1.0, turn))
    throttle = max(-1.0, min(1.0, throttle))

    throttle = curve(apply_deadzone(throttle))
    turn = curve(apply_deadzone(turn))

    L, R = mix_to_lr(throttle, turn)

    with lock:
        state["remote_L"] = L
        state["remote_R"] = R
        state["remote_ts"] = time.time()

    return jsonify({"ok": True, "L": L, "R": R})

@app.post("/api/stop")
def api_stop():
    with lock:
        state["remote_L"] = 0
        state["remote_R"] = 0
        state["remote_ts"] = time.time()
    return jsonify({"ok": True})

def main():
    threading.Thread(target=sender_loop, daemon=True).start()
    threading.Thread(target=gamepad_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()
