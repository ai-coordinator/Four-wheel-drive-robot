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

const toggleQR = document.getElementById("toggleQR");
const qrPip = document.getElementById("qrPip");
const qrImg = document.getElementById("qrImg");

function updateQR(){
  if(!toggleQR) return;
  if(toggleQR.checked){
    qrPip.style.display = "block";
    // Cache-bust so it refreshes if host/port changes
    qrImg.src = "/qr.png?ts=" + Date.now();
  }else{
    qrPip.style.display = "none";
    qrImg.removeAttribute("src");
  }
}

if(toggleQR){
  toggleQR.addEventListener("change", updateQR);
  updateQR();
}



// ============================
// Kiosk / Animal Face Mode (stable rewrite)
// - Kiosk is "animal face only" (no video in kiosk)
// - Open via topbar 🐾Kiosk button
// - In kiosk:
//     * Tap face -> back to normal UI (also exits fullscreen if active)
//     * Long-press (>=600ms) -> toggle fullscreen
//     * ESC -> back to normal UI
// ============================
const kioskEl = document.getElementById("kiosk");
const kioskFace = kioskEl ? kioskEl.querySelector(".kioskFace") : null;

function qsBool(name){
  const v = new URLSearchParams(location.search).get(name);
  return v === "1" || v === "true" || v === "yes";
}

let kioskMode = false;

function setKiosk(on){
  kioskMode = !!on;
  if(!kioskEl) return;

  // Hide/show the normal UI to prevent any "double" visuals
  document.body.classList.toggle("kiosk-on", kioskMode);

  kioskEl.style.display = kioskMode ? "flex" : "none";
  kioskEl.setAttribute("aria-hidden", kioskMode ? "false" : "true");
}

async function toggleFullscreen(){
  try{
    if(!document.fullscreenElement){
      await document.documentElement.requestFullscreen();
    }else{
      await document.exitFullscreen();
    }
  }catch(e){
    // ignore browser restrictions
  }
}

async function exitFullscreenIfAny(){
  if(document.fullscreenElement){
    try{ await document.exitFullscreen(); }catch(e){}
  }
}

async function backToNormal(){
  await exitFullscreenIfAny();
  setKiosk(false);
}

function openKiosk(fullscreen){
  setKiosk(true);
  // Fullscreen is only reliable when called from a user gesture (e.g., button tap)
  if(fullscreen){ toggleFullscreen(); }
}

if(kioskEl){
  // Allow direct link open: ?kiosk=1
  if(qsBool("kiosk")){
    openKiosk(false); // URL open: no auto fullscreen (browser restriction)
  }

  // Background tap closes kiosk (tap outside face)
  kioskEl.addEventListener("click", (ev)=>{
    if(ev.target === kioskEl){
      backToNormal();
    }
  });

  document.addEventListener("keydown", (ev)=>{
    if(ev.key === "Escape" && kioskMode){
      backToNormal();
    }
  });
}

// Face gestures
(function(){
  if(!kioskFace) return;

  // Tap / long-press handling (pointer events)
  let timer = null;
  let longFired = false;

  function clear(){
    if(timer){ clearTimeout(timer); timer = null; }
    longFired = false;
  }

  kioskFace.addEventListener("pointerdown", (ev)=>{
    ev.stopPropagation();
    clear();
    timer = setTimeout(()=>{
      longFired = true;
      toggleFullscreen();
    }, 600);
  });

  kioskFace.addEventListener("pointerup", (ev)=>{
    ev.stopPropagation();
    if(timer){ clearTimeout(timer); timer = null; }
    if(!longFired){
      backToNormal();
    }
    clear();
  });

  kioskFace.addEventListener("pointercancel", clear);

  // Fallback for environments without pointer events
  kioskFace.addEventListener("click", (ev)=>{
    ev.stopPropagation();
    backToNormal();
  });
})();

// Kiosk button on normal UI
const btnKiosk = document.getElementById("btnKiosk");
if(btnKiosk){
  btnKiosk.addEventListener("click", (ev)=>{
    ev.stopPropagation();
    openKiosk(true);
  });
}

