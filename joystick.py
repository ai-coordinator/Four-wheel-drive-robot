import time
import serial
import pygame

# SERIAL_PORT = "COM8"
SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200

DEADZONE = 0.18
MAX_OUT = 127
SEND_HZ = 50

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

def main():
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.01)
    time.sleep(2.0)  # UNOリセット待ち

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise RuntimeError("ゲームパッドが見つかりません。")

    js = pygame.joystick.Joystick(0)
    js.init()
    print("Gamepad:", js.get_name())
    print("Sending to:", SERIAL_PORT)

    dt = 1.0 / SEND_HZ
    last_send = 0.0

    try:
        while True:
            pygame.event.pump()

            # 左スティック：X=axis0, Y=axis1 が多い（上が-1, 下が+1）
            turn_raw = -js.get_axis(0)       # 左右
            throttle_raw = -js.get_axis(1)  # 前後（前を+にするため反転）

            throttle = apply_deadzone(throttle_raw)
            turn = apply_deadzone(turn_raw)

            # 差動ミキシング（左スティック1本で前後左右）
            left = throttle + turn
            right = throttle - turn

            # |1|超えを正規化（斜め入力でも飽和せず扱いやすい）
            m = max(1.0, abs(left), abs(right))
            left /= m
            right /= m

            L = to_int127(left)
            R = to_int127(right)

            now = time.time()
            if now - last_send >= dt:
                ser.write(f"{L},{R}\n".encode("ascii"))
                last_send = now

            if js.get_button(0):  # Aボタン
                L, R = 0, 0

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            ser.write(b"0,0\n")
            time.sleep(0.05)
        except Exception:
            pass
        ser.close()
        pygame.quit()
        print("Stopped.")

if __name__ == "__main__":
    main()
