import time, serial
ser = serial.Serial("COM8", 115200, timeout=0.1)
time.sleep(2)

for _ in range(200):
    ser.write(b"60,60\n")   # 前進
    time.sleep(0.02)

ser.write(b"0,0\n")         # 停止
ser.close()
print("done")
