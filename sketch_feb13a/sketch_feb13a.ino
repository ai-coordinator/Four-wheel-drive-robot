#include <SoftwareSerial.h>

// SoftwareSerial(RX, TX)
// RXは使わないので適当なピンでOK（D10など）
// TXはD11（あなたがMDDS10へ繋いだピン）
SoftwareSerial mdds(10, 11);

const uint8_t ADDR = 0;           // DIPのアドレス（0-7）
const uint32_t PC_BAUD = 115200;  // PC <-> UNO (USB)
const uint32_t MDDS_BAUD = 57600; // UNO -> MDDS10（あなたの環境で動いた値）
const uint16_t FAILSAFE_MS = 200; // 途切れたら停止

uint8_t toCmd(int speed) {        // -127..+127 (0 stop)
  if (speed > 127) speed = 127;
  if (speed < -127) speed = -127;
  return (uint8_t)(127 + speed);  // 127=stop
}

// motor: 0=Left, 1=Right
void sendPacket(uint8_t motor, int speed) {
  uint8_t h = 0x55;
  uint8_t ch_addr = ((motor & 0x01) << 3) | (ADDR & 0x07);
  uint8_t cmd = toCmd(speed);
  uint8_t sum = (uint8_t)(h + ch_addr + cmd);

  mdds.write(h);
  mdds.write(ch_addr);
  mdds.write(cmd);
  mdds.write(sum);
}

void stopBoth() {
  sendPacket(0, 0);
  sendPacket(1, 0);
}

// ---- PCから "L,R\n" を受ける ----
String lineBuf;
uint32_t lastCmdMs = 0;

bool parseLine(const String& s, int &L, int &R) {
  int comma = s.indexOf(',');
  if (comma < 0) return false;
  String a = s.substring(0, comma);
  String b = s.substring(comma + 1);
  a.trim(); b.trim();
  L = a.toInt();
  R = b.toInt();
  if (L > 127) L = 127; if (L < -127) L = -127;
  if (R > 127) R = 127; if (R < -127) R = -127;
  return true;
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.begin(PC_BAUD);   // PCと会話
  mdds.begin(MDDS_BAUD);   // MDDS10へ送信

  delay(1000);             // MDDS10起動待ち（重要）
  stopBoth();
  lastCmdMs = millis();

  Serial.println("READY: send \"L,R\\n\"  (-127..127)");
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      int L, R;
      if (parseLine(lineBuf, L, R)) {
        sendPacket(0, L);
        sendPacket(1, R);
        lastCmdMs = millis();
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); // 受信のたび点滅
      }
      lineBuf = "";
    } else if (c != '\r') {
      if (lineBuf.length() < 32) lineBuf += c;
      else lineBuf = "";
    }
  }

  // 途切れたら停止
  if (millis() - lastCmdMs > FAILSAFE_MS) {
    stopBoth();
    lastCmdMs = millis();
  }
}
