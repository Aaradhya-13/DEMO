/*
 * firmware/esp32_lora_node.ino
 * -----------------------------
 * ESP32 + SX1276 LoRa transceiver firmware for the flood rescue mesh.
 *
 * Responsibilities:
 *   - Receive the fixed-size binary distress packet over the air.
 *   - Validate CRC-16 in firmware (mirrors packet/binary_protocol.py).
 *   - Echo every valid received packet to Serial USB as "RX:<hex>\n"
 *     so mesh_radio/lora_gateway.py can pick it up.
 *   - Listen on Serial USB for "TX:<hex>\n" lines and broadcast that
 *     packet over the air (uplink from the gateway).
 *
 * All radio parameters (frequency, bandwidth, spreading factor, coding
 * rate, sync word) are read from compile-time config macros below so a
 * single firmware source supports every regional/regulatory profile —
 * change the macros, not the logic.
 *
 * Library: RadioLib (https://github.com/jgromes/RadioLib)
 */

#include <RadioLib.h>

// ---------------------------------------------------------------------
// Radio configuration (override via build flags / config macros; not
// fixed business logic, just the physical-layer profile for this node).
// ---------------------------------------------------------------------
#ifndef LORA_FREQUENCY_MHZ
#define LORA_FREQUENCY_MHZ 868.0     // 868 MHz (EU) or 433.0 (India/Asia) — set per deployment region
#endif
#ifndef LORA_BANDWIDTH_KHZ
#define LORA_BANDWIDTH_KHZ 125.0
#endif
#ifndef LORA_SPREADING_FACTOR
#define LORA_SPREADING_FACTOR 11     // SF11/SF12 recommended for long-range flood-zone links
#endif
#ifndef LORA_CODING_RATE
#define LORA_CODING_RATE 8           // 4/8 coding rate for maximum robustness
#endif
#ifndef LORA_SYNC_WORD
#define LORA_SYNC_WORD 0x34          // Private network sync word
#endif
#ifndef LORA_TX_POWER_DBM
#define LORA_TX_POWER_DBM 20
#endif

// ---------------------------------------------------------------------
// SX1276 pin mapping (adjust to match the specific ESP32 dev board /
// LoRa shield wiring being used).
// ---------------------------------------------------------------------
#ifndef PIN_LORA_NSS
#define PIN_LORA_NSS 18
#endif
#ifndef PIN_LORA_DIO0
#define PIN_LORA_DIO0 26
#endif
#ifndef PIN_LORA_RESET
#define PIN_LORA_RESET 14
#endif
#ifndef PIN_LORA_DIO1
#define PIN_LORA_DIO1 33
#endif

#define SERIAL_BAUD 115200
#define PACKET_SIZE_BYTES 27   // matches packet/binary_protocol.py _TOTAL_SIZE
#define SERIAL_LINE_MAX_LEN 128

SX1276 radio = new Module(PIN_LORA_NSS, PIN_LORA_DIO0, PIN_LORA_RESET, PIN_LORA_DIO1);

volatile bool packetReceivedFlag = false;

void ICACHE_RAM_ATTR onPacketReceived() {
  packetReceivedFlag = true;
}

// ---------------------------------------------------------------------
// CRC-16/ANSI — must exactly mirror crcmod's "crc-16" predefined function
// used in packet/binary_protocol.py so both ends agree on validity.
// Polynomial 0x8005, init 0x0000, reflected in/out, no xorout.
// ---------------------------------------------------------------------
uint16_t crc16Ansi(const uint8_t *data, size_t length) {
  uint16_t crc = 0x0000;
  for (size_t i = 0; i < length; i++) {
    crc ^= (uint16_t)data[i];
    for (uint8_t bit = 0; bit < 8; bit++) {
      if (crc & 0x0001) {
        crc = (crc >> 1) ^ 0xA001; // reflected 0x8005
      } else {
        crc >>= 1;
      }
    }
  }
  return crc;
}

void logStatus(const char *label, int state) {
  Serial.print("STATUS:");
  Serial.print(label);
  Serial.print("=");
  Serial.println(state);
}

void setupRadio() {
  int state = radio.begin(
      LORA_FREQUENCY_MHZ,
      LORA_BANDWIDTH_KHZ,
      LORA_SPREADING_FACTOR,
      LORA_CODING_RATE,
      LORA_SYNC_WORD,
      LORA_TX_POWER_DBM
  );

  if (state != RADIOLIB_ERR_NONE) {
    logStatus("radio_init_failed", state);
    while (true) {
      delay(1000); // halt: a misconfigured radio must not silently run
    }
  }
  logStatus("radio_init_ok", state);

  radio.setDio0Action(onPacketReceived, RISING);

  state = radio.startReceive();
  if (state != RADIOLIB_ERR_NONE) {
    logStatus("start_receive_failed", state);
  }
}

void hexEncode(const uint8_t *data, size_t length, char *outBuffer) {
  static const char hexChars[] = "0123456789abcdef";
  for (size_t i = 0; i < length; i++) {
    outBuffer[i * 2]     = hexChars[(data[i] >> 4) & 0x0F];
    outBuffer[i * 2 + 1] = hexChars[data[i] & 0x0F];
  }
  outBuffer[length * 2] = '\0';
}

bool hexDecode(const char *hexString, uint8_t *outBuffer, size_t maxOutLen, size_t *decodedLen) {
  size_t hexLen = strlen(hexString);
  if (hexLen % 2 != 0 || (hexLen / 2) > maxOutLen) {
    return false;
  }
  for (size_t i = 0; i < hexLen; i += 2) {
    char highNibbleChar = hexString[i];
    char lowNibbleChar = hexString[i + 1];
    uint8_t highNibble, lowNibble;

    if (highNibbleChar >= '0' && highNibbleChar <= '9') highNibble = highNibbleChar - '0';
    else if (highNibbleChar >= 'a' && highNibbleChar <= 'f') highNibble = highNibbleChar - 'a' + 10;
    else if (highNibbleChar >= 'A' && highNibbleChar <= 'F') highNibble = highNibbleChar - 'A' + 10;
    else return false;

    if (lowNibbleChar >= '0' && lowNibbleChar <= '9') lowNibble = lowNibbleChar - '0';
    else if (lowNibbleChar >= 'a' && lowNibbleChar <= 'f') lowNibble = lowNibbleChar - 'a' + 10;
    else if (lowNibbleChar >= 'A' && lowNibbleChar <= 'F') lowNibble = lowNibbleChar - 'A' + 10;
    else return false;

    outBuffer[i / 2] = (highNibble << 4) | lowNibble;
  }
  *decodedLen = hexLen / 2;
  return true;
}

void handleReceivedRadioPacket() {
  uint8_t buffer[PACKET_SIZE_BYTES];
  int state = radio.readData(buffer, PACKET_SIZE_BYTES);

  if (state != RADIOLIB_ERR_NONE) {
    logStatus("read_data_failed", state);
    radio.startReceive();
    return;
  }

  // Validate CRC-16 over the first (PACKET_SIZE_BYTES - 2) bytes against
  // the trailing big-endian uint16 CRC field, mirroring the Python side.
  uint16_t computedCrc = crc16Ansi(buffer, PACKET_SIZE_BYTES - 2);
  uint16_t receivedCrc = ((uint16_t)buffer[PACKET_SIZE_BYTES - 2] << 8) | buffer[PACKET_SIZE_BYTES - 1];

  if (computedCrc != receivedCrc) {
    Serial.print("STATUS:crc_mismatch computed=0x");
    Serial.print(computedCrc, HEX);
    Serial.print(" received=0x");
    Serial.println(receivedCrc, HEX);
    radio.startReceive();
    return;
  }

  char hexBuffer[PACKET_SIZE_BYTES * 2 + 1];
  hexEncode(buffer, PACKET_SIZE_BYTES, hexBuffer);

  Serial.print("RX:");
  Serial.println(hexBuffer);

  radio.startReceive();
}

void handleSerialUplink() {
  static char lineBuffer[SERIAL_LINE_MAX_LEN];
  static size_t lineLength = 0;

  while (Serial.available() > 0) {
    char incomingChar = Serial.read();

    if (incomingChar == '\n') {
      lineBuffer[lineLength] = '\0';

      if (strncmp(lineBuffer, "TX:", 3) == 0) {
        uint8_t packetBuffer[PACKET_SIZE_BYTES];
        size_t decodedLen = 0;

        if (hexDecode(lineBuffer + 3, packetBuffer, PACKET_SIZE_BYTES, &decodedLen)
            && decodedLen == PACKET_SIZE_BYTES) {
          int state = radio.transmit(packetBuffer, PACKET_SIZE_BYTES);
          logStatus("tx_result", state);
          radio.startReceive(); // return to RX mode after transmitting
        } else {
          Serial.println("STATUS:tx_decode_failed");
        }
      }

      lineLength = 0;
    } else if (lineLength < SERIAL_LINE_MAX_LEN - 1) {
      lineBuffer[lineLength++] = incomingChar;
    }
    // else: silently drop overflow characters until next newline
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) {
    delay(10);
  }
  Serial.println("STATUS:boot");
  setupRadio();
}

void loop() {
  if (packetReceivedFlag) {
    packetReceivedFlag = false;
    handleReceivedRadioPacket();
  }
  handleSerialUplink();
}
