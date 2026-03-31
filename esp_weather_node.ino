/*
 * ================================================================
 *  NimbusCast IoT Weather Node — ESP8266 / ESP32
 *  Sensors: DHT22 (Temp/Humidity) + BMP280 (Pressure/Altitude)
 *           + MQ-135 (Air Quality) + LDR (Light/UV proxy)
 *
 *  Data Flow:
 *    ESP Module → MQTT (AWS IoT Core) → Lambda → S3 Bucket
 *                                              → DynamoDB (hot cache)
 *
 *  Board: NodeMCU ESP8266 or ESP32 DevKit
 *  IDE:   Arduino IDE 2.x
 *  Libs:  PubSubClient, ArduinoJson, DHT, Adafruit_BMP280,
 *         WiFiClientSecure, NTPClient
 * ================================================================
 */

#include <Arduino.h>

#ifdef ESP32
  #include <WiFi.h>
  #include <WiFiClientSecure.h>
#else
  #include <ESP8266WiFi.h>
  #include <WiFiClientSecure.h>
#endif

#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <Wire.h>
#include <Adafruit_BMP280.h>
#include <NTPClient.h>
#include <WiFiUDP.h>

// ── WiFi Credentials ─────────────────────────────────────────
#define WIFI_SSID     "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"

// ── AWS IoT Core ──────────────────────────────────────────────
#define AWS_IOT_ENDPOINT   "YOUR_ENDPOINT.iot.ap-south-1.amazonaws.com"
#define AWS_IOT_PORT       8883
#define THING_NAME         "nimbus-node-001"
#define MQTT_TOPIC_PUB     "nimbus/sensors/data"
#define MQTT_TOPIC_CMD     "nimbus/sensors/command"
#define MQTT_TOPIC_STATUS  "nimbus/sensors/status"

// ── TLS Certificates (paste from AWS IoT Console) ─────────────
// Download from: AWS Console > IoT Core > Security > Certificates
static const char AWS_ROOT_CA[] PROGMEM = R"EOF(
-----BEGIN CERTIFICATE-----
[PASTE YOUR AWS ROOT CA CERTIFICATE HERE]
-----END CERTIFICATE-----
)EOF";

static const char DEVICE_CERT[] PROGMEM = R"EOF(
-----BEGIN CERTIFICATE-----
[PASTE YOUR DEVICE CERTIFICATE HERE]
-----END CERTIFICATE-----
)EOF";

static const char PRIVATE_KEY[] PROGMEM = R"EOF(
-----BEGIN RSA PRIVATE KEY-----
[PASTE YOUR PRIVATE KEY HERE]
-----END RSA PRIVATE KEY-----
)EOF";

// ── Pin Definitions ───────────────────────────────────────────
#define DHT_PIN        D4     // GPIO2  — DHT22 data pin
#define DHT_TYPE       DHT22
#define SDA_PIN        D2     // GPIO4  — I2C SDA (BMP280)
#define SCL_PIN        D1     // GPIO5  — I2C SCL (BMP280)
#define MQ135_PIN      A0     // Analog — Air quality sensor
#define LDR_PIN        D7     // GPIO13 — Light sensor (digital)
#define STATUS_LED     D0     // Onboard LED

// ── Timing ────────────────────────────────────────────────────
#define PUBLISH_INTERVAL_MS  30000   // Send data every 30 seconds
#define RECONNECT_DELAY_MS   5000
#define WATCHDOG_TIMEOUT_S   60

// ── Sensor Objects ────────────────────────────────────────────
DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_BMP280 bmp;
WiFiClientSecure wifiClient;
PubSubClient mqttClient(wifiClient);
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "pool.ntp.org", 19800, 60000); // IST UTC+5:30

// ── State ─────────────────────────────────────────────────────
unsigned long lastPublish = 0;
unsigned long lastReconnect = 0;
uint32_t publishCount = 0;
bool bmpOk = false;

// ── Sensor Reading Struct ─────────────────────────────────────
struct SensorData {
  float temperature;      // °C
  float humidity;         // %
  float pressure;         // hPa
  float altitude;         // meters
  int   airQualityRaw;    // ADC 0-1023
  float airQualityPPM;    // estimated PPM
  bool  lightDetected;    // LDR
  float heatIndex;        // calculated
  float dewPoint;         // calculated
  float absoluteHumidity; // g/m³
  bool  valid;
};

// ── Setup ─────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println(F("\n[NimbusCast] ESP Weather Node Starting..."));

  pinMode(STATUS_LED, OUTPUT);
  digitalWrite(STATUS_LED, LOW);

  Wire.begin(SDA_PIN, SCL_PIN);
  dht.begin();

  bmpOk = bmp.begin(0x76);
  if (!bmpOk) {
    Serial.println(F("[WARN] BMP280 not found at 0x76, trying 0x77"));
    bmpOk = bmp.begin(0x77);
  }
  if (bmpOk) {
    bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,
                    Adafruit_BMP280::SAMPLING_X2,
                    Adafruit_BMP280::SAMPLING_X16,
                    Adafruit_BMP280::FILTER_X16,
                    Adafruit_BMP280::STANDBY_MS_500);
    Serial.println(F("[OK] BMP280 initialized"));
  }

  connectWiFi();

  wifiClient.setCACert(AWS_ROOT_CA);
  wifiClient.setCertificate(DEVICE_CERT);
  wifiClient.setPrivateKey(PRIVATE_KEY);

  mqttClient.setServer(AWS_IOT_ENDPOINT, AWS_IOT_PORT);
  mqttClient.setCallback(onMQTTMessage);
  mqttClient.setBufferSize(512);

  timeClient.begin();
  timeClient.update();

  connectMQTT();
  publishStatus("online");

  Serial.println(F("[NimbusCast] Ready. Publishing every 30s..."));
  blinkLED(3, 200);
}

// ── Main Loop ─────────────────────────────────────────────────
void loop() {
  if (!mqttClient.connected()) {
    unsigned long now = millis();
    if (now - lastReconnect > RECONNECT_DELAY_MS) {
      lastReconnect = now;
      if (!WiFi.isConnected()) connectWiFi();
      connectMQTT();
    }
  }
  mqttClient.loop();
  timeClient.update();

  unsigned long now = millis();
  if (now - lastPublish >= PUBLISH_INTERVAL_MS) {
    lastPublish = now;
    SensorData data = readAllSensors();
    if (data.valid) {
      publishSensorData(data);
      blinkLED(1, 100);
    } else {
      Serial.println(F("[WARN] Sensor read failed, skipping publish"));
    }
  }
}

// ── WiFi Connection ───────────────────────────────────────────
void connectWiFi() {
  Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500); Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println(F("\n[WiFi] Failed! Restarting..."));
    delay(3000); ESP.restart();
  }
}

// ── MQTT Connection ───────────────────────────────────────────
void connectMQTT() {
  Serial.printf("[MQTT] Connecting to %s...\n", AWS_IOT_ENDPOINT);
  if (mqttClient.connect(THING_NAME)) {
    Serial.println(F("[MQTT] Connected to AWS IoT Core!"));
    mqttClient.subscribe(MQTT_TOPIC_CMD);
    Serial.printf("[MQTT] Subscribed to %s\n", MQTT_TOPIC_CMD);
  } else {
    Serial.printf("[MQTT] Failed, rc=%d\n", mqttClient.state());
  }
}

// ── Read All Sensors ──────────────────────────────────────────
SensorData readAllSensors() {
  SensorData d;
  d.valid = false;

  // DHT22: Temperature & Humidity
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  if (isnan(h) || isnan(t)) {
    Serial.println(F("[DHT22] Read failed"));
    return d;
  }
  d.temperature = t;
  d.humidity    = h;

  // BMP280: Pressure & Altitude
  if (bmpOk) {
    d.pressure = bmp.readPressure() / 100.0F; // Pa to hPa
    d.altitude = bmp.readAltitude(1013.25);
  } else {
    d.pressure = 1013.25;
    d.altitude = 0.0;
  }

  // MQ-135: Air Quality (raw + estimated PPM)
  d.airQualityRaw = analogRead(MQ135_PIN);
  // Simplified conversion (calibrate with known gas for accuracy)
  d.airQualityPPM = map(d.airQualityRaw, 0, 1023, 400, 2000);

  // LDR: Light level
  d.lightDetected = digitalRead(LDR_PIN);

  // Derived metrics
  d.heatIndex       = dht.computeHeatIndex(t, h, false);
  d.dewPoint        = t - ((100.0 - h) / 5.0);
  d.absoluteHumidity = (6.112 * exp((17.67 * t) / (t + 243.5)) * h * 2.1674) / (273.15 + t);

  d.valid = true;

  Serial.printf("[Sensors] T=%.1f°C H=%.1f%% P=%.1fhPa AQ=%d HI=%.1f\n",
                t, h, d.pressure, d.airQualityRaw, d.heatIndex);
  return d;
}

// ── Publish Sensor Data ───────────────────────────────────────
void publishSensorData(SensorData& d) {
  StaticJsonDocument<512> doc;

  doc["device_id"]     = THING_NAME;
  doc["timestamp"]     = timeClient.getEpochTime();
  doc["iso_time"]      = timeClient.getFormattedTime();
  doc["sequence"]      = ++publishCount;
  doc["firmware"]      = "1.2.0";

  JsonObject sensors = doc.createNestedObject("sensors");
  sensors["temperature"]        = round(d.temperature * 10.0) / 10.0;
  sensors["humidity"]           = round(d.humidity * 10.0) / 10.0;
  sensors["pressure"]           = round(d.pressure * 10.0) / 10.0;
  sensors["altitude"]           = round(d.altitude * 10.0) / 10.0;
  sensors["heat_index"]         = round(d.heatIndex * 10.0) / 10.0;
  sensors["dew_point"]          = round(d.dewPoint * 10.0) / 10.0;
  sensors["abs_humidity"]       = round(d.absoluteHumidity * 100.0) / 100.0;
  sensors["air_quality_raw"]    = d.airQualityRaw;
  sensors["air_quality_ppm"]    = d.airQualityPPM;
  sensors["light_detected"]     = d.lightDetected;

  JsonObject meta = doc.createNestedObject("meta");
  meta["wifi_rssi"]   = WiFi.RSSI();
  meta["free_heap"]   = ESP.getFreeHeap();
  meta["uptime_s"]    = millis() / 1000;
  meta["location"]    = "New Delhi, IN";
  meta["lat"]         = 28.6139;
  meta["lon"]         = 77.2090;

  char payload[512];
  serializeJson(doc, payload);

  if (mqttClient.publish(MQTT_TOPIC_PUB, payload, false)) {
    Serial.printf("[MQTT] Published %u bytes (seq=%u)\n", strlen(payload), publishCount);
  } else {
    Serial.println(F("[MQTT] Publish failed!"));
  }
}

// ── Publish Status ────────────────────────────────────────────
void publishStatus(const char* status) {
  StaticJsonDocument<128> doc;
  doc["device_id"] = THING_NAME;
  doc["status"]    = status;
  doc["timestamp"] = timeClient.getEpochTime();
  char buf[128];
  serializeJson(doc, buf);
  mqttClient.publish(MQTT_TOPIC_STATUS, buf);
}

// ── MQTT Message Handler ──────────────────────────────────────
void onMQTTMessage(char* topic, byte* payload, unsigned int length) {
  String msg = "";
  for (unsigned int i = 0; i < length; i++) msg += (char)payload[i];
  Serial.printf("[CMD] %s → %s\n", topic, msg.c_str());

  StaticJsonDocument<128> doc;
  if (!deserializeJson(doc, msg)) {
    const char* cmd = doc["command"] | "";
    if (strcmp(cmd, "reboot") == 0) {
      publishStatus("rebooting"); delay(500); ESP.restart();
    } else if (strcmp(cmd, "ping") == 0) {
      publishStatus("pong");
    } else if (strcmp(cmd, "read_now") == 0) {
      SensorData d = readAllSensors();
      if (d.valid) publishSensorData(d);
    }
  }
}

// ── LED Helper ────────────────────────────────────────────────
void blinkLED(int times, int ms) {
  for (int i = 0; i < times; i++) {
    digitalWrite(STATUS_LED, HIGH); delay(ms);
    digitalWrite(STATUS_LED, LOW);  delay(ms);
  }
}
