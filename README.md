# NimbusCast — IoT Cloud Weather Monitoring System
**ESP8266/ESP32 → AWS IoT Core → Lambda → S3 → LSTM ML → Dashboard**

---

## Architecture

```
ESP8266/ESP32 (DHT22 + BMP280 + MQ-135)
      │  MQTT TLS (port 8883)
      ▼
AWS IoT Core ── Thing: nimbus-node-001
      │  IoT Rule → SQL: SELECT * FROM 'nimbus/sensors/data'
      ▼
AWS Lambda (nimbus-iot-ingestion, Python 3.12)
      ├──► S3  raw/year=Y/month=M/day=D/hour=H/<device>_<ts>.json
      ├──► S3  daily/Y/M/D/nimbus-node-001.ndjson  (Athena-ready)
      ├──► DynamoDB  nimbus-sensor-readings  (hot cache, TTL 7d)
      └──► SNS  nimbus-weather-alerts  (threshold violations)

AWS EventBridge (cron 0 2 * * ?)
      │  Daily retrain trigger
      ▼
Lambda (nimbus-ml-inference)
      ├── Load S3 daily/*.ndjson → Feature engineering
      ├── Train BiLSTM(128) + BiLSTM(64) + Dense(64)
      ├── Save s3://nimbus-weather-data/models/nimbus_lstm_v1.h5
      └── Save s3://nimbus-weather-data/models/metadata.json

API Gateway → Lambda (nimbus-api) → DynamoDB + S3
      │
      ▼
Dashboard (frontend/dashboard.html)
  ├── Live ESP sensor readings (WebSocket simulation)
  ├── OWM API integration (current + forecast)
  ├── S3 bucket explorer
  ├── LSTM model predictions
  └── AWS SNS alert management
```

---

## Project Files

```
nimbus/
├── firmware/
│   └── esp_weather_node.ino    ← Flash to ESP8266/ESP32
├── backend/
│   ├── lambda_ingestion.py     ← IoT → S3 + DynamoDB Lambda
│   ├── cloudformation.yaml     ← Deploy all AWS infra
│   └── (add lambda_api.py, lambda_ml.py for full deploy)
├── ml/
│   └── lstm_model.py           ← BiLSTM training + inference
├── frontend/
│   └── dashboard.html          ← Full dashboard (open in browser)
└── README.md
```

---

## Step 1: Hardware Setup

### Parts List
| Component | Purpose | Pin |
|-----------|---------|-----|
| ESP8266 NodeMCU / ESP32 DevKit | Main controller | — |
| DHT22 | Temperature + Humidity | D4 (GPIO2) |
| BMP280 | Pressure + Altitude | I2C: D1(SCL), D2(SDA) |
| MQ-135 | Air Quality (CO2/NH3) | A0 (analog) |
| 10kΩ resistor | DHT22 pull-up | D4 → 3.3V |

### Wiring
```
DHT22:   VCC→3.3V, GND→GND, DATA→D4 (+ 10kΩ to 3.3V)
BMP280:  VCC→3.3V, GND→GND, SDA→D2, SCL→D1
MQ-135:  VCC→5V,   GND→GND, AOUT→A0
```

---

## Step 2: AWS Infrastructure

### Deploy with CloudFormation
```bash
aws cloudformation deploy \
  --template-file backend/cloudformation.yaml \
  --stack-name nimbus-stack \
  --parameter-overrides AlertEmail=ojas@youremail.com \
  --capabilities CAPABILITY_NAMED_IAM \
  --region ap-south-1
```

### Get Outputs
```bash
aws cloudformation describe-stacks \
  --stack-name nimbus-stack \
  --query 'Stacks[0].Outputs'
```

### Create IoT Thing & Certificates
```bash
# Create thing
aws iot create-thing --thing-name nimbus-node-001

# Create certificate
aws iot create-keys-and-certificate \
  --set-as-active \
  --certificate-pem-outfile device.crt \
  --public-key-outfile device.key.pub \
  --private-key-outfile device.key

# Attach policy (create policy first)
aws iot create-policy \
  --policy-name NimbusNodePolicy \
  --policy-document '{
    "Version":"2012-10-17",
    "Statement":[{
      "Effect":"Allow",
      "Action":["iot:Connect","iot:Publish","iot:Subscribe","iot:Receive"],
      "Resource":"arn:aws:iot:ap-south-1:*:*"
    }]
  }'

aws iot attach-policy \
  --policy-name NimbusNodePolicy \
  --target <certificate-arn>

aws iot attach-thing-principal \
  --thing-name nimbus-node-001 \
  --principal <certificate-arn>
```

---

## Step 3: Flash ESP Firmware

### Arduino IDE Setup
1. Add ESP8266 board: `http://arduino.esp8266.com/stable/package_esp8266com_index.json`
2. Install libraries: `PubSubClient`, `ArduinoJson`, `DHT sensor library`, `Adafruit BMP280`, `NTPClient`
3. Open `firmware/esp_weather_node.ino`
4. Edit credentials:
   ```cpp
   #define WIFI_SSID      "YourWiFi"
   #define WIFI_PASSWORD  "YourPassword"
   #define AWS_IOT_ENDPOINT "xxxx.iot.ap-south-1.amazonaws.com"
   ```
5. Paste certificates from `device.crt`, `device.key`, and AWS Root CA
6. Select board: **NodeMCU 1.0 (ESP-12E Module)**
7. Upload!

### Verify Connection (Serial Monitor @ 115200)
```
[NimbusCast] ESP Weather Node Starting...
[OK] BMP280 initialized
[WiFi] Connected! IP: 192.168.1.105
[MQTT] Connected to AWS IoT Core!
[MQTT] Subscribed to nimbus/sensors/command
[NimbusCast] Ready. Publishing every 30s...
[Sensors] T=31.4°C H=62.0% P=1008.3hPa AQ=487 HI=34.1
[MQTT] Published 412 bytes (seq=1)
```

---

## Step 4: Deploy Lambda Functions

```bash
# Package ingestion lambda
zip lambda_ingestion.zip backend/lambda_ingestion.py

aws lambda update-function-code \
  --function-name nimbus-iot-ingestion \
  --zip-file fileb://lambda_ingestion.zip

# Package ML lambda
pip install tensorflow scikit-learn -t ml_pkg/
cp ml/lstm_model.py ml_pkg/
zip -r lambda_ml.zip ml_pkg/

aws lambda update-function-code \
  --function-name nimbus-ml-inference \
  --zip-file fileb://lambda_ml.zip
```

---

## Step 5: Train ML Model

```bash
# Install deps
pip install tensorflow scikit-learn boto3 pandas numpy

# Train (reads from S3, saves model back to S3)
python ml/lstm_model.py --train --epochs 50

# Test inference
python ml/lstm_model.py --predict
```

---

## Step 6: Open Dashboard

Simply open `frontend/dashboard.html` in any browser.

### For production, deploy to:
```bash
# S3 static website
aws s3 sync frontend/ s3://nimbus-dashboard-bucket/ --acl public-read

# Or CloudFront distribution for HTTPS
```

---

## S3 Data Schema

### Raw JSON (every 30s per node)
```json
{
  "device_id": "nimbus-node-001",
  "timestamp": 1710849600,
  "server_iso": "2024-03-19T14:00:00+00:00",
  "date_partition": "2024/03/19",
  "sensors": {
    "temperature": 31.4,
    "humidity": 62.0,
    "pressure": 1008.3,
    "altitude": 216.4,
    "heat_index": 34.1,
    "dew_point": 23.2,
    "abs_humidity": 18.3,
    "air_quality_raw": 487,
    "air_quality_ppm": 892
  },
  "meta": {
    "wifi_rssi": -62,
    "free_heap": 28432,
    "uptime_s": 3600,
    "location": "New Delhi, IN",
    "lat": 28.6139,
    "lon": 77.2090
  },
  "aqi_category": "Moderate",
  "comfort_index": 72.4
}
```

### Query with AWS Athena
```sql
CREATE EXTERNAL TABLE nimbus_raw (
  device_id STRING, timestamp BIGINT,
  sensors STRUCT<temperature:FLOAT, humidity:FLOAT, pressure:FLOAT>
)
PARTITIONED BY (year STRING, month STRING, day STRING, hour STRING)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://nimbus-weather-data/raw/';

SELECT AVG(sensors.temperature), MAX(sensors.humidity)
FROM nimbus_raw
WHERE year='2024' AND month='03';
```

---

## Estimated AWS Costs (1 ESP node, 30s interval)

| Service | Usage/month | Cost |
|---------|-------------|------|
| IoT Core | ~87K messages | ~$0.09 |
| Lambda | ~87K invocations | Free tier |
| S3 | ~100MB storage | ~$0.002 |
| DynamoDB | On-demand | ~$0.05 |
| SNS | ~10 alerts | ~$0.00 |
| **Total** | | **~$0.15/month** |

---

*Built by Ojas Saraswat — NimbusCast v1.2*
