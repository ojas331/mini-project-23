"""
================================================================
 NimbusCast — AWS Lambda: IoT Data Ingestion
 Trigger:  AWS IoT Core Rule → Lambda
 Actions:  1. Store raw JSON to S3 (partitioned by date/device)
           2. Write latest reading to DynamoDB (hot cache)
           3. Trigger ML inference if enough data accumulated
           4. Push anomaly alerts to SNS

 Deploy:   AWS Lambda (Python 3.12, 256MB, 30s timeout)
           IAM Role: nimbus-lambda-role
           Env vars: S3_BUCKET, DYNAMODB_TABLE, SNS_TOPIC_ARN,
                     ML_ENDPOINT_NAME (SageMaker)
================================================================
"""

import json
import boto3
import os
import time
import math
from datetime import datetime, timezone
from decimal import Decimal

# AWS clients (initialized once, reused across invocations)
s3         = boto3.client('s3')
dynamodb   = boto3.resource('dynamodb')
sns        = boto3.client('sns')

# Config from environment variables
S3_BUCKET        = os.environ.get('S3_BUCKET', 'nimbus-weather-data')
DYNAMODB_TABLE   = os.environ.get('DYNAMODB_TABLE', 'nimbus-sensor-readings')
SNS_TOPIC_ARN    = os.environ.get('SNS_TOPIC_ARN', '')
ML_ENDPOINT_NAME = os.environ.get('ML_ENDPOINT_NAME', 'nimbus-lstm-endpoint')

# Alert thresholds
THRESHOLDS = {
    'temperature':  {'min': -10, 'max': 50,  'unit': '°C'},
    'humidity':     {'min': 5,   'max': 98,  'unit': '%'},
    'pressure':     {'min': 920, 'max': 1080,'unit': 'hPa'},
    'air_quality_ppm': {'min': 0, 'max': 1000,'unit': 'ppm'},
}


def lambda_handler(event, context):
    """
    Main Lambda entry point.
    Event structure: raw MQTT payload from IoT Core Rule.
    """
    print(f"[Lambda] Received event: {json.dumps(event)}")

    try:
        # 1. Parse & validate payload
        data = parse_payload(event)

        # 2. Enrich with derived fields
        data = enrich_data(data)

        # 3. Store to S3 (raw + partitioned)
        s3_key = store_to_s3(data)
        print(f"[S3] Stored at: s3://{S3_BUCKET}/{s3_key}")

        # 4. Update DynamoDB hot cache
        store_to_dynamodb(data)

        # 5. Check thresholds → SNS alerts
        alerts = check_thresholds(data)
        if alerts:
            publish_alerts(alerts, data)

        # 6. Return success
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'ok',
                's3_key': s3_key,
                'device_id': data.get('device_id'),
                'alerts_triggered': len(alerts),
            })
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        raise


def parse_payload(event):
    """Parse and validate incoming IoT payload."""
    if isinstance(event, str):
        data = json.loads(event)
    elif isinstance(event, dict):
        data = event
    else:
        raise ValueError(f"Unexpected event type: {type(event)}")

    required = ['device_id', 'sensors', 'timestamp']
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    return data


def enrich_data(data):
    """Add server-side timestamps, AQI category, comfort index."""
    sensors = data.get('sensors', {})

    # Server timestamp (authoritative)
    now = datetime.now(timezone.utc)
    data['server_timestamp']  = int(now.timestamp())
    data['server_iso']        = now.isoformat()
    data['date_partition']    = now.strftime('%Y/%m/%d')
    data['hour_partition']    = now.strftime('%H')

    # AQI category from PPM
    ppm = sensors.get('air_quality_ppm', 400)
    data['aqi_category'] = classify_aqi(ppm)

    # Comfort index (0-100)
    t = sensors.get('temperature', 25)
    h = sensors.get('humidity', 60)
    data['comfort_index'] = compute_comfort_index(t, h)

    # Apparent temperature (Steadman)
    ws = 0  # No wind sensor in basic kit
    data['apparent_temp'] = t + 0.33 * (h / 100 * 6.105 * math.exp(25.22 * (t - 273.16) / t - 5.31 * math.log(t / 273.16))) - 4.0

    return data


def compute_comfort_index(temp, humidity):
    """Simple comfort score 0-100 (100 = perfect)."""
    t_score   = max(0, 100 - abs(temp - 22) * 5)
    h_score   = max(0, 100 - abs(humidity - 50) * 1.5)
    return round((t_score * 0.6 + h_score * 0.4), 1)


def classify_aqi(ppm):
    if ppm < 600:   return 'Good'
    if ppm < 800:   return 'Moderate'
    if ppm < 1000:  return 'Unhealthy'
    return 'Hazardous'


def store_to_s3(data):
    """
    Store data to S3 with Hive-style partitioning:
    s3://nimbus-weather-data/raw/year=2024/month=03/day=19/hour=14/<device>_<ts>.json

    Also appends to daily aggregate file for efficient querying with Athena.
    """
    device_id = data.get('device_id', 'unknown')
    ts        = data.get('server_timestamp', int(time.time()))
    date_part = data.get('date_partition', '2024/01/01')
    hour_part = data.get('hour_partition', '00')
    year, month, day = date_part.split('/')

    # Individual reading
    s3_key = (
        f"raw/year={year}/month={month}/day={day}/hour={hour_part}/"
        f"{device_id}_{ts}.json"
    )

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=json.dumps(data, indent=2),
        ContentType='application/json',
        StorageClass='STANDARD_IA',          # cheaper for infrequent access
        Metadata={
            'device-id':   device_id,
            'temperature': str(data['sensors'].get('temperature', 0)),
            'ingested-at': data['server_iso'],
        }
    )

    # Append to daily NDJSON (Newline-Delimited JSON) for Athena
    daily_key = f"daily/{year}/{month}/{day}/{device_id}.ndjson"
    try:
        existing = s3.get_object(Bucket=S3_BUCKET, Key=daily_key)
        existing_body = existing['Body'].read().decode('utf-8')
    except s3.exceptions.NoSuchKey:
        existing_body = ''

    new_body = existing_body + json.dumps(data) + '\n'
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=daily_key,
        Body=new_body.encode('utf-8'),
        ContentType='application/x-ndjson',
    )

    return s3_key


def store_to_dynamodb(data):
    """Write latest reading to DynamoDB for real-time dashboard queries."""
    table = dynamodb.Table(DYNAMODB_TABLE)

    sensors = data.get('sensors', {})

    # Convert floats to Decimal (DynamoDB requirement)
    def to_decimal(v):
        if isinstance(v, float):
            return Decimal(str(round(v, 4)))
        return v

    item = {
        'device_id':        data['device_id'],
        'timestamp':        data['server_timestamp'],
        'iso_time':         data['server_iso'],
        'temperature':      to_decimal(sensors.get('temperature')),
        'humidity':         to_decimal(sensors.get('humidity')),
        'pressure':         to_decimal(sensors.get('pressure')),
        'altitude':         to_decimal(sensors.get('altitude')),
        'heat_index':       to_decimal(sensors.get('heat_index')),
        'dew_point':        to_decimal(sensors.get('dew_point')),
        'air_quality_ppm':  to_decimal(sensors.get('air_quality_ppm')),
        'aqi_category':     data.get('aqi_category', 'Good'),
        'comfort_index':    to_decimal(data.get('comfort_index')),
        'wifi_rssi':        data.get('meta', {}).get('wifi_rssi', 0),
        'uptime_s':         data.get('meta', {}).get('uptime_s', 0),
        'ttl':              int(time.time()) + 86400 * 7,  # expire after 7 days
    }

    table.put_item(Item=item)
    print(f"[DynamoDB] Wrote item for {data['device_id']}")


def check_thresholds(data):
    """Return list of triggered threshold violations."""
    sensors  = data.get('sensors', {})
    triggered = []

    for metric, limits in THRESHOLDS.items():
        val = sensors.get(metric)
        if val is None:
            continue
        if val > limits['max']:
            triggered.append({
                'metric':    metric,
                'value':     val,
                'threshold': limits['max'],
                'direction': 'HIGH',
                'unit':      limits['unit'],
                'severity':  'WARNING' if val < limits['max'] * 1.1 else 'CRITICAL',
            })
        elif val < limits['min']:
            triggered.append({
                'metric':    metric,
                'value':     val,
                'threshold': limits['min'],
                'direction': 'LOW',
                'unit':      limits['unit'],
                'severity':  'WARNING',
            })
    return triggered


def publish_alerts(alerts, data):
    """Publish threshold alerts to SNS for notifications."""
    if not SNS_TOPIC_ARN:
        print("[SNS] No topic ARN configured, skipping alerts")
        return

    device_id = data.get('device_id', 'unknown')
    for alert in alerts:
        msg = {
            'device_id': device_id,
            'alert':     alert,
            'location':  data.get('meta', {}).get('location', 'Unknown'),
            'timestamp': data.get('server_iso'),
        }
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"[NimbusCast] {alert['severity']}: {alert['metric']} is {alert['direction']}",
            Message=json.dumps(msg, indent=2),
            MessageAttributes={
                'severity': {'DataType': 'String', 'StringValue': alert['severity']},
                'device':   {'DataType': 'String', 'StringValue': device_id},
            }
        )
        print(f"[SNS] Alert published: {alert['metric']} = {alert['value']}{alert['unit']}")
