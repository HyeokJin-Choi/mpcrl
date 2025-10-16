#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mqtt_sensor_sim.py
- MQTT로 가짜 센서값을 주기적으로 퍼블리시 (테스트용)
- mpc_mqtt.py 와 같은 브로커/토픽 설정을 사용하세요.
"""

import json
import time
import random
import paho.mqtt.client as mqtt

BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883
QOS = 1

TOPIC_SENSOR_COMBINED = "farm/greenhouse/sensors"
TOPIC_SENSOR_FIELDS = {
    "temp":  "farm/greenhouse/sensors/temp",
    "hum":   "farm/greenhouse/sensors/hum",
    "co2":   "farm/greenhouse/sensors/co2",
    "light": "farm/greenhouse/sensors/light"
}

USE_COMBINED = True   # True면 합본 JSON, False면 개별 토픽 발행
INTERVAL_SEC = 5

def main():
    client = mqtt.Client(client_id="sensor-sim")
    client.connect(BROKER_HOST, BROKER_PORT, 60)
    client.loop_start()

    try:
        while True:
            # 랜덤 센서값 (현실적인 범위 근처)
            temp  = round(random.uniform(19.0, 27.0), 2)
            hum   = round(random.uniform(50.0, 80.0), 2)
            co2   = round(random.uniform(380.0, 950.0), 1)
            light = round(random.uniform(100.0, 600.0), 1)

            ts = int(time.time())

            if USE_COMBINED:
                payload = json.dumps({"temp": temp, "hum": hum, "co2": co2, "light": light, "ts": ts})
                client.publish(TOPIC_SENSOR_COMBINED, payload=payload, qos=QOS, retain=False)
                print("[SIM] PUB combined:", payload)
            else:
                for name, val in [("temp", temp), ("hum", hum), ("co2", co2), ("light", light)]:
                    t = TOPIC_SENSOR_FIELDS[name]
                    payload = json.dumps({"value": val, "ts": ts})
                    client.publish(t, payload=payload, qos=QOS, retain=False)
                    print(f"[SIM] PUB {name}: {payload}")

            time.sleep(INTERVAL_SEC)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
