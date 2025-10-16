#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mqtt_actuator_monitor.py
- 액추에이터 명령 토픽들을 구독해서 콘솔에 출력
- mpc_mqtt.py 가 퍼블리시하는 명령을 모니터링
"""

import json
import paho.mqtt.client as mqtt

BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883
QOS = 1

TOPIC_ACTUATORS = [
    "farm/greenhouse/actuators/heater",
    "farm/greenhouse/actuators/humidifier",
    "farm/greenhouse/actuators/co2_valve",
    "farm/greenhouse/actuators/led"
]

def on_connect(client, userdata, flags, rc):
    print(f"[MON] Connected rc={rc}")
    for t in TOPIC_ACTUATORS:
        client.subscribe(t, qos=QOS)

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8", errors="ignore")
        data = json.loads(payload)
    except Exception:
        data = payload
    print(f"[MON] {msg.topic}  ←  {data}")

def main():
    client = mqtt.Client(client_id="actuator-monitor")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_HOST, BROKER_PORT, 60)
    client.loop_forever()

if __name__ == "__main__":
    main()
