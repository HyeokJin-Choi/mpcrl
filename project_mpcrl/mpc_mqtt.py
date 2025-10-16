#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mpc_mqtt.py
- 라즈베리파이에서 실행하는 메인 러너
- MQTT로 센서 수신 → MPC로 제어값 계산 → MQTT로 액추에이터 명령 퍼블리시
- learning.py 의 LearningMpc + theta_params.json 사용
"""

import json
import time
import threading
import queue
from typing import Dict, Any, Optional

import paho.mqtt.client as mqtt

# === 프로젝트 내부 모듈 (동일 디렉터리에 위치) ===
from learning import LearningMpc  # 당신이 올린 MPC 구현
# theta_params.json 은 같은 폴더에 존재해야 함

# =========================
# ====== 환경 설정부 ======
# =========================

BROKER_HOST = "127.0.0.1"    # EMQX/Mosquitto 브로커 주소
BROKER_PORT = 1883
MQTT_QOS = 1
CLIENT_ID = "rpi-greenhouse-mpc"

# 센서 입력 토픽(둘 중 하나 방식 사용 가능)
# 1) 단일 JSON 합본
TOPIC_SENSOR_COMBINED = "farm/greenhouse/sensors"
# 2) 개별 항목
TOPIC_SENSOR_FIELDS = {
    "temp":  "farm/greenhouse/sensors/temp",
    "hum":   "farm/greenhouse/sensors/hum",
    "co2":   "farm/greenhouse/sensors/co2",
    "light": "farm/greenhouse/sensors/light"
}

# 액추에이터 명령 퍼블리시 토픽
TOPIC_ACTUATORS = {
    "heater":      "farm/greenhouse/actuators/heater",
    "humidifier":  "farm/greenhouse/actuators/humidifier",
    "co2_valve":   "farm/greenhouse/actuators/co2_valve",
    "led":         "farm/greenhouse/actuators/led"
}

# 제어 주기(초)
CONTROL_PERIOD_SEC = 60

# 상태 벡터 구성(예: x=[temp, hum, co2, light])
STATE_KEYS = ["temp", "hum", "co2", "light"]

# 출력 범위/제약 (MPC 파라미터 y_min_k, y_max_k 설정에 활용하려는 의도)
# 실제 프로젝트 조건에 맞게 조정하세요.
Y_BOUNDS = {
    "temp":  (18.0, 24.0),   # °C
    "hum":   (55.0, 75.0),   # %RH
    "co2":   (400.0, 900.0), # ppm
    "light": (150.0, 500.0)  # umol/m2/s (예시)
}

# =========================
# ===== 내부 상태/큐 =====
# =========================

latest_state: Dict[str, float] = {}   # 최신 센서값 캐시
state_lock = threading.Lock()

cmd_queue: "queue.Queue[Dict[str, float]]" = queue.Queue()

# =========================
# ====== 유틸 함수들 =====
# =========================

def _coerce_float(obj: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if obj is None:
            return default
        if isinstance(obj, (int, float)):
            return float(obj)
        return float(str(obj))
    except Exception:
        return default

def build_state_vector() -> Optional[list]:
    """latest_state에서 STATE_KEYS 순서대로 벡터 생성. 하나라도 없다면 None."""
    with state_lock:
        vals = []
        for k in STATE_KEYS:
            v = latest_state.get(k, None)
            fv = _coerce_float(v)
            if fv is None:
                return None
            vals.append(fv)
        return vals

def load_theta_params(path: str = "theta_params.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    params = data.get("params", {})
    if not params:
        raise RuntimeError("theta_params.json 에 'params' 키가 없습니다.")
    return params

# =========================
# ====== MQTT 핸들러 =====
# =========================

def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected rc={rc}")
    # LWT 알림 확인용 (옵션)
    # 센서 토픽 구독
    client.subscribe(TOPIC_SENSOR_COMBINED, qos=MQTT_QOS)
    for t in TOPIC_SENSOR_FIELDS.values():
        client.subscribe(t, qos=MQTT_QOS)

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode("utf-8", errors="ignore")
    # print(f"[MQTT] RX {topic}: {payload}")

    try:
        if topic == TOPIC_SENSOR_COMBINED:
            # {"temp":..., "hum":..., "co2":..., "light":..., "ts":...}
            data = json.loads(payload)
            with state_lock:
                for k in STATE_KEYS:
                    if k in data:
                        latest_state[k] = _coerce_float(data[k], latest_state.get(k))
        else:
            # 개별 항목 JSON: {"value":..., "ts":...}
            for name, t in TOPIC_SENSOR_FIELDS.items():
                if t == topic:
                    data = json.loads(payload)
                    v = _coerce_float(data.get("value", None))
                    if v is not None:
                        with state_lock:
                            latest_state[name] = v
                    break
    except Exception as e:
        print(f"[MQTT] on_message parse error: {e}")

def publish_actuators(client: mqtt.Client, cmd: Dict[str, float]):
    # 액추에이터 명령을 각각 토픽으로 퍼블리시
    # payload: {"value": float, "ts": int}
    ts = int(time.time())
    for name, val in cmd.items():
        topic = TOPIC_ACTUATORS.get(name)
        if not topic:
            continue
        payload = json.dumps({"value": float(val), "ts": ts})
        client.publish(topic, payload=payload, qos=MQTT_QOS, retain=False)

# =========================
# ====== MPC 실행부  ======
# =========================

class MpcRunner:
    def __init__(self):
        print("[MPC] 초기화 시작")
        # 학습 파라미터 로드
        self.theta = load_theta_params("theta_params.json")

        # MPC 인스턴스 생성
        # 주의: learning.py 의 시그니처에 맞게 인자 조정 필요할 수 있음
        self.mpc = LearningMpc(
            greenhouse_env=None,
            test=None,
            np_random=None,
            prediction_horizon=6*4,      # 6시간 * 4step/h = 24 step (예시)
            prediction_model="rk4",
            constrain_control_rate=True
        )

        # 파라미터 주입: LearningMpc 가 제공하는 방식에 맞춰 세팅
        # 여기서는 파라미터 이름 그대로 set_value 형태를 가정합니다.
        # learning.py에서 self.parameter(...)로 등록한 이름들에 값 바인딩.
        self._apply_theta_to_mpc(self.theta)
        print("[MPC] 초기화 완료")

    def _apply_theta_to_mpc(self, theta: Dict[str, Any]):
        """
        theta_params.json 에 있는 파라미터를 MPC에 바인딩.
        - V0, c_u, c_dy, c_y, y_fin, w, olb, oub, p_0..p_N 등
        """
        # 배열/스칼라 섞여있으므로 안전하게 처리
        def as1d(name):
            v = theta.get(name)
            if v is None:
                return None
            if isinstance(v, (list, tuple)):
                return v
            return [v]

        # 비용/제약 파라미터
        for name in ["V0", "c_dy", "c_y", "y_fin"]:
            v = as1d(name)
            if v is not None:
                self.mpc.set_value(name, v)

        for name in ["c_u", "w", "olb", "oub"]:
            v = as1d(name)
            if v is not None:
                self.mpc.set_value(name, v)

        # 동역학 파라미터 p_0...p_27 (개수는 모델에 따라 다름)
        # 못 찾으면 그냥 스킵
        i = 0
        while True:
            key = f"p_{i}"
            if key not in theta:
                break
            self.mpc.set_value(key, [theta[key]])
            i += 1

        # 출력 제약(y_min_k, y_max_k)은 매 스텝 update 예정
        # 외란 d 도 매 스텝 0 또는 예보치 반영

    def step(self, x: list) -> Dict[str, float]:
        """
        상태 x 로부터 최적 제어 입력 계산.
        - LearningMpc.solve(...) 의 시그니처에 맞게 호출하세요.
        - 본 예시는 self.mpc.solve(x) 를 가정.
        """
        # 출력 제약 세팅 (예: 온실 출력이 STATE_KEYS 순서와 1:1 매핑일 때)
        # 실사용시 모델 출력/상태 차원 정의에 맞게 보정 필요
        try:
            for k, key in enumerate(STATE_KEYS):
                ymin, ymax = Y_BOUNDS[key]
                self.mpc.set_value(f"y_min_{k}", [[ymin] for _ in range(len(STATE_KEYS))])
                self.mpc.set_value(f"y_max_{k}", [[ymax] for _ in range(len(STATE_KEYS))])
        except Exception:
            # learning.py 구성과 다르면 위 구간을 주석 처리하거나 맞게 수정
            pass

        # 외란 d (예: 0으로)
        try:
            self.mpc.set_value("d", [[0.0] * self.mpc.prediction_horizon for _ in range(len(STATE_KEYS))])
        except Exception:
            pass

        # === 핵심: MPC 최적화 호출 ===
        try:
            u = self.mpc.solve(x)  # 학습된 파라미터 적용된 최적화
        except Exception as e:
            print(f"[MPC] solve 실패, 안전모드로 대체: {e}")
            u = self._fallback_policy(x)

        # 액추에이터 맵핑 (모델 입력 순서에 맞춰 매핑 수정)
        # 예시: [heater, humidifier, co2_valve, led] 4채널 가정
        cmd = {
            "heater":     float(u[0]) if len(u) > 0 else 0.0,
            "humidifier": float(u[1]) if len(u) > 1 else 0.0,
            "co2_valve":  float(u[2]) if len(u) > 2 else 0.0,
            "led":        float(u[3]) if len(u) > 3 else 0.0,
        }
        return cmd

    def _fallback_policy(self, x: list) -> list:
        """MPC 실패 시 간단한 P제어 대체(안전모드). 프로젝트에 맞게 수정."""
        # 목표값(중간치)
        setpoints = [
            sum(Y_BOUNDS[k]) / 2.0 for k in STATE_KEYS
        ]
        kp = [0.1, 0.1, 0.05, 0.05]
        u = []
        for i in range(min(len(setpoints), len(x), len(kp))):
            u.append(kp[i] * (setpoints[i] - float(x[i])))
        # 액추에이터 4채널 가정
        while len(u) < 4:
            u.append(0.0)
        return u

# =========================
# ====== 메인 루프  =======
# =========================

def control_loop(client: mqtt.Client, mpc_runner: MpcRunner):
    """CONTROL_PERIOD_SEC 주기로 상태를 읽고 MPC 계산→퍼블리시."""
    print("[Main] 제어 루프 시작")
    while True:
        x = build_state_vector()
        if x is None:
            print("[Main] 센서값 대기중... (필요 키: {})".format(STATE_KEYS))
            time.sleep(1)
            continue

        cmd = mpc_runner.step(x)
        publish_actuators(client, cmd)
        print(f"[Main] x={x} → cmd={cmd}")
        time.sleep(CONTROL_PERIOD_SEC)

def main():
    # MQTT 클라이언트 설정
    client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)
    client.will_set("farm/greenhouse/rpi/status", json.dumps({"status": "offline"}), qos=MQTT_QOS, retain=True)
    client.on_connect = on_connect
    client.on_message = on_message

    # 브로커 연결
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()

    # 상태 알림
    client.publish("farm/greenhouse/rpi/status", json.dumps({"status": "online"}), qos=MQTT_QOS, retain=True)

    # MPC 러너 시작
    mpc_runner = MpcRunner()

    try:
        control_loop(client, mpc_runner)
    except KeyboardInterrupt:
        print("\n[Main] 종료 신호 수신")
    finally:
        client.publish("farm/greenhouse/rpi/status", json.dumps({"status": "offline"}), qos=MQTT_QOS, retain=True)
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
