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
import numpy as np
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
def _mpc_set_any(mpc_obj, name, val):
    """
    csnlp/learning.MPC 구현 차이에 따라 파라미터/변수 값을 세팅하는
    가능한 메서드를 순차적으로 시도한다.
    성공하면 True, 전부 실패하면 False를 반환.
    """
    try_methods = [
        "set_value",        # 일부 구현
        "set_par",          # 다른 구현
        "set_parameter",    # 다른 구현
        "set_var",          # 변수 세팅용 구현일 수도
        "assign",           # 드물게 assign 사용
        "value",            # casadi/optibase 계열일 때 값 주입
    ]

    for m in try_methods:
        if hasattr(mpc_obj, m):
            try:
                getattr(mpc_obj, m)(name, val)
                return True
            except Exception:
                pass

    # nlp 내부로 우회 시도
    if hasattr(mpc_obj, "nlp"):
        nlp = getattr(mpc_obj, "nlp")
        for m in try_methods:
            if hasattr(nlp, m):
                try:
                    getattr(nlp, m)(name, val)
                    return True
                except Exception:
                    pass

    return False



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

# 변경(v5)
def on_connect(client, userdata, flags, reasonCode, properties=None):
    print(f"[MQTT] Connected reasonCode={reasonCode}")
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

# === Minimal test config stub for LearningMpc ===
class SimpleTestConfig:
    """
    learning.py가 즉시 참조하는 Test 설정의 필수 필드만 가진 더미 객체.
    """
    def __init__(self,
                 discount_factor: float = 0.99,
                 p_perturb=None,
                 p_learn=None):
        self.discount_factor = float(discount_factor)
        # 학습/고정 파라미터 사전(learning.py에서 바로 수정하므로 dict 여야 함)
        self.learnable_pars_init = {}
        self.fixed_pars = {}
        # 파라미터 섭동/학습 인덱스
        self.p_perturb = list(p_perturb) if p_perturb is not None else []
        self.p_learn = list(p_learn) if p_learn is not None else []



# === Minimal greenhouse env stub for LearningMpc ===
class SimpleGreenhouseEnv:
    """
    LearningMpc가 참조하는 최소 속성만 제공하는 더미 환경.
    필요한 경우 속성을 추가하세요.
    """
    def __init__(self, nx: int, nu: int, ny: int, nd: int, dt: float):
        # 필수 차원 정보
        self.nx = nx    # state dimension
        self.nu = nu    # control/actuator dimension
        self.ny = ny    # output dimension
        self.nd = nd    # disturbance dimension

        # 샘플링 주기: learning.py가 ts를 찾으므로 둘 다 제공합니다.
        self.dt = float(dt)
        self.ts = float(dt)   # ←★ 핵심: ts 별칭

        # (방어적 기본값) 제약/경계: 넉넉한 범위로 초기화
        BIG = 1e9
        self.x_lb = [-BIG] * nx
        self.x_ub = [ BIG] * nx
        self.u_lb = [-BIG] * nu
        self.u_ub = [ BIG] * nu
        self.y_lb = [-BIG] * ny
        self.y_ub = [ BIG] * ny

        # (옵션) 이름 정보가 필요할 수도 있어 기본값 부여
        self.state_names = getattr(self, "state_names", None)
        self.input_names = getattr(self, "input_names", None)
        self.output_names = getattr(self, "output_names", None)



# =========================
# ====== MPC 실행부  ======
# =========================
class MpcRunner:
    def __init__(self):
        print("[MPC] Init start")
        self.theta = load_theta_params("theta_params.json")

        env_stub = SimpleGreenhouseEnv(
            nx=len(STATE_KEYS),
            nu=3,
            ny=len(STATE_KEYS),
            nd=len(STATE_KEYS),
            dt=float(CONTROL_PERIOD_SEC)
        )

        # ★ Test 설정 더미
        test_stub = SimpleTestConfig(
            discount_factor=0.99,
            p_perturb=[],   # 파라미터 섭동 없음
            p_learn=[]      # 학습 파라미터 없음(필요시 인덱스 넣으세요)
        )

        self.mpc = LearningMpc(
            greenhouse_env=env_stub,
            test=test_stub,          # ★ 여기!
            np_random=None,
            prediction_horizon=6*4,
            prediction_model="rk4",
            constrain_control_rate=True
        )

        self._apply_theta_to_mpc(self.theta)
        print("[MPC] 초기화 완료")

    def _apply_theta_to_mpc(self, theta: Dict[str, Any]):
        """
        theta_params.json 에 있는 파라미터를 MPC에 바인딩.
        - V0, c_u, c_dy, c_y, y_fin, w, olb, oub, p_0..p_N 등
        """
        # 1) 키 전량 시도(알 수 없는 키는 WARN 후 스킵)
        for name, v in theta.items():
            ok = _mpc_set_any(self.mpc, name, v)
            if not ok:
                print(f"[MPC][WARN] cannot set '{name}' via known APIs; skipped.")

        # 2) 스칼라/배열 혼재 방지용 헬퍼
        def as1d(key):
            v = theta.get(key)
            if v is None:
                return None
            return list(v) if isinstance(v, (list, tuple)) else [v]

        # 3) 비용/제약 대표 키들 재확인(있을 때만 세팅)
        for key in ["V0", "c_dy", "c_y", "y_fin", "c_u", "w", "olb", "oub"]:
            v = as1d(key)
            if v is not None:
                _mpc_set_any(self.mpc, key, v)

        # 4) 동역학 파라미터 p_0, p_1, ... 연속 세팅
        i = 0
        while True:
            key = f"p_{i}"
            if key not in theta:
                break
            _mpc_set_any(self.mpc, key, [theta[key]])
            i += 1
        # 출력 제약(y_min_k/y_max_k), 외란(d)은 step()에서 매 루프 갱신
        
    def _solve_with_formats(self, x):
        import numpy as np
        
        N = self.mpc.prediction_horizon
        nx = len(STATE_KEYS)
        nd = len(STATE_KEYS)  # 현재 외란 차원 가정

        # 1) x_0
        params = {
            "x_0": np.asarray(x, dtype=float).reshape(nx, 1)
        }

        # 2) d (외란) — 일단 0
        params["d"] = np.zeros((nd, N))

        # 3) 출력 제약 y_min_k / y_max_k (k = 0..N-1)
        ymin_vec = np.array([Y_BOUNDS[k][0] for k in STATE_KEYS], dtype=float).reshape(nx,1)
        ymax_vec = np.array([Y_BOUNDS[k][1] for k in STATE_KEYS], dtype=float).reshape(nx,1)
        for k in range(N):
            params[f"y_min_{k}"] = ymin_vec
            params[f"y_max_{k}"] = ymax_vec

        # 4) 비용/가중치/경계 등 (θ에서 가져오되, 내부 이름과 형식을 맞춰야 함)
        #   - 지금은 내부 파라미터명이 정확히 뭔지 몰라서 WARN이 납니다.
        #   - 만약 내부가 'p'라는 벡터로 등록되어 있다면:
        # params["p"] = np.asarray([theta[f"p_{i}"] for i in range(PDIM)], dtype=float).reshape(-1,1)

        # 5) solve 호출을 "딕셔너리 한 방에"
        u = self.mpc.solve(params)
        
        last_exc = None

        # 후보 입력 포맷들 차례대로 시도
        candidates = [
            np.asarray(x, dtype=float).reshape(-1, 1),                        # (nx,1) numpy
            {"x":  np.asarray(x, dtype=float).reshape(-1, 1)},               # dict 'x'
            {"x0": np.asarray(x, dtype=float).reshape(-1, 1)},               # dict 'x0'
        ]

        for i, arg in enumerate(candidates, 1):
            try:
                return self.mpc.solve(arg)
            except Exception as e:
                last_exc = e  # 가장 최근 예외 저장하고 다음 포맷 시도
                # 필요하면 어떤 포맷 실패했는지 로그
                # print(f"[MPC] solve try#{i} failed: {e}")

        # 전부 실패하면 마지막 예외를 포함해서 올림
        raise RuntimeError(f"solve input formats all failed: {last_exc}")


    def _clamp_cmd(self, cmd, lo=0.0, hi=1.0):
        for k in cmd:
            v = float(cmd[k])
            cmd[k] = hi if v > hi else lo if v < lo else v
        return cmd


    def step(self, x: list) -> Dict[str, float]:
        """
        상태 x 로부터 최적 제어 입력 계산.
        - LearningMpc.solve(...) 의 시그니처에 맞게 호출하세요.
        - 본 예시는 self.mpc.solve(x) 를 가정.
        """
        # 출력 제약 세팅 (예: 온실 출력이 STATE_KEYS 순서와 1:1 매핑일 때)
        # 실사용시 모델 출력/상태 차원 정의에 맞게 보정 필요
        try:
            params = self._build_params_for_solve(x)
            u = self.mpc.solve(params)        # ★ 딕셔너리 한 방에
            # u shape에 따라 리스트로 변환
            u = np.asarray(u).reshape(-1).tolist()
        except Exception as e:
            print(f"[MPC] Solve failed, fallback policy used: {e}")
            u = self._fallback_policy(x)

        # 액추에이터 매핑 + 안전 클램핑(실기 보호용: 남겨두는 걸 권장)
        cmd = {
            "heater":     float(u[0]) if len(u) > 0 else 0.0,
            "humidifier": float(u[1]) if len(u) > 1 else 0.0,
            "co2_valve":  float(u[2]) if len(u) > 2 else 0.0,
            "led":        float(u[3]) if len(u) > 3 else 0.0,  # nu=3이므로 보통 0.0
        }
        # 하드웨어 보호용 클램핑(내부 제약이 먹어도 안전장치로 유지 권장)
        for k in cmd:
            cmd[k] = max(0.0, min(1.0, cmd[k]))
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
        
    def _build_params_for_solve(self, x):
        """
        csnlp 기반 MPC가 요구하는 파라미터(에러 메시지에 나온 이름들)를
        한 번에 dict로 만들어 solve(params)로 넘긴다.
        """
        nx = len(STATE_KEYS)
        ny = len(STATE_KEYS)   # 출력=상태 1:1 가정
        nd = len(STATE_KEYS)
        N  = int(getattr(self.mpc, "prediction_horizon", 24))  # 이름상 0..N 포함이면 N+1개 필요할 수 있음

        params = {}

        # 0) 초기상태
        params["x_0"] = np.asarray(x, dtype=float).reshape(nx, 1)

        # 1) 외란 d (일단 0)
        params["d"] = np.zeros((nd, N), dtype=float)

        # 2) 출력 제약: y_min_k / y_max_k  (에러에 y_min_24, y_max_24가 보여서 N 포함 가능)
        ymin_vec = np.array([Y_BOUNDS[k][0] for k in STATE_KEYS], dtype=float).reshape(ny, 1)
        ymax_vec = np.array([Y_BOUNDS[k][1] for k in STATE_KEYS], dtype=float).reshape(ny, 1)
        for k in range(N + 1):  # ★ 중요: 0..N 모두 넣기
            params[f"y_min_{k}"] = ymin_vec
            params[f"y_max_{k}"] = ymax_vec

        # 3) 입력(액추에이터) 경계가 요구되면 olb/oub 채움 (nu=3 가정: heater, humidifier, co2)
        nu = 3
        params.setdefault("olb", np.zeros((nu, 1), dtype=float))   # 0
        params.setdefault("oub", np.ones((nu, 1), dtype=float))    # 1

        # 4) 비용/가중치/말기비용 등 (θ에서 있으면 넣고, 없으면 스킵)
        def put_vec(name, length=None, default=None):
            v = self.theta.get(name, default)
            if v is None:
                return
            arr = np.asarray(v, dtype=float).reshape(-1, 1)
            if length is not None and arr.shape[0] != length:
                # 길이 맞추기 (필요시 잘라내거나 패딩)
                arr = np.resize(arr, (length, 1))
            params[name] = arr

        # 출력/입력 가중치 (길이는 보통 ny/nu)
        put_vec("c_y", length=ny)
        put_vec("c_dy", length=ny)
        put_vec("c_u", length=nu)
        put_vec("w",   length=ny)      # 모델에 따라 해석 다름
        put_vec("V0",  length=ny)      # 초기 상태 가중
        put_vec("y_fin", length=ny)    # 말기 목표/가중

        # 5) 동역학 파라미터 p_* 세트 (에러에 p_0..p_27가 직접 등장 → 각 키로 채움)
        #    θ JSON에 p_0..p_27가 있다면 그대로 집어넣기
        i = 0
        while True:
            key = f"p_{i}"
            if key not in self.theta:
                break
            params[key] = float(self.theta[key])
            i += 1

        return params

# =========================
# ====== 메인 루프  =======
# =========================

def control_loop(client: mqtt.Client, mpc_runner: MpcRunner):
    """CONTROL_PERIOD_SEC 주기로 상태를 읽고 MPC 계산→퍼블리시."""
    print("[Main] 제어 루프 시작")
    while True:
        x = build_state_vector()
        if x is None:
            print("[Main] Waiting sensor values... required keys:", STATE_KEYS)
            time.sleep(1)
            continue

        cmd = mpc_runner.step(x)
        publish_actuators(client, cmd)
        print(f"[Main] x={x} → cmd={cmd}")
        time.sleep(CONTROL_PERIOD_SEC)

def main():
    # MQTT 클라이언트 설정
    client = mqtt.Client(client_id=CLIENT_ID, protocol=mqtt.MQTTv5)
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
