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
        theta_params.json 에 있는 파라미터를 MPC에 바인딩하기 위한
        전처리만 수행합니다. (실제 주입은 _build_params_for_solve 에서)
        초기화 시점에 self.mpc에 직접 set하지 않아 경고 스팸을 방지합니다.
        """
        def as1d(key):
            v = theta.get(key)
            if v is None:
                return None
            return list(v) if isinstance(v, (list, tuple)) else [v]
    
        # 필요시 여기서 값의 스케일/형태만 점검하고,
        # solve 단계에서 사용할 self.theta 그대로 유지합니다.
        # 예: as1d("c_y"); as1d("c_u"); 등으로 유효성만 미리 확인 가능
        for key in ["V0", "c_dy", "c_y", "y_fin", "c_u", "w", "olb", "oub"]:
            _ = as1d(key)
        # p_0 ~ p_N 같은 연속 파라미터도 존재만 확인
        i = 0
        while True:
            k = f"p_{i}"
            if k not in theta:
                break
            _ = theta[k]  # 사용 여부만 확인
            i += 1

    def safe_fallback(self, x):
        # x = [온도, 습도, CO2ppm, 광량] 가정
        T, RH, CO2, PAR = x
        cmd = dict(heater=0.0, humidifier=0.0, co2_valve=0.0, led=0.0)
    
        if T < 18.0: cmd["heater"] = 0.3
        if RH < 40.0: cmd["humidifier"] = 0.2
        if CO2 < 450.0: cmd["co2_valve"] = 0.1
        if PAR < 200.0: cmd["led"] = 0.2

    return cmd
        
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
        # u = self.mpc.solve(params)
        try:
            u = self.mpc.solve(x, params=self._build_params_for_solve(x))
        except Exception as e:
            # ❌ 잘못된 예: logger.error("... %s", params)
            # ✅ 이렇게 바꾸세요:
            logger.exception("[MPC] Solve failed, using fallback. reason=%s", e)
            u = self.safe_fallback(x)  # 최소 안전동작
        
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

    def _fallback_policy(self, x: list) -> list:
        """MPC 실패 시 간단한 P제어 대체(안전모드)."""
        setpoints = [sum(Y_BOUNDS[k]) / 2.0 for k in STATE_KEYS]
        kp = [0.1, 0.1, 0.05]  # ★ 3채널만
        u = []
        for i in range(min(len(kp), 3)):  # ★ 0..2까지만
            u.append(kp[i] * (setpoints[i] - float(x[i])))
        while len(u) < 3:
            u.append(0.0)
        return u

        
    def _build_params_for_solve(self, x):
        import numpy as np
    
        nx = ny = nd = 4
        nu = 3
        N  = int(getattr(self.mpc, "prediction_horizon", 24))
    
        P = {}
        P["x_0"] = np.asarray(x, dtype=float).reshape(nx, 1)
        P["d"]   = np.zeros((nd, N), dtype=float)
    
        y_min_vec = np.array([Y_BOUNDS[k][0] for k in STATE_KEYS], dtype=float).reshape(ny, 1)
        y_max_vec = np.array([Y_BOUNDS[k][1] for k in STATE_KEYS], dtype=float).reshape(ny, 1)
        for k in range(N + 1):
            P[f"y_min_{k}"] = y_min_vec
            P[f"y_max_{k}"] = y_max_vec
    
        def put_vec(name, length):
            if name not in self.theta:
                return
            arr = np.asarray(self.theta[name], dtype=float).reshape(-1, 1)
            if arr.shape[0] == 1:
                arr = np.full((length, 1), float(arr[0, 0]))  # 스칼라 → 길이 맞춤
            elif arr.shape[0] != length:
                arr = np.resize(arr, (length, 1))
            P[name] = arr
    
        # 출력 관련(4×1), 입력 관련(3×1)
        put_vec("c_y",   ny)
        put_vec("c_dy",  ny)
        put_vec("V0",    ny)
        put_vec("y_fin", ny)
        put_vec("w",     ny)
        put_vec("c_u",   nu)
        put_vec("olb",   nu)
        put_vec("oub",   nu)
    
        # ☆ 여기만 교체: p_0..p_n 을 4×1로 브로드캐스트해서 넣기
        i = 0
        while True:
            key = f"p_{i}"
            if key not in self.theta:
                break
            sval = float(self.theta[key])
            P[key] = np.full((ny, 1), sval, dtype=float)   # ★ 4×1로 맞춤
            i += 1
    
        # (선택) 디버그: 각 파라미터 shape 확인
        for k, v in P.items():
            try:
                print(f"[DBG] param {k}: {np.asarray(v).shape}")
            except Exception:
                pass
    
        return P



    
    def step(self, x):
        try:
            params = self._build_params_for_solve(x)
            u = self.mpc.solve(params)  # (heater, humidifier, co2_valve) 3채널
            u = np.asarray(u, float).reshape(-1).tolist()
        except Exception as e:
            print(f"[MPC] Solve failed, fallback policy used: {e}")
            u = self._fallback_policy(x)  # 길이 3
    
        # ★ LED는 임시 규칙 기반: PAR(조도) 낮으면 켬
        PAR = float(x[3])
        led_cmd = 0.2 if PAR < 200.0 else 0.0
    
        cmd = {
            "heater":     float(u[0]) if len(u)>0 else 0.0,
            "humidifier": float(u[1]) if len(u)>1 else 0.0,
            "co2_valve":  float(u[2]) if len(u)>2 else 0.0,
            "led":        led_cmd,  # ★ MPC 바깥에서 결정
        }
        for k in cmd:
            cmd[k] = 0.0 if cmd[k] < 0.0 else 1.0 if cmd[k] > 1.0 else cmd[k]
        return cmd



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
    client = mqtt.Client(
        client_id=CLIENT_ID,
        protocol=mqtt.MQTTv5,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2  # v1 폐기 경고 제거
    )
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
