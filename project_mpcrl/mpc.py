import json
from learning import LearningMpc
import time

# ✅ 최종 학습 파라미터 로드
with open('theta_params.json', 'r') as f:
    theta_data = json.load(f)
theta = theta_data['params']

# MPC 초기화
mpc = LearningMpc(
    greenhouse_env=None,
    test=None,
    prediction_model="rk4",
    np_random=None,
    constrain_control_rate=True,
)

# ✅ 파라미터 적용
mpc.set_learned_parameters(theta)

# 제어 루프
while True:
    x = read_sensors()          # ESP32 센서 데이터
    u = mpc.solve(x)            # MPC 최적 제어 계산
    control_actuators(u)        # 액추에이터로 전달
    time.sleep(60)              # 1분 단위 제어 주기

