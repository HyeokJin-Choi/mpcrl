# nohup python -u q_learning_greenhouse.py > run.log 2>&1 & 
# disown
# 위 두 명령어로 실행시킬 것.

# ps -ef | grep q_learning_greenhouse.py 프로세스 확인
# tail -f run.log 로그 파일 모니터링
# top -p 13719 CPU/RAM 사용 확인, 숫자는 배정된 PID 번호를 사용할 것.

# grep "\[CKPT\] Saved at episode" -n run.log | tail 지금 몇 에피소드까지 끝났는지 확인

# nohup python -u q_learning_greenhouse.py > run.log 2>&1 & 
# disown
# 위 두 명령어로 실행시킬 것.

import os, time, signal
import importlib
import logging
import pickle
import sys
from typing import Literal

import casadi as cs
import numpy as np
from gymnasium.wrappers import NormalizeReward, TimeLimit

from mpcrl import LearnableParameter, LearnableParametersDict
from mpcrl.wrappers.agents import Evaluate, Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

from agents.greenhouse_agent import GreenhouseLearningAgent
from greenhouse.env import LettuceGreenHouse
from greenhouse.model import Model
from mpcs.learning import LearningMpc
from utils.plot import plot_greenhouse


# --------------------------------------------------------
# ---------- Utils: safe array shaping & extraction ----------
import numpy as np

def squeeze_last_if_one(a):
    """
    마지막 축의 크기가 1이면 squeeze. 에피소드별 길이가 다른 ragged 리스트/배열(object dtype)도 안전 처리.
    """
    a = np.asarray(a, dtype=object)
    if a.dtype == object:  # 에피소드 리스트 형태
        out = []
        for ep in a:
            ep_arr = np.asarray(ep)
            if ep_arr.ndim >= 1 and ep_arr.shape[-1] == 1:
                ep_arr = np.squeeze(ep_arr, axis=-1)
            out.append(ep_arr)
        return np.array(out, dtype=object)
    # 일반 ndarray
    if a.ndim >= 1 and a.shape[-1] == 1:
        return np.squeeze(a, axis=-1)
    return a


def stack_disturbances(env):
    """
    MonitorEpisodes 래퍼에 저장된 전체 에피소드의 외란 프로파일을 object 배열로 반환.
    없으면 빈 object 배열.
    """
    try:
        d = env.get_wrapper_attr('disturbance_profiles_all_episodes')
    except Exception:
        d = None
    if d is None:
        return np.array([], dtype=object)
    return np.asarray(d, dtype=object)


def build_TD_matrix(td_errors, R_tr, num_episodes):
    """
    TD 에러를 에피소드 경계에 맞춰 2D(object->ragged safe)로 재배열.
    R_tr의 에피소드 길이를 기준으로 잘라낸다.
    """
    ep_lens = [len(r) for r in R_tr]
    flat = np.asarray(td_errors, dtype=float).ravel()
    total = sum(ep_lens)
    flat = flat[:total] if total > 0 else flat[:0]
    if len(ep_lens) > 1 and total > 0:
        splits = np.cumsum(ep_lens[:-1])
        per_ep = np.split(flat, splits)
    elif total > 0:
        per_ep = [flat]
    else:
        per_ep = []
    # 길이 정규화(최솟값 기준) → 시각화 코드가 stack 가능한 모양을 기대함
    if not per_ep:
        return np.empty((0, 0))
    min_len = min(len(v) for v in per_ep)
    if min_len == 0:
        return np.empty((0, 0))
    return np.stack([v[:min_len] for v in per_ep], axis=0)

def _stack_ragged(obj_arr, target_len=None):
    """
    에피소드별 길이가 다른 시퀀스(2D/3D)를 (episodes, T, ...)로 스택합니다.
    target_len이 주어지면 앞에서부터 그 길이에 맞춰 잘라 스택합니다.
    """
    arrs = [np.asarray(ep) for ep in obj_arr]
    lens = [a.shape[0] for a in arrs] if arrs else [0]
    if not arrs or min(lens) == 0:
        return np.empty((0, 0))
    T = min(lens) if target_len is None else min(target_len, min(lens))
    return np.stack([a[:T] for a in arrs], axis=0)

# ------------------------------------------------------------

# (선택) 평가 저장 분기에서 NameError 방지용 기본값
X_ev = None
U_ev = None
R_ev = None
d_ev = None

# --------------------------------------------------------

STORE_DATA = True
PLOT = True

# 0) Config 로딩
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    mod = importlib.import_module(f"sims.configs.{config_file}")
    test = mod.Test()
else:
    from sims.configs.test_80 import Test  # type: ignore
    test = Test()

# 1) 공통 경로/디렉터리
os.makedirs("results", exist_ok=True)
CKPT_PATH = f"results/{test.test_ID}_ckpt.pkl"   # 예: results/test_80_ckpt.pkl

# ---------- 체크포인트 유틸 ----------
def agent_param_values(agent):
    """에이전트의 learnable 파라미터 현재 값 사전으로 추출"""
    out = {}
    for name, lp in agent.learnable_parameters.items():
        out[name] = np.array(lp.value, dtype=float)
    return out

def load_agent_param_values(agent, values: dict):
    """저장된 파라미터 값을 에이전트에 주입"""
    for name, val in values.items():
        if name in agent.learnable_parameters:
            agent.learnable_parameters[name].value = np.array(val, dtype=float)

def save_ckpt(ep_idx: int, agent, meta=None):
    data = {
        "episode_index": ep_idx,
        "test_id": test.test_ID,
        "params": agent_param_values(agent),
        "meta": meta or {"time": time.time()},
    }
    with open(CKPT_PATH, "wb") as f:
        pickle.dump(data, f)

def load_ckpt_if_exists(agent):
    if not os.path.exists(CKPT_PATH):
        return None
    with open(CKPT_PATH, "rb") as f:
        data = pickle.load(f)
    if data.get("test_id") != test.test_ID:
        return None  # 다른 실험의 ckpt는 무시
    load_agent_param_values(agent, data.get("params", {}))
    return data
# -------------------------------------

np_random = np.random.default_rng(test.seed if getattr(test, "model_parameters_from_seed", False) else 1)

episode_len = test.ep_len
train_env = MonitorEpisodes(
    TimeLimit(
        LettuceGreenHouse(
            growing_days=test.num_days,
            model_type=test.base_model,
            cost_parameters_dict=test.rl_cost,
            disturbance_profiles_type=test.disturbance_type,
            noisy_disturbance=test.noisy_disturbance,
            clip_action_variation=test.clip_action_variation,
        ),
        max_episode_steps=int(episode_len),
    )
)
if test.normalize_reward:
    train_env = NormalizeReward(train_env, test.discount_factor)

eval_env = MonitorEpisodes(
    TimeLimit(
        LettuceGreenHouse(
            growing_days=test.num_days,
            model_type=test.base_model,
            cost_parameters_dict=test.rl_cost,
            disturbance_profiles_type=test.disturbance_type,
            noisy_disturbance=test.noisy_disturbance,
            clip_action_variation=test.clip_action_variation,
        ),
        max_episode_steps=int(episode_len),
    )
)

prediction_model: Literal["euler", "rk4"] = "rk4"
mpc = LearningMpc(
    greenhouse_env=train_env,
    test=test,
    prediction_model=prediction_model,
    np_random=np_random,
    constrain_control_rate=True,
)

param_bounds = Model.get_learnable_parameter_bounds()
param_bounds.update(test.learn_bounds)

learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(
            name,
            val.shape,
            val,
            sym=mpc.parameters[name],
            lb=param_bounds[name][0] if name in param_bounds.keys() else -np.inf,
            ub=param_bounds[name][1] if name in param_bounds.keys() else np.inf,
        )
        for name, val in mpc.learnable_pars_init.items()
    )
)

agent = Evaluate(
    Log(
        RecordUpdates(
            GreenhouseLearningAgent(
                mpc=mpc,
                update_strategy=test.update_strategy,
                discount_factor=mpc.discount_factor,
                optimizer=test.optimizer,
                learnable_parameters=learnable_pars,
                fixed_parameters=mpc.fixed_pars,
                exploration=test.exploration,
                experience=test.experience,
                hessian_type=test.hessian_type,
                record_td_errors=True,
            )
        ),
        level=logging.DEBUG,
        log_frequencies={"on_timestep_end": 1},
        to_file=True,
        log_name=f"log_{test.test_ID}",
    ),
    eval_env,
    hook="on_episode_end",
    # ★ 짧은 러닝에서도 최소 1회 평가가 돌도록
    frequency=1,
    eval_immediately=True,
    deterministic=True,
    raises=False,
    env_reset_options={
        "initial_day": test.initial_day,
        "noise_coeff": test.noise_coeff if test.noisy_disturbance else 1.0,
    } if test.disturbance_type == "single" else {},
    seed=test.seed,
)

# ---------- 안전 종료용 시그널 핸들러 ----------
current_ep_for_sig = -1
def _save_and_exit(signum, frame):
    try:
        if 'agent' in globals():
            save_ckpt(current_ep_for_sig, agent, meta={"signal": int(signum), "time": time.time()})
    finally:
        os._exit(1)

for sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(sig, _save_and_exit)
    except Exception:
        pass
# -----------------------------------------------

# ---------- Resume ----------
resume_info = load_ckpt_if_exists(agent)
start_ep = 0
if resume_info is not None:
    start_ep = int(resume_info.get("episode_index", -1)) + 1
    print(f"[CKPT] Resuming from episode {start_ep} with params restored.")
# ---------------------------

# ---------- 학습 루프(에피소드 1개씩) ----------
for ep in range(start_ep, test.num_episodes):
    current_ep_for_sig = ep
    agent.train(
        env=train_env,
        episodes=1,  # 한 번에 1개
        seed=test.seed,
        raises=False,
        env_reset_options={
            "initial_day": test.initial_day,
            "noise_coeff": test.noise_coeff if test.noisy_disturbance else 1.0,
        } if test.disturbance_type == "single" else {},
    )
    save_ckpt(ep, agent, meta={"after_episode": ep, "time": time.time()})
    print(f"[CKPT] Saved at episode {ep}")

# 모든 에피소드 완료 → ckpt 정리(선택) 2025/09/29 주석처리
# try:
#     if os.path.exists(CKPT_PATH):
#         os.remove(CKPT_PATH)
#         print("[CKPT] Completed. Checkpoint removed.")
# except Exception:
#     pass
# -----------------------------------------------

print(np.mean(agent.solve_times))

# ---------- 데이터 추출 ----------

# Train 데이터 (먼저 꺼내서 에피소드별 원시 시퀀스 확보: ragged/object)
X_tr = np.asarray(train_env.observations, dtype=object)   # 각 ep: (T_x_plus1, nx)
U_tr = squeeze_last_if_one(train_env.actions)             # 각 ep: (T_u, nu)
R_tr = np.asarray(train_env.rewards, dtype=object)        # 각 ep: (T_r,)
d_tr = stack_disturbances(train_env)                      # 각 ep: (T_d, nd)

# TD를 에피소드 경계 기준으로 안전하게 재구성
TD = build_TD_matrix(agent.td_errors, R_tr, test.num_episodes)

# 파라미터 업데이트 히스토리 reshape도 안전하게 (에피소드 경계 기준)
param_dict = {}
for key, val in agent.updates_history.items():
    temp = [val[0]] * test.skip_first
    val = [*temp, *val[1:]]
    flat = np.asarray(val, dtype=float).ravel()
    ep_lens = [len(r) for r in R_tr]
    total = sum(ep_lens)
    flat = flat[:total]
    if len(ep_lens) > 1:
        splits = np.cumsum(ep_lens[:-1])
        per_ep = np.split(flat, splits)
    else:
        per_ep = [flat]
    min_len = min(len(v) for v in per_ep) if per_ep else 0
    if min_len == 0:
        param_dict[key] = np.empty((0, 0))
    else:
        param_dict[key] = np.stack([v[:min_len] for v in per_ep], axis=0)

# ---------- 길이 정렬 & 3D 스택 ----------
# 각 시퀀스의 최소 길이 계산
Tx = min(len(np.asarray(ep)) for ep in X_tr) if len(X_tr) else 0      # 상태 길이 (T_x_plus1)
Tu = min(len(np.asarray(ep)) for ep in U_tr) if len(U_tr) else 0      # 입력 길이 (T_u)
Tr = min(len(np.asarray(ep)) for ep in R_tr) if len(R_tr) else 0      # 보상 길이 (T_r)
Td = min(len(np.asarray(ep)) for ep in d_tr) if len(d_tr) else 0      # 외란 길이 (T_d)

# 공통 시간축 T 설정: X는 (T+1), U/R/d는 (T)
T = max(0, min(Tu, Tr, Td, Tx - 1))

# T가 0이면(데이터가 비어 있으면) 플롯은 건너뛰고 저장만
if T <= 0:
    print("[WARN] No timesteps to plot (T<=0). Skipping plots.")
    PLOT = False


# 정렬된 3D/2D 텐서로 변환
X = _stack_ragged(X_tr, target_len=T + 1)   # (episodes, T+1, nx?) or (episodes, T+1)
U = _stack_ragged(U_tr, target_len=T)       # (episodes, T,   nu?) or (episodes, T)
R = _stack_ragged(R_tr, target_len=T)       # (episodes, T)
d = _stack_ragged(d_tr, target_len=T)       # (episodes, T,   nd?) or (episodes, T)

# 🔧 차원 보정: 2D로 떨어졌으면 마지막 축을 추가해 3D로 맞춤
if X.ndim == 2: X = X[:, :, np.newaxis]
if U.ndim == 2: U = U[:, :, np.newaxis]
if d.ndim == 2: d = d[:, :, np.newaxis]


# (선택) 방어적 체크
# assert X.ndim == 3 and U.ndim == 3 and d.ndim == 3, f"Shapes not 3D: {X.shape}, {U.shape}, {d.shape}"
# assert X.shape[1] == U.shape[1] + 1 == d.shape[1] + 1 == R.shape[1] + 1, "Time lengths misaligned"

# ---------- 플롯 ----------
if PLOT:
    # ※ ragged 원본(X_tr 등)이 아니라, 정렬된 텐서(X/U/R/d)를 넘깁니다.
    plot_greenhouse(X, U, d, R, TD)
# -------------------------

# ---------- 저장 ----------
identifier_tr = f"results/{test.test_ID}_train"
identifier_ev = f"results/{test.test_ID}_eval"

if STORE_DATA:
    with open(f"{identifier_tr}.pkl", "wb") as file:
        pickle.dump(
            {
                "name": identifier_tr,
                "X": X,           # 정렬된 텐서 저장
                "U": U,
                "R": R,
                "d": d,
                "TD": TD,
                "param_dict": param_dict,
            },
            file,
        )
    if X_ev is not None:
        with open(f"{identifier_ev}.pkl", "wb") as file:
            pickle.dump(
                {"name": identifier_ev, "X": X_ev, "U": U_ev, "R": R_ev, "d": d_ev},
                file,
            )
# -------------------------
