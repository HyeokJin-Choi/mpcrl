# nohup python -u q_learning_greenhouse.py > run.log 2>&1 & 
# disown
# 위 두 명령어로 실행시킬 것.

# ps -ef | grep q_learning_greenhouse.py 프로세스 확인
# tail -f run.log 로그 파일 모니터링
# top -p 13719 CPU/RAM 사용 확인, 숫자는 배정된 PID 번호를 사용할 것.

# grep "\[CKPT\] Saved at episode" -n run.log | tail 지금 몇 에피소드까지 끝났는지 확인

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

# test_80.py 파일의 내용을 확인할 것.

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

# 모든 에피소드 완료 → ckpt 정리(선택)
try:
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
        print("[CKPT] Completed. Checkpoint removed.")
except Exception:
    pass
# -----------------------------------------------

print(np.mean(agent.solve_times))

# ---------- 데이터 추출 ----------
TD = agent.td_errors
TD = np.asarray(TD).reshape(test.num_episodes, -1)

param_dict = {}
for key, val in agent.updates_history.items():
    temp = [val[0]] * test.skip_first  # 처음 skip_first 만큼 첫 값 반복
    val = [*temp, *val[1:]]           # index 1부터는 업데이트 반영
    param_dict[key] = np.asarray(val).reshape(test.num_episodes, -1)

def squeeze_last_if_one(a):
    a = np.asarray(a)
    if a.ndim >= 1 and a.shape[-1] == 1:
        return np.squeeze(a, axis=-1)
    return a

def stack_disturbances(env):
    d = env.get_wrapper_attr('disturbance_profiles_all_episodes')
    # 빈 데이터 방어
    if d is None or (isinstance(d, list) and len(d) == 0):
        return np.empty((0, 0, 0))
    if isinstance(d, list):
        arrs = []
        for a in d:
            a = np.asarray(a)
            if a.ndim == 1:
                a = a[:, None]           # (steps,) -> (steps,1)
            # (features, steps) 형태면 (steps, features)로
            if a.ndim == 2 and a.shape[0] < a.shape[1]:
                a = a.T
            arrs.append(a)
        if len(arrs) == 0:
            return np.empty((0, 0, 0))
        # 에피소드 간 길이 다르면 최소 길이에 맞춰 자르기
        min_len = min(a.shape[0] for a in arrs)
        feat_dim = min(a.shape[1] for a in arrs)
        arrs = [a[:min_len, :feat_dim] for a in arrs]
        return np.stack(arrs, axis=0)  # (episodes, steps, features)
    d = np.asarray(d)
    if d.ndim == 3:
        # (episodes, features, steps) → (episodes, steps, features)
        if d.shape[1] < d.shape[2]:
            return np.transpose(d, (0, 2, 1))
        return d
    if d.ndim == 2:
        return d[None, ...]
    return np.empty((0, 0, 0))
# ----------------------------------

# Train 데이터
X_tr = np.asarray(train_env.observations)
U_tr = squeeze_last_if_one(train_env.actions)
R_tr = np.asarray(train_env.rewards)
d_tr = stack_disturbances(train_env)

# Eval 데이터 (없을 수 있음 → 조건부)
def has_episodes(env):
    obs = np.asarray(env.observations, dtype=object)
    return obs.size > 0

if has_episodes(eval_env):
    X_ev = np.asarray(eval_env.observations)
    U_ev = squeeze_last_if_one(eval_env.actions)
    R_ev = np.asarray(eval_env.rewards)
    d_ev = stack_disturbances(eval_env)
else:
    X_ev = U_ev = R_ev = d_ev = None

# ---------- 플롯 ----------
if PLOT:
    plot_greenhouse(X_tr, U_tr, d_tr, R_tr, TD)
# -------------------------

# ---------- 저장 ----------
identifier_tr = f"results/{test.test_ID}_train"
identifier_ev = f"results/{test.test_ID}_eval"

if STORE_DATA:
    with open(f"{identifier_tr}.pkl", "wb") as file:
        pickle.dump(
            {
                "name": identifier_tr,
                "X": X_tr,
                "U": U_tr,
                "R": R_tr,
                "d": d_tr,
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
