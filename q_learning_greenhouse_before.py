import importlib
import logging
import pickle
import sys
from typing import Literal

import casadi as cs
import numpy as np
from gymnasium.wrappers import NormalizeReward, TimeLimit

# import networkx as netx
from mpcrl import LearnableParameter, LearnableParametersDict
from mpcrl.wrappers.agents import Evaluate, Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

from agents.greenhouse_agent import GreenhouseLearningAgent
from greenhouse.env import LettuceGreenHouse
from greenhouse.model import Model
from mpcs.learning import LearningMpc
from utils.plot import plot_greenhouse

# 디버깅중 -------
# import warnings
# from mpcrl.core.errors import MpcSolverWarning
# warnings.simplefilter("error", MpcSolverWarning)
# ---------------

STORE_DATA = True
PLOT = True

# if a config file passed on command line, otherwise use default config file
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    mod = importlib.import_module(f"sims.configs.{config_file}")
    test = mod.Test()
else:
    from sims.configs.test_80 import Test  # type: ignore

    test = Test()
    #test.num_episodes = 3       # 에피소드 3개로 축소
    #test.ep_len = 1000          # (옵션) 에피소드 길이 단축
    #test.num_days = 1           # (옵션) 재배일수 단축 → 시뮬 시간도 짧아짐



np_random = np.random.default_rng(test.seed if test.model_parameters_from_seed else 1)

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
param_bounds = (
    Model.get_learnable_parameter_bounds()
)  # includes bounds just on model parameters
param_bounds.update(
    test.learn_bounds
)  # ad dalso the bounds on other learnable parameters
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
    Log(  # type: ignore[var-annotated]
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
    frequency=10,  # eval once every 10 episodes 기본은 10임.
    eval_immediately=False, # 기본은 False
    deterministic=True,
    raises=False,
    env_reset_options={
        "initial_day": test.initial_day,
        "noise_coeff": test.noise_coeff if test.noisy_disturbance else 1.0,
    }
    if test.disturbance_type == "single"
    else {},
    seed=test.seed,
)
# evaluate train
agent.train(
    env=train_env,
    episodes=test.num_episodes,
    seed=test.seed,
    raises=False,
    env_reset_options={
        "initial_day": test.initial_day,
        "noise_coeff": test.noise_coeff if test.noisy_disturbance else 1.0,
    }
    if test.disturbance_type == "single"
    else {},
)

print(np.mean(agent.solve_times))

# extract data
TD = agent.td_errors
TD = np.asarray(TD).reshape(test.num_episodes, -1)
param_dict = {}
for key, val in agent.updates_history.items():
    temp = [
        val[0]
    ] * test.skip_first  # repeat the first value as first skip_first updates are not performed
    val = [*temp, *val[1:]]  # take index from 1 as first valeu is prior to any updates
    param_dict[key] = np.asarray(val).reshape(test.num_episodes, -1)

# 디버깅중 -----
def squeeze_last_if_one(a):
    a = np.asarray(a)
    if a.ndim >= 1 and a.shape[-1] == 1:
        return np.squeeze(a, axis=-1)
    return a

def stack_disturbances(env):
    d = env.get_wrapper_attr('disturbance_profiles_all_episodes')
    # d가 list인 경우(에피소드별 2D 배열):
    if isinstance(d, list):
        arrs = []
        for a in d:
            a = np.asarray(a)
            # a가 (features, steps)라면 (steps, features)로 전치
            if a.ndim == 2 and a.shape[0] < a.shape[1]:
                a = a.T
            # 혹시 1D인 경우(드물지만) 길이를 steps로 보고 특성 1로 확장
            if a.ndim == 1:
                a = a[:, None]
            arrs.append(a)

        # 에피소드마다 길이가 다르면 최소 길이에 맞춰 자르기(간단/보수적)
        min_len = min(a.shape[0] for a in arrs)
        feat_dim = min(a.shape[1] for a in arrs)
        arrs = [a[:min_len, :feat_dim] for a in arrs]

        return np.stack(arrs, axis=0)  # (episodes, steps, features)

    # 이미 넘파이 배열인 경우
    d = np.asarray(d)
    if d.ndim == 3:
        # (episodes, features, steps) → (episodes, steps, features)로 정규화
        if d.shape[1] < d.shape[2]:
            return np.transpose(d, (0, 2, 1))
        return d
    if d.ndim == 2:
        return d[None, ...]  # single-episode fallback
    return d


U_tr = squeeze_last_if_one(train_env.actions)
U_ev = squeeze_last_if_one(eval_env.actions)

# from train env
X_tr = np.asarray(train_env.observations)
# U_tr = np.asarray(train_env.actions).squeeze(-1)
R_tr = np.asarray(train_env.rewards)
# d_tr = np.asarray(train_env.disturbance_profiles_all_episodes).transpose(0, 2, 1)
d_tr = stack_disturbances(train_env)



X_ev = np.asarray(eval_env.observations)
# U_ev = np.asarray(eval_env.actions).squeeze(-1)
R_ev = np.asarray(eval_env.rewards)
# d_ev = np.asarray(eval_env.disturbance_profiles_all_episodes).transpose(0, 2, 1)
d_ev = stack_disturbances(eval_env)

# -----------------------------

if PLOT:  # plot training data
    plot_greenhouse(X_tr, U_tr, d_tr, R_tr, TD)

identifier_tr = test.test_ID + "_train"
identifier_ev = test.test_ID + "_eval"
if STORE_DATA:
    with open(
        f"{identifier_tr}.pkl",
        "wb",
    ) as file:
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
    with open(
        f"{identifier_ev}.pkl",
        "wb",
    ) as file:
        pickle.dump(
            {"name": identifier_ev, "X": X_ev, "U": U_ev, "R": R_ev, "d": d_ev}, file
        )
