import numpy as np
from collections import defaultdict

smoothing_window = 5
testing_frequency = 10
consecutive_steps_for_convergence = 200
y_tick_pad = -2
savedir = "deterministic_appendix"

q_init_values = ["-10.0"]

mon_to_label = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": "Simple",
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": "Penalty",
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": "Button",
    "Gridworld-Medium-3x3-v0_mes50/iNMonitor_nm5": "N-Monitor",
    "Gridworld-Medium-3x3-v0_mes50/iLimitedTimeMonitor": "Limited Time",
    "Gridworld-Medium-3x3-v0_mes50/iLimitedUseMonitor_mb7": "Limited Use",
    "Gridworld-Medium-3x3-v0_mes50/iLimitedUseBonusMonitor_mb7": "Limited Use",
    "Gridworld-Medium-3x3-v0_mes50/iLevelMonitor_nl4": "Level Monitor",
}

mon_to_opt = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": 0.99,
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": 0.9509,
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": 0.552,
    "Gridworld-Medium-3x3-v0_mes50/iNMonitor_nm5": 0.9568,
    "Gridworld-Medium-3x3-v0_mes50/iLimitedTimeMonitor": 0.9509,
    "Gridworld-Medium-3x3-v0_mes50/iLimitedUseMonitor_mb7": 0.9509,
    "Gridworld-Medium-3x3-v0_mes50/iLimitedUseBonusMonitor_mb7": 1.7019,
    "Gridworld-Medium-3x3-v0_mes50/iLevelMonitor_nl4": 0.9509,
}

alg_to_label = {
    "oracle_with_reward_model": "Oracle",
    # 'oracle': 'Oracle',
    "reward_model": "Rew. Model",
    "q_sequential": "Sequential",
    "q_joint": "Joint",
    "ignore": r"Ignore $\bot$",
    "reward_is_0.": r"$\bot$ = 0",
}

mon_to_xlim = defaultdict(lambda: 10000)
mon_to_xticks = defaultdict(lambda: np.arange(0, 10001, 2000))
mon_to_xlim = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": 500,
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": 2000,
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": 2000,
    "Gridworld-Medium-3x3-v0_mes50/iNMonitor_nm5": 5000,
    "Gridworld-Medium-3x3-v0_mes50/iLimitedTimeMonitor": 10000,
    "Gridworld-Medium-3x3-v0_mes50/iLimitedUseMonitor_mb7": 10000,
    "Gridworld-Medium-3x3-v0_mes50/iLimitedUseBonusMonitor_mb7": 10000,
    "Gridworld-Medium-3x3-v0_mes50/iLevelMonitor_nl4": 10000,
}
mon_to_xticks = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": [0, 250, 500],
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": np.arange(0, 2001, 500),
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": np.arange(0, 2001, 500),
    "Gridworld-Medium-3x3-v0_mes50/iNMonitor_nm5": np.arange(0, 5001, 1000),
    "Gridworld-Medium-3x3-v0_mes50/iLimitedTimeMonitor": np.arange(0, 10001, 2000),
    "Gridworld-Medium-3x3-v0_mes50/iLimitedUseMonitor_mb7": np.arange(0, 10001, 2000),
    "Gridworld-Medium-3x3-v0_mes50/iLimitedUseBonusMonitor_mb7": np.arange(0, 10001, 2000),
    "Gridworld-Medium-3x3-v0_mes50/iLevelMonitor_nl4": np.arange(0, 10001, 2000),
}

mon_to_ylim = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": [-0.1, 1.05],
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": [-0.1, 1.05],
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": [-3.1, 1.05],
    "Gridworld-Medium-3x3-v0_mes50/iNMonitor_nm5": [-0.1, 1.05],
    "Gridworld-Medium-3x3-v0_mes50/iLimitedTimeMonitor": [-0.1, 1.05],
    "Gridworld-Medium-3x3-v0_mes50/iLimitedUseMonitor_mb7": [-2.1, 1.05],
    "Gridworld-Medium-3x3-v0_mes50/iLimitedUseBonusMonitor_mb7": [-2.0, 2.05],
    "Gridworld-Medium-3x3-v0_mes50/iLevelMonitor_nl4": [-2.1, 1.05],
}
mon_to_yticks = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": [0, 0.5, 1.0],
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": [0, 0.5, 1.0],
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0],
    "Gridworld-Medium-3x3-v0_mes50/iNMonitor_nm5": [0, 0.5, 1.0],
    "Gridworld-Medium-3x3-v0_mes50/iLimitedTimeMonitor": [0, 0.5, 1.0],
    "Gridworld-Medium-3x3-v0_mes50/iLimitedUseMonitor_mb7": [-2.0, -1.0, 0.0, 1.0],
    "Gridworld-Medium-3x3-v0_mes50/iLimitedUseBonusMonitor_mb7": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "Gridworld-Medium-3x3-v0_mes50/iLevelMonitor_nl4": [-2.0, -1.0, 0.0, 1.0],
}
