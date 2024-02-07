import numpy as np
from collections import defaultdict

smoothing_window = 20
testing_frequency = 10
consecutive_steps_for_convergence = 2000
y_tick_pad = -2
savedir = "noisy_appendix"

q_init_values = ["-10.0"]

mon_to_label = {
    "Gridworld-Easy-3x3-Stochastic-v0_mes50/iStatelessBinaryMonitor": "Simple",
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iStatelessBinaryMonitor": "Penalty",
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iToySwitchMonitor": "Button",
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iNMonitor_nm5": "N-Monitor",
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLimitedTimeMonitor": "Limited Time",
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLimitedUseMonitor_mb7": "Limited Use",
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLimitedUseBonusMonitor_mb7": "Limited Use",
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLevelMonitor_nl4": "Level Monitor",
}

mon_to_opt = {
    "Gridworld-Easy-3x3-Stochastic-v0_mes50/iStatelessBinaryMonitor": 0.99,
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iStatelessBinaryMonitor": 0.9509,
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iToySwitchMonitor": 0.552,
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iNMonitor_nm5": 0.9568,
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLimitedTimeMonitor": 0.9509,
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLimitedUseMonitor_mb7": 0.9509,
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLimitedUseBonusMonitor_mb7": 1.7019,
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLevelMonitor_nl4": 0.9509,
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

mon_to_xlim = defaultdict(lambda: 100000)
mon_to_xticks = defaultdict(lambda: np.arange(0, 100001, 20000))

mon_to_ylim = {
    "Gridworld-Easy-3x3-Stochastic-v0_mes50/iStatelessBinaryMonitor": [-0.2, 1.05],
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iStatelessBinaryMonitor": [-0.2, 1.05],
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iToySwitchMonitor": [-1.4, 1.05],
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iNMonitor_nm5": [-0.2, 1.05],
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLimitedTimeMonitor": [-0.2, 1.05],
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLimitedUseMonitor_mb7": [-1.1, 1.05],
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLimitedUseBonusMonitor_mb7": [-1.1, 2.05],
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLevelMonitor_nl4": [-2.1, 1.05],
}
mon_to_yticks = {
    "Gridworld-Easy-3x3-Stochastic-v0_mes50/iStatelessBinaryMonitor": [0.0, 0.5, 1.0],
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iStatelessBinaryMonitor": [0, 0.5, 1.0],
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iToySwitchMonitor": [-1.0, 0.0, 1.0],
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iNMonitor_nm5": [0, 0.5, 1.0],
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLimitedTimeMonitor": [0, 0.5, 1.0],
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLimitedUseMonitor_mb7": [-1.0, 0.0, 1.0],
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLimitedUseBonusMonitor_mb7": [-1.0, 0.0, 1.0, 2.0],
    "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLevelMonitor_nl4": [-2.0, -1.0, 0.0, 1.0],
}
