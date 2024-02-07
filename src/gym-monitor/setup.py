# fmt: off
from setuptools import setup

packages = ["gym_monitor"]
install_requires = [
    "gymnasium",
    "pygame"
]
entry_points = {
    "gymnasium.envs": ["Gym-Monitor=gym_monitor.gym:register_envs"]
}

setup(
    name="Gym-Monitor",
    version="0.0.1",
    license="GPL",
    packages=packages,
    entry_points=entry_points,
    install_requires=install_requires,
)
