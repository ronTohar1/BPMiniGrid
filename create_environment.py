import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from minigrid_wrappers import ObjectsLocationWrapper, OnlyImageObservation
from bp_gym import BPGymEnv
from stable_baselines3.common.monitor import Monitor
from minigrid.wrappers import ImgObsWrapper


def create_environment(env_name="MiniGrid-BlockedUnlockPickup-v0", render_mode=None, add_strategies=True, vanilla=False):
    # env_name = "MiniGrid-DoorKey-8x8-v0"
    env = gym.make(env_name, render_mode=render_mode, max_episode_steps=5000)
    
    if not render_mode:
        env = gym.make(env_name)
    
    if vanilla:
        return env
    
    
    env = FullyObsWrapper(env)
    env = ObjectsLocationWrapper(env)
    # env = OnlyImageObservation(env)
    env = ImgObsWrapper(env)
    env = BPGymEnv(env, add_strategies=add_strategies, as_image=True, axis=2)
    
    env = Monitor(env)
    return env