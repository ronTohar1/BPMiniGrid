import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from minigrid_wrappers import ObjectsLocationWrapper, OnlyImageObservation
from bp_gym import BPGymEnv
from stable_baselines3.common.monitor import Monitor
from minigrid.wrappers import ImgObsWrapper

from gymnasium.wrappers.frame_stack import FrameStack

def create_environment(env_name="MiniGrid-BlockedUnlockPickup-v0", render_mode=None, add_strategies=True, stack_frames=None,):
    # env_name = "MiniGrid-DoorKey-8x8-v0"
    env = gym.make(env_name, render_mode=render_mode, max_episode_steps=500)
    
    if not render_mode:
        env = gym.make(env_name)
    
    env = FullyObsWrapper(env)
    env = ObjectsLocationWrapper(env)
    # env = OnlyImageObservation(env)
    env = ImgObsWrapper(env)
    env = BPGymEnv(env, add_strategies=add_strategies, as_image=True, axis=2)
    
    if stack_frames:
        env = FrameStack(env, stack_frames)
        
    env = Monitor(env)
    return env