from stable_baselines3.common.env_util import make_vec_env
import argparse
import minigrid
from stable_baselines3 import PPO, DQN, A2C
from minigrid_feature_extractor import MinigridFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from gymnasium.wrappers.flatten_observation import FlattenObservation
import numpy as np

def create_env():
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    # env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="human")
    # env = FlattenObservation(env)
    env = ImgObsWrapper(env)
    return env

def train():
    env_name = "MiniGrid-DoorKey-5x5-v0"


    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        # net_arch=[128,128],
    )

    seed = np.random.randint(0, 2**31 - 1)
    seed2 = np.random.randint(0, 2**31 - 1)
    env = make_vec_env(create_env, n_envs=6, seed=seed)
    eval_env = make_vec_env(create_env, n_envs=1, seed=seed2)

    # env = create_env()
    # eval_env = create_env()

    eval_callback = EvalCallback(eval_env, best_model_save_path="./models/",
                                log_path="./logs/", eval_freq=100,               
                                deterministic=True, render=False, n_eval_episodes=10)
    
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./tensorboard/", policy_kwargs=policy_kwargs, )
    model.learn(total_timesteps=100_000, tb_log_name="dqn_5x5", callback=eval_callback)

if __name__ == "__main__":
    train()