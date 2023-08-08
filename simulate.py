# Simulate a run of the game

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, DQN, A2C

import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper

def create_env():
    # env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
    env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="human")
    # env = FlattenObservation(env)
    env = ImgObsWrapper(env)
    return env

def simulate():

    model = PPO.load("./models/best_model.zip")
    # env = create_env()
    env = make_vec_env(create_env, n_envs=1, seed=0)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
        print("step",i)
        if dones:
            obs = env.reset()

if __name__ == "__main__":
    simulate()