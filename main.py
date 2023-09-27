import gymnasium as gym
from matplotlib import pyplot as plt
from minigrid.wrappers import SymbolicObsWrapper, FullyObsWrapper
from minigrid.core.constants import OBJECT_TO_IDX
import numpy as np
from minigrid.manual_control import ManualControl
from minigrid_wrappers import ObjectsLocationWrapper, OnlyImageObservation, CountBackAndForthTurns
from bp_gym import BPGymEnv
from create_environment import create_environment
def main():

    env = gym.make("MiniGrid-BlockedUnlockPickup-v0", render_mode="human")
    env = FullyObsWrapper(env)
    # env = SymbolicObsWrapper(env)
    obs,info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        print(obs["image"].shape)
        image = obs["image"]

    env.close()


def main2():
    # env = gym.make("MiniGrid-BlockedUnlockPickup-v0", render_mode="human")
    # env = FullyObsWrapper(env)
    # env = ObjectsLocationWrapper(env, print_location=True)
    # env = OnlyImageObservation(env)
    # env = BPGymEnv(env, add_strategies=True, as_image=True, axis=2)
    # print(env.observation_space)
    # env = ManualControl(env)

    # env = create_environment(env_name="MiniGrid-KeyCorridorS3R3-v0",add_strategies=False, render_mode="human")
    # env = CountBackAndForthTurns(env)
    env = create_environment("MiniGrid-Unlock-v0", render_mode="human", add_strategies=True, partially_observable=False)
    print(env.observation_space)
    env = ManualControl(env)
    env.start()


if __name__ == '__main__':
    # main()
    main2()