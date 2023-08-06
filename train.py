import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from minigrid_feature_extractor import MinigridFeaturesExtractor
import gymnasium as gym
from bp_gym import BPGymEnv

def train():
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    env = gym.make("MiniGrid-BlockedUnlockPickup-v0")
    env = BPGymEnv(env)
    # env = ImgObsWrapper(env)

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tensorboard/")
    model.learn(2e5)

if __name__ == '__main__':
    train()