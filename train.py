import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback
from minigrid_feature_extractor import MinigridFeaturesExtractor
import gymnasium as gym
from bp_gym import BPGymEnv
from create_environment import create_environment


def train():

    add_strategies = False
    num_episodes = 300_000
    verbose=0


    env = create_environment(add_strategies=add_strategies)
    eval_env = create_environment(add_strategies=add_strategies)


    lr = 5e-4
    gamma = 0.99
    ls = 10000
    bs = 32
    model = DQN
    tf=10
    model_name = model.__name__+ f'_lr{lr}_gamma{gamma}_ls{ls}_bs{bs}_steps{num_episodes/1000000}_tf{tf}'

    policy = "CnnPolicy"
    model_path = "./models/" + model_name
    log_path = "./logs/"
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_path,
                                log_path=log_path, eval_freq=1000,
                                deterministic=True, render=False, n_eval_episodes=10)


    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    model = DQN(policy, env, verbose=verbose, tensorboard_log="./tensorboard/",
                policy_kwargs=policy_kwargs,
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=1000,
                batch_size=32,
                gamma=0.99,
                train_freq=tf,
                gradient_steps=1,
                target_update_interval=50,)
    
    model.learn(num_episodes, callback=eval_callback, log_interval=100, tb_log_name=model_name)

if __name__ == '__main__':
    train()