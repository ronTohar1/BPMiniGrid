import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback
from minigrid_feature_extractor import MinigridFeaturesExtractor
import gymnasium as gym
from bp_gym import BPGymEnv
from create_environment import create_environment
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.env_util import make_vec_env
import argparse


def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--add_bp","-bp", action="store_true", help="Add BP strategies to the environment")
    parser.add_argument("--name_addition","--name", type=str, default="", help="Addition to the model name")
    parser.add_argument("--agent",'--a', choices=['ppo','dqn','a2c','rppo'], default='dqn', help="Choose the agent to train")
    parser.add_argument("--env_name","-env", choices=["easy","hard"], default="easy", help="Choose the environment to train on" )
    args = parser.parse_args()

    choices = {"easy":"MiniGrid-DoorKey-5x5-v0","hard":"MiniGrid-BlockedUnlockPickup-v0"}
    env_name = choices[args.env_name]
    add_strategies = args.add_bp
    num_episodes = 1_000_000
    verbose=0
    num_cpus = 4

    name = args.name_addition

    def create():
        return create_environment(add_strategies=add_strategies, env_name=env_name)
    eval_env = create()
    env = make_vec_env(create, n_envs=num_cpus, seed=0)
    # env = create()

    model = {"ppo":PPO,"dqn":DQN,"a2c":A2C,"rppo":RecurrentPPO}[args.agent]
    lr = 5e-4
    gamma = 0.99
    ls = 10000
    bs = 32
    tf=10_0000000
    explore_frac = 0.7

    model_name = model.__name__+ f'_lr{lr}_gamma{gamma}_ls{ls}_bs{bs}_steps{num_episodes/1000000}_tf{tf}_expfrac{explore_frac}'
    model_name += name
    model_name += "_BP" if add_strategies else "_NOBP"

    policy = "CnnPolicy" if args.agent != "rppo" else "CnnLstmPolicy"
    model_path = "./models/" + model_name
    log_path = "./logs/"
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_path,
                                log_path=log_path, eval_freq=1000,
                                deterministic=True, render=False, n_eval_episodes=10)


    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )
    # agent = DQN(policy, env, verbose=verbose, tensorboard_log="./tensorboard/",
    #             policy_kwargs=policy_kwargs, exploration_fraction=explore_frac,
    #             learning_rate=lr,
    #             buffer_size=15000,
    #             learning_starts=ls,
    #             batch_size=bs,
    #             gamma=gamma,
    #             train_freq=tf,
    #             gradient_steps=1,
    #             target_update_interval=50,)

    
    agent = model(policy, env, verbose=verbose, tensorboard_log=f"./tensorboard/{env_name}", policy_kwargs=policy_kwargs)
    agent.learn(num_episodes, callback=eval_callback, log_interval=100, tb_log_name=model_name)

if __name__ == '__main__':
    train()