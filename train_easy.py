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
    parser.add_argument("--name_addition","-name", type=str, default="", help="Addition to the model name")
    parser.add_argument("--agent",'--a', choices=['ppo','dqn','a2c','rppo'], default='ppo', help="Choose the agent to train")
    args = parser.parse_args()


    add_strategies = args.add_bp
    num_episodes = 500_000
    verbose=0
    num_cpus = 6
    seed = 0
    name = args.name_addition

    def create():
        return create_environment(add_strategies=add_strategies, env_name="MiniGrid-DoorKey-8x8-v0")
    eval_env = create()

    env = make_vec_env(create, n_envs=num_cpus, seed=50)
    # eval_env = make_vec_env(create, n_envs=1,seed=50)

    lr = 0.000001
    gamma = 0.99
    ls = 100000
    bs = 32
    model = {"ppo":PPO,"dqn":DQN,"a2c":A2C,"rppo":RecurrentPPO}[args.agent]
    tf=10
    explore_frac = 0.4
    model_name = model.__name__
    # model_name += f'_lr{lr}_gamma{gamma}_ls{ls}_bs{bs}_steps{num_episodes/1000000}_tf{tf}_expfrac{explore_frac}'
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
        net_arch=[512,512],
    )
    # agent = DQN(policy, env, verbose=verbose, tensorboard_log="./tensorboard/",
    #             policy_kwargs=policy_kwargs, exploration_fraction=explore_frac,
    #             learning_rate=lr,
    #             buffer_size=100000,
    #             learning_starts=ls,
    #             batch_size=bs,
    #             gamma=gamma,
    #             train_freq=tf,
    #             gradient_steps=1,
    #             target_update_interval=500,)


    agent = model(policy, env, verbose=verbose, tensorboard_log="./tensorboard/", policy_kwargs=policy_kwargs, seed=seed) 
    agent.learn(num_episodes, callback=eval_callback, log_interval=1, tb_log_name=model_name)

if __name__ == '__main__':
    train()