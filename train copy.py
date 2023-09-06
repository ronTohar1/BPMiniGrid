import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback
from minigrid_feature_extractor import MinigridFeaturesExtractor
from minigrid_feature_extractor_3d import MinigridFeaturesExtractor3D 
import gymnasium as gym
from bp_gym import BPGymEnv
from create_environment import create_environment
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.env_util import make_vec_env
import argparse

import visualkeras


def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--add_bp","-bp", action="store_true", help="Add BP strategies to the environment")
    parser.add_argument("--name_addition","-name", type=str, default="", help="Addition to the model name")
    parser.add_argument("--agent_class",'--a', choices=['ppo','dqn','a2c','rppo'], default='ppo', help="Choose the agent to train")
    parser.add_argument("--frame_stack",'-fs', type=int, default=None, help="Number of frames to stack")
    parser.add_argument("--logdir",'-l', type=str, default="./tensorboard/", help="Directory to store the logs in")
    parser.add_argument("--num_episodes",'-n', type=int, default=3_000_000, help="Number of episodes to train for")
    parser.add_argument("--env_index",'-e', type=int, default=1, help="Index of the environment to train on")
    parser.add_argument("--learning_rate",'-lr', type=float, default=0.0001, help="Learning rate of the agent")
    parser.add_argument("--network_architecture",'-na', type=str, default="[128,128]", help="Network architecture of the agent")
    parser.add_argument("--features_dim",'-fd', type=int, default=512, help="Dimension of the features extracted from the image")
    parser.add_argument("--seed",'-s', type=int, default=0, help="Seed for the agent")
    args = parser.parse_args()


    add_strategies = args.add_bp
    name = args.name_addition
    frame_stack = args.frame_stack
    tensorlog = args.logdir
    num_episodes = args.num_episodes
    env_index = args.env_index
    lr = args.learning_rate
    net_arch = eval(args.network_architecture)
    features_dim = args.features_dim
    seed = args.seed
    model = {"ppo":PPO,"dqn":DQN,"a2c":A2C,"rppo":RecurrentPPO}[args.agent_class]


    env_names = ["DoorKey-6x6-v0",
                 "DoorKey-8x8-v0",
                 "BlockedUnlockPickup-v0",
                 "KeyCorridorS3R2-v0",
                 "KeyCorridorS3R3-v0",
                 "Unlock-v0",
                 "UnlockPickup-v0",
                 ]
    env_name = f"MiniGrid-{env_names[env_index]}"
        
    verbose=0
    num_cpus = 1
    def create():
        return create_environment(add_strategies=add_strategies, env_name=env_name, stack_frames=frame_stack)
    
    eval_env = create()
    # env = make_vec_env(create, n_envs=num_cpus, seed=50)
    env = create()
    # eval_env = make_vec_env(create, n_envs=1,seed=50)

    # lr = 0.0001
    # net_arch = [128,128]
    # features_dim = 512

    gamma = 0.99
    ls = 100000
    bs = 32
    tf=10
    explore_frac = 0.4
    
    model_name = model.__name__
    # model_name += f'_lr{lr}_gamma{gamma}_ls{ls}_bs{bs}_steps{num_episodes/1000000}_tf{tf}_expfrac{explore_frac}'
    model_name += f'_ep{num_episodes/1_000_000}M_lr{lr}_'
    model_name += name + f'_{net_arch}_fd{features_dim}'
    model_name += f'_Frames{frame_stack}' if frame_stack is not None else ""
    model_name += f'_{env_name}'
    model_name += "_BP" if add_strategies else "_NOBP"


    policy = "CnnPolicy" if args.agent_class != "rppo" else "CnnLstmPolicy"
    model_path = "./models/" + model_name
    log_path = "./logs/"
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_path,
                                log_path=log_path, eval_freq=1000,
                                deterministic=True, render=False, n_eval_episodes=10)


    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor if frame_stack is None else MinigridFeaturesExtractor3D,
        features_extractor_kwargs=dict(features_dim=features_dim),
        net_arch=net_arch,
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


    agent = model(policy, env, verbose=verbose, tensorboard_log=tensorlog, policy_kwargs=policy_kwargs, seed=seed
                  , learning_rate=lr) 
    agent.learn(2, callback=eval_callback, log_interval=1, tb_log_name=model_name)
    # print(agent.policy)
    v

if __name__ == '__main__':
    train()