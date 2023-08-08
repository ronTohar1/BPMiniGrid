import os
import pickle as pkl
import random
import sys
import time
from pprint import pprint

import optuna
from absl import flags
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from create_environment import create_environment
from stable_baselines3.common.env_util import make_vec_env
from minigrid_feature_extractor import MinigridFeaturesExtractor

from sample_params import sample_ppo_params
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from typing import Optional

import gymnasium as gym

import torch
FLAGS = flags.FLAGS
FLAGS(sys.argv)


class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(
            self,
            eval_env: gym.Env,
            trial: optuna.Trial,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 0,
            best_model_save_path: Optional[str] = None,
            log_path: Optional[str] = None,
            callback_after_eval: Optional[BaseCallback] = None
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            callback_after_eval=callback_after_eval
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            continue_training = super()._on_step()
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elapsed time ?
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if needed
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return continue_training


N_TRIALS = 100
N_STARTUP_TRIALS = 0
N_EVALUATIONS = 10
N_TIMESTEPS = int(10_000)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 1

study_path = "minigames/move_to_beacon/optuna/4"



def objective(trial: optuna.Trial) -> float:

    # time.sleep(random.random() * 16)

    add_strategies = False
    # add_strategies = trial.suggest_categorical("add_strategies", [True, False])

    sampled_hyperparams = sample_ppo_params(trial)
    
    create = lambda: create_environment(add_strategies=add_strategies, env_name="MiniGrid-DoorKey-5x5-v0")
    env = make_vec_env(create, n_envs=6, seed=50)
    eval_env = create_environment(add_strategies=add_strategies)
    eval_env = Monitor(eval_env)

    model = PPO("CnnPolicy", env=env, seed=None, verbose=0, **sampled_hyperparams)

    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=50, verbose=1)
    eval_callback = TrialEvalCallback(
        env, trial,
        n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=False,
        callback_after_eval=stop_callback
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward

if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize", storage="sqlite:///db.sqlite3")
    # study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=600)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))

# if __name__ == "__main__":

#     sampler = TPESampler(n_startup_trials=10, multivariate=True)
#     pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=10)

#     study = optuna.create_study(
#         sampler=sampler,
#         pruner=pruner,
#         load_if_exists=True,
#         direction="maximize",
#     )

#     try:
#         study.optimize(objective, n_jobs=4, n_trials=128)
#     except KeyboardInterrupt:
#         pass

#     print("Number of finished trials: ", len(study.trials))

#     trial = study.best_trial
#     print(f"Best trial: {trial.number}")
#     print("Value: ", trial.value)

#     print("Params: ")
#     for key, value in trial.params.items():
#         print(f"    {key}: {value}")

#     study.trials_dataframe().to_csv(f"{study_path}/report.csv")

#     with open(f"{study_path}/study.pkl", "wb+") as f:
#         pkl.dump(study, f)

#     try:
#         fig1 = plot_optimization_history(study)
#         fig2 = plot_param_importances(study)
#         fig3 = plot_parallel_coordinate(study)

#         fig1.show()
#         fig2.show()
#         fig3.show()

#     except (ValueError, ImportError, RuntimeError) as e:
#         print("Error during plotting")
#         print(e)