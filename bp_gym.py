import gymnasium
from gymnasium import spaces
from bppy import BProgram
from bp_wrapper import BPwrapper
from strategy_bthreads import create_strategies, number_of_bthreads, bthreads_progress
# from priority_event_selection_strategy import PriorityEventSelectionStrategy
import numpy as np
from bppy import *
# from util import BOARD_SIZE
from gymnasium import ObservationWrapper
from bp_events import *

class BPGymEnv(ObservationWrapper):
    def __init__(self, env, add_strategies=False, as_image=False, axis=0, **kwargs): # Expecting an environment with a gymnasium interface
        super().__init__(env, **kwargs)
        self.add_strategies = add_strategies
        # self.env = env
        # self.action_space = env.action_space
        # self.observation_space = env.observation_space
        self.as_image = as_image
        self.axis = axis

        # initialize the bprogram containing the strategies
        if (self.add_strategies and number_of_bthreads() > 0):
            self.n_bthreads = number_of_bthreads()
            if as_image:
                # channels = env.observation_space.shape[axis]
                # w,h= env.observation_space.shape[1], env.observation_space.shape[2]
                # shape=(number_of_bthreads() + channels, w, h)
                
                low,high,shape,type = 0, 255, env.observation_space.shape, env.observation_space.dtype
                new_shape = list(shape)
                new_shape[axis] += self.n_bthreads
                new_shape = tuple(new_shape)

                observation_space = spaces.Box(shape=new_shape, low=low, high=high, dtype=type)
                self.observation_space = observation_space
            else:
                observation_space = spaces.Tuple((self.observation_space, spaces.Box(low=-np.inf, high=np.inf, shape=(1,number_of_bthreads()), dtype=np.float32)))
                self.observation_space = spaces.flatten_space(observation_space)
            
            self.bprog = BPwrapper()

    # Concatenate the observations from the environment and the strategies
    # env_obs - the observation from the environment (numpy array of 2 - 10x10 boards)
    # bp_obs - the observation from the strategies (numpy array of x - 10x10 boards)
    def _concat_observations(self, env_obs, bp_obs):
        if not self.as_image:
            return (env_obs, bp_obs)
        else:
            new_shape = list(self.observation_space.shape)
            new_shape[self.axis] = self.n_bthreads
            new_shape = tuple(new_shape)
            bp_obs = np.reshape(bp_obs, new_shape)
            return np.concatenate((env_obs, bp_obs), axis=self.axis)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        if (self.add_strategies):
            agent_action_event = ExternalEvent("Agent Action",{"observation":observation, "action":action, "info":info})
            self.bprog.super_step(agent_action_event)
            bp_obs = self._get_bp_observation()
            observation = self._concat_observations(observation, bp_obs)

        return observation, reward, terminated, truncated, info 
    
    
    def reset(self,seed=None, options=None):
        observation, info = self.env.reset()

        if (self.add_strategies):
            self._reset_strategies(observation.shape)
            obs_strats = self._get_bp_observation()
            observation = self._concat_observations(observation, obs_strats)

        return observation, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()

    def _get_bp_observation(self):
        strategies = bthreads_progress.values()
        return np.array([np.array(strategy) for strategy in strategies])
    
    def _reset_strategies(self, observation_shape):
        bprogram = BProgram(bthreads=create_strategies(observation_shape),
                             event_selection_strategy=SimpleEventSelectionStrategy(),
                             listener=PrintBProgramRunnerListener())
        self.bprog.reset(bprogram)