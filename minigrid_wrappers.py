import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.core import Env
from minigrid.core.constants import OBJECT_TO_IDX
import numpy as np

# Adds location of some objects to the info dict
class ObjectsLocationWrapper(ObservationWrapper):
    def __init__(self, env: Env, print_location=False, **kwargs):
        super().__init__(env,**kwargs)
        self.print_location = print_location
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["objects_location"] = self._get_location_info(obs)
        return obs, info
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
    
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["objects_location"] = self._get_location_info(obs)
        return obs, reward, terminated, truncated, info
    
    def _get_location_info(self, obs):
        image = obs["image"]

        def get_object_location(image,object_name):
            object_idx = OBJECT_TO_IDX[object_name]
            object_loc = np.where(image[:,:,0] == object_idx)
            object_loc = (object_loc[0][0],object_loc[1][0]) if len(object_loc[0]) > 0 else None
            return object_loc

        # make the locations a tuple of 2 elements
        key_loc = get_object_location(image,"key")
        box_loc = get_object_location(image,"box")
        agent_loc = get_object_location(image,"agent")
        ball_loc = get_object_location(image,"ball")
        door_loc = get_object_location(image,"door")
        door_state = obs["image"][door_loc[0],door_loc[1],2] if door_loc is not None else None
        goal_loc = get_object_location(image,"goal")
        return {"key": key_loc, 
                "box": box_loc,
                "agent": agent_loc, 
                "ball": ball_loc, 
                "door": door_loc, 
                "door_state": door_state,
                "goal": goal_loc,}


class OnlyImageObservation(ObservationWrapper):

    def __init__(self, env: Env, **kwargs):
        super().__init__(env,**kwargs)
        self.observation_space = self.observation_space["image"]
        self.action_space = env.action_space

    def observation(self, observation):
        return observation["image"]



    