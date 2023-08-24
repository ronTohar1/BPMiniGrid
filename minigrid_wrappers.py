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
            if len(object_loc[0]) == 0:
                object_loc = None
            else:
                object_loc = list(zip(object_loc[0],object_loc[1])) if object_loc else None
                if len(object_loc) == 1:
                    object_loc = object_loc[0]
        
            return object_loc
        
        def get_states(objects_locations):
            if objects_locations is None:
                return None
            if not isinstance(objects_locations,list):
                objects_locations = [objects_locations]
            states = [obs["image"][loc[0],loc[1],2] for loc in objects_locations]
            if len(states) == 1:
                    states = states[0]
            return states

        # make the locations a tuple of 2 elements
        key_loc = get_object_location(image,"key")
        box_loc = get_object_location(image,"box")
        agent_loc = get_object_location(image,"agent")
        ball_loc = get_object_location(image,"ball")
        door_loc = get_object_location(image,"door")

        door_state = get_states(door_loc) 

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



    