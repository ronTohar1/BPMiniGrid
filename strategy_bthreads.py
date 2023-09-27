from enum import Enum
from bppy import *
# from util import BOARD_SIZE as bsize, UPDATE_STATE as update_state
from itertools import product
from gymnasium import spaces, Env
import numpy as np
import math
from itertools import combinations
from bp_events import *
# shared data - maybe something like {num_ships_found, num_2_size_found, found_5_size:True\False....} 
# And then bthreads can listen to those events (wait for update data event) and than look at the data.
# e.g found the 2 size ship, then no need to hit in distances of 2 only at 3 and more.

bthreads_progress = {}

ACTIONS_ALL = {
        0: 'Turn Left',
        1: 'Turn Right',
        2: 'Forward',
        3: 'Pickup',
        4: 'Drop',
		5: "Toggle",
		6: "Done",
}

ACTIONS = {
	"left": 0,
	"right": 1,
	"forward": 2,
	"pickup": 3,
	"drop": 4,
	"toggle": 5,
	"done": 6
}
ACTIONS_NAMES = Enum('ACTIONS', list(ACTIONS.keys()))
DIRECTOINS = {
	"right": 0,
	"down": 1,
	"left": 2,
	"up": 3
}


def new_observation(observation_shape):
	bt_obs_shape = (observation_shape[0], observation_shape[1],1)
	return np.zeros(bt_obs_shape)

def update_observation(name, space):
	bthreads_progress[name] = space

####################################################################
####################################################################

agent_events = EventSet(lambda event: isinstance(event, ExternalEvent))
internal_events = EventSet(lambda event: not isinstance(event, ExternalEvent))

pick_up_action = EventSet(lambda event: event in agent_events and event.data["action"] == ACTIONS["pickup"])
drop_action = EventSet(lambda event: event in agent_events and event.data["action"] == ACTIONS["drop"])

def get_cell_in_front_of_agent(agent_pos, agent_dir):
	if agent_dir == DIRECTOINS["right"]:
		return (agent_pos[0] + 1, agent_pos[1])
	elif agent_dir == DIRECTOINS["down"]:
		return (agent_pos[0], agent_pos[1] + 1)
	elif agent_dir == DIRECTOINS["left"]:
		return (agent_pos[0] - 1, agent_pos[1])
	elif agent_dir == DIRECTOINS["up"]:
		return (agent_pos[0], agent_pos[1] - 1)
	else:
		raise Exception("Invalid agent direction")

def is_key_in_front_of_agent(obs, info):
	agent_dir = info["agent_direction"]
	agent_pos = info["agent_pos"]
	key_pos = info["key_pos"]
	cell_in_front_of_agent = get_cell_in_front_of_agent(agent_pos, agent_dir)
	return cell_in_front_of_agent == key_pos

@b_thread
def pick_up_key(obs_shape):
	name = "pick_up_key"
	bthread_obs = new_observation(obs_shape)
	update_observation(name, bthread_obs)
	previous_obs = None
	while True:
		e = yield {waitFor: agent_events}
		obs, action, info = e.data["observation"], e.data["action"], e.data["info"]
		if ACTIONS["pickup"] == action and is_key_in_front_of_agent(previous_obs, info):
			yield {request: BEvent("picked up key")}
		previous_obs = obs

def dropped_key(obs_shape):
	name = "dropped_key"
	bthread_observation = new_observation(obs_shape)
	update_observation(name, bthread_observation)
	while True:
		yield {waitFor: BEvent("picked up key")}
		e = yield {waitFor: drop_action}
		obs, info = e.data["observation"], e.data["info"]
		while not is_key_in_front_of_agent(obs, info):
			e = yield {waitFor: agent_events}
			obs, info = e.data["observation"], e.data["info"]
		yield {request: BEvent("dropped key")}



####################################################################


internal_bthreads = [pick_up_key,
					 dropped_key,
					 
					 ]

observable_bthreads = []



def create_strategies(observation_shape):
    bthreads = [x(observation_shape) for x in internal_bthreads + observable_bthreads]
    return bthreads

def number_of_bthreads():
    return len(strategies_bts)


# class GymBProgram(BProgram):
# 	def __init__(self, env: Env, bthreads=None, source_name=None, event_selection_strategy=None, listener=None):
# 		self.env = env
# 		self.observation_space = env.observation_space
# 		self.action_space = env.action_space

# 		# super init
# 		super().__init__(bthreads, source_name, event_selection_strategy, listener)

	