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


def init_observation(observation_shape, name):
	bt_obs_shape = (observation_shape[0], observation_shape[1],1)
	bthreads_progress[name] = np.zeros(bt_obs_shape)
	# return np.zeros(bt_obs_shape)

def update_observation(name, value):
	bthreads_progress[name].fill(value)

# def update_observation(name, obs):
# 	bthreads_progress[name] = obs

####################################################################
####################################################################

EventList = EventSetList

agent_events = EventSet(lambda event: isinstance(event, ExternalEvent) and event.name == "Agent Action")
reset_event = EventSet(lambda event: isinstance(event, ExternalEvent) and event.name == "Reset")
internal_events = EventSet(lambda event: not isinstance(event, ExternalEvent))

pick_up_action = EventSet(lambda event: event in agent_events and event.data["action"] == ACTIONS["pickup"])
drop_action = EventSet(lambda event: event in agent_events and event.data["action"] == ACTIONS["drop"])
toggle_action = EventSet(lambda event: event in agent_events and event.data["action"] == ACTIONS["toggle"])
forward_action = EventSet(lambda event: event in agent_events and event.data["action"] == ACTIONS["forward"])

picked_up_key = EventSet(lambda event: event.name == "picked up key")
dropped_key = EventSet(lambda event: event.name == "dropped key")
unlocked_door = EventSet(lambda event: event.name == "unlocked door")

def get_distance(pos1, pos2):
	return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])

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

def is_in_front_of_agent(obs,info,item):
	agent_dir = info["agent_direction"]
	agent_pos = info["objects_location"]["agent"]
	item_pos = info["objects_location"][item]
	cell_in_front_of_agent = get_cell_in_front_of_agent(agent_pos, agent_dir)
	return cell_in_front_of_agent == item_pos

def is_key_in_front_of_agent(obs, info):
	return is_in_front_of_agent(obs, info, "key")

def is_door_in_front_of_agent(obs, info):
	return is_in_front_of_agent(obs, info, "door")

def get_distance_from(obs,info,item):
	agent_pos = info["objects_location"]["agent"]
	item_pos = info["objects_location"][item]
	return get_distance(agent_pos, item_pos)

def get_distance_from_key(obs, info):
	return get_distance_from(obs, info, "key")

def get_distance_from_door(obs, info):
	return get_distance_from(obs, info, "door")

####################################################################
####################################################################

@b_thread
def pick_up_key_bt():
	previous_obs = None
	previous_info = None
	while True:
		e = yield {waitFor: agent_events}
		obs, action, info = e.data["observation"], e.data["action"], e.data["info"]
		if ACTIONS["pickup"] == action and is_key_in_front_of_agent(previous_obs, previous_info):
			yield {request: BEvent("picked up key", {"observation":obs, "info":info})}
		previous_obs = obs
		previous_info = info

	# picked_key = False
	# while True:
	# 	e = yield {waitFor: pick_up_action}
	# 	info = e.data["info"]
	# 	key_pos = info["objects_location"]["key"]
	# 	if not picked_key and not key_pos:
	# 		yield {request: BEvent("picked up key")}
	# 		picked_key = True


@b_thread
def drop_key_bt():
	while True:
		yield {waitFor: picked_up_key}
		while True:
			e = yield {waitFor: drop_action}
			obs, info = e.data["observation"], e.data["info"]
			if is_key_in_front_of_agent(obs, info): # dropped key successfully
				break
			
		yield {request: BEvent("dropped key", {"observation":obs, "info":info})}

@b_thread
def unlock_door_bt():
	while True:
		yield {waitFor: picked_up_key}
		e = yield {waitFor: EventList([dropped_key, toggle_action])}
		while e not in dropped_key: # as long as the key is not dropped
			obs, info = e.data["observation"], e.data["info"]
			if is_door_in_front_of_agent(obs, info):
				yield {request: BEvent("unlocked door")}
				return
			else:
				e = yield {waitFor: EventList([dropped_key, toggle_action])}

@b_thread
def unlock_env_level_bt(obs_shape):
	name = "unlock level"
	init_observation(obs_shape, name)
	while True:
		update_observation(name, 0) # at level 0
		yield {waitFor: picked_up_key}
		update_observation(name, 1) # at level 1
		e = yield {waitFor: [dropped_key, unlocked_door]}
		if e in unlocked_door:
			update_observation(name, 2) # at level 2 - finished the episode successfully
			yield {waitFor: All()}

@b_thread
def unlock_env_distance_from_objective_bt(obs_shape):
	name = "unlock_env_distance_from_objective"
	init_observation(obs_shape, name)
	e = yield {waitFor: reset_event}
	distance = get_distance_from_key(e.data["observation"], e.data["info"])
	key_on_agent = False

	while True:
		update_observation(name, distance)
		print("distance from objective: ", distance)
		e = yield {waitFor: EventList([forward_action, picked_up_key, dropped_key, unlocked_door])}
		if e in picked_up_key:
			key_on_agent = True
			distance = get_distance_from_door(e.data["observation"], e.data["info"])
		elif e in dropped_key:
			key_on_agent = False
			distance = 1
	
		if e in unlocked_door:
			update_observation(name, 0)
			yield {waitFor: All()}

		if e in forward_action:
			obs, info = e.data["observation"], e.data["info"]
			distance = get_distance_from_door(obs,info) if key_on_agent else get_distance_from_key(obs,info)
		


####################################################################


internal_bthreads_unlock = [pick_up_key_bt,
					 drop_key_bt,
					 unlock_door_bt,
					 ]

observable_bthreads_unlock = [
							# unlock_env_level_bt,
							  unlock_env_distance_from_objective_bt,
							  ]

bthreads = {
	"MiniGrid-Unlock-v0": (internal_bthreads_unlock, observable_bthreads_unlock),
}

def create_strategies(observation_shape, env_name):
	internal_bthreads, observable_bthreads = bthreads[env_name]
	return [x() for x in internal_bthreads] + [x(observation_shape) for x in observable_bthreads]


def number_of_bthreads(env_name):
    return len(bthreads[env_name][1])


class GymBProgram(BProgram):
	def __init__(self, env: Env,bthreads=None, source_name=None, event_selection_strategy=None, listener=None):
		self.env = env

		# super init
		super().__init__(bthreads, source_name, event_selection_strategy, listener)

	def create_strategies(self):
		pass
	

	