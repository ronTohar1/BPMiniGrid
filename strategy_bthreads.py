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
	h,w,_ = observation_shape
	pass
	# return np.zeros(bt_obs_shape)

def update_observation(name, value):
	pass
# def update_observation(name, obs):
# 	bthreads_progress[name] = obs


class BThreadObservation():
	def __init__(self, name, obs_shape):
		self.name = name
		self.observation = None
		self.obs_shape = obs_shape

	def init_observation(self):
		self.observation = np.zeros(self.obs_shape)

	def update_observation(self, value):
		self.observation.fill(value)

	def get_observation(self):
		return self.observation


class BThreadsObservations():
	def __init__(self):
		# self.observation_shape = (observation_shape[0], observation_shape[1])
		self.bthreads_obs = []

	# def init_bthread_obs(self, name):
	# 	self.bthreads_obs[name] = np.zeros(self.observation_shape)

	# def update_observation(self, name, value):
	# 	self.bthreads_obs[name].fill(value)

	def get_observations(self):
		return np.array([bt_obs.get_observation() for bt_obs in self.bthreads_obs])

	def add_new_bt_observation(self, bthread_observation: BThreadObservation):
		self.bthreads_obs.append(bthread_observation)
		bthread_observation.init_observation()

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
opened_door = EventSet(lambda event: event.name == "opened door")
closed_door = EventSet(lambda event: event.name == "closed door")
picked_up_box = EventSet(lambda event: event.name == "picked up box")

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

def is_door_unlocked(obs, info):
	door_state = info["objects_location"]["door_state"]
	if door_state is None: # agent is on the door so its unlocked and open
		return True
	return door_state != 2 # 2 is locked

def is_door_open(obs,info):
	door_state = info["objects_location"]["door_state"]
	if door_state is None: # agent is on the door so its unlocked and open
		return True
	return door_state == 1 # 1 is open

def agent_is_right_to_door(obs, info):
	agent_pos = info["objects_location"]["agent"]
	door_pos = info["objects_location"]["door"]
	if door_pos is None:
		return True
	return agent_pos[0] > door_pos[0]


def get_distance_from(obs,info,item):
	agent_pos = info["objects_location"]["agent"]
	item_pos = info["objects_location"][item]
	return get_distance(agent_pos, item_pos)

def get_distance_from_key(obs, info):
	return get_distance_from(obs, info, "key")

def get_distance_from_door(obs, info):
	return get_distance_from(obs, info, "door")

def get_key_position(obs,info):
	return info["objects_location"]["key"]

def get_box_position(obs,info):
	return info["objects_location"]["box"]
####################################################################

@b_thread
def count_left_turns(bt_obs: BThreadObservation):
	turns_left = 0
	while True:
		bt_obs.update_observation(turns_left)
		e = yield {waitFor: agent_events}
		if e.data["action"] == ACTIONS["left"]:
			turns_left += 1
		else:
			turns_left = 0

@b_thread
def count_right_turns(bt_obs:BThreadObservation):
	turns_right = 0
	while True:
		bt_obs.update_observation(turns_right)
		e = yield {waitFor: agent_events}
		if e.data["action"] == ACTIONS["right"]:
			turns_right += 1
		else:
			turns_right = 0

####################################################################
####################################################################

@b_thread
def pick_up_key_bt():
	# e = yield {waitFor: reset_event}
	# previous_obs, previous_info = e.data["observation"], e.data["info"]
	# while True:
	# 	e = yield {waitFor: agent_events}
	# 	obs, action, info = e.data["observation"], e.data["action"], e.data["info"]
	# 	if ACTIONS["pickup"] == action and is_key_in_front_of_agent(previous_obs, previous_info):
	# 		yield {request: BEvent("picked up key", {"observation":obs, "info":info})}
	# 	previous_obs, previous_info = obs, info
		

	picked_key = False
	while True:
		e = yield {waitFor: EventList([pick_up_action, dropped_key])}
		if e in dropped_key:
			picked_key = False
			continue
		obs, info = e.data["observation"], e.data["info"]
		key_pos = get_key_position(obs, info)
		if not picked_key and key_pos is None: # the agent didn't already have the key and the key is on the agent now
			yield {request: BEvent("picked up key", {"observation":obs, "info":info})}
			picked_key = True

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
	# while True:
	# 	yield {waitFor: picked_up_key}
	# 	e = yield {waitFor: EventList([dropped_key, toggle_action])}
	# 	while e not in dropped_key: # as long as the key is not dropped
	# 		obs, info = e.data["observation"], e.data["info"]
	# 		if is_door_in_front_of_agent(obs, info):
	# 			yield {request: BEvent("unlocked door", {"observation":obs, "info":info})}
	# 			return
	# 		else:
	# 			e = yield {waitFor: EventList([dropped_key, toggle_action])}

	while True:
		e = yield {waitFor: toggle_action}
		obs, info = e.data["observation"], e.data["info"]
		if is_door_unlocked(obs, info):
			yield {request: BEvent("unlocked door", {"observation":obs, "info":info})}
			return 

@b_thread
def unlock_env_level_bt(bt_obs:BThreadObservation):
	name = "unlock level"
	while True:
		bt_obs.update_observation(0)
		yield {waitFor: picked_up_key}
		bt_obs.update_observation(1)
		e = yield {waitFor: EventList([dropped_key, unlocked_door])}
		if e in unlocked_door:
			bt_obs.update_observation(2)
			return 

@b_thread
def unlock_env_distance_from_objective_bt(bt_obs: BThreadObservation):
	e = yield {waitFor: reset_event}
	distance = get_distance_from_key(e.data["observation"], e.data["info"])
	key_on_agent = False
	while True:
		bt_obs.update_observation(distance)
		e = yield {waitFor: EventList([forward_action, picked_up_key, dropped_key, unlocked_door])}
		if e in picked_up_key:
			key_on_agent = True
			distance = get_distance_from_door(e.data["observation"], e.data["info"])
		elif e in dropped_key:
			key_on_agent = False
			distance = 1
	
		if e in unlocked_door:
			bt_obs.update_observation(0)
			return

		if e in forward_action:
			obs, info = e.data["observation"], e.data["info"]
			distance = get_distance_from_door(obs,info) if key_on_agent else get_distance_from_key(obs,info)

####################################################################

@b_thread
def open_door_bt():
	while True:
		e = yield {waitFor: unlocked_door}
		yield {request: BEvent("opened door", {"observation":e.data["observation"], "info":e.data["info"]})}
		while True:
			yield {waitFor: closed_door}
			while True:
				e = yield {waitFor: toggle_action}
				if is_door_open(e.data["observation"], e.data["info"]):
					yield {request: BEvent("opened door", {"observation":e.data["observation"], "info":e.data["info"]})}
					break

@b_thread
def closed_door_bt():
	while True:
		yield {waitFor: opened_door}
		while True:
			yield {waitFor: toggle_action}
			if not is_door_open(e.data["observation"], e.data["info"]):
				yield {request: BEvent("closed door", {"observation":e.data["observation"], "info":e.data["info"]})}
				break

@b_thread
def picked_up_box_bt():
	while True:
		yield {waitFor: pick_up_action}
		obs, info = e.data["observation"], e.data["info"]
		if get_box_position(obs,info) is None:
			yield {request: BEvent("picked up box", {"observation":obs, "info":info})}

@b_thread
def unlock_pickup_env_level_bt(bt_obs:BThreadObservation):
	has_key = False
	open_door = False
	while True:
		if open_door:
			bt_obs.update_observation(3)
			if not has_key:
				bt_obs.update_observation(4)
		else:
			bt_obs.update_observation(0)
			if has_key:
				bt_obs.update_observation(1)

		e = yield {waitFor: [picked_up_key, dropped_key, opened_door, closed_door, picked_up_box]}
		if e in picked_up_key:
			has_key = True
		if e in dropped_key:
			has_key = False
		if e in opened_door:
			open_door = True
		if e in closed_door:
			open_door = False
		if e in picked_up_box:
			bt_obs.update_observation(5)
			return
		

@b_thread
def unlock_pickup_env_distance_from_objective_bt(bt_obs: BThreadObservation):
	e = yield {waitFor: reset_event}
	distance = get_distance_from_key(e.data["observation"], e.data["info"])
	key_on_agent = False

	while True:
		bt_obs.update_observation(distance)
							

			




####################################################################


observable_bthreads_general = [
							  count_left_turns,
							  count_right_turns,
]

internal_bthreads_unlock = [pick_up_key_bt,
					 drop_key_bt,
					 unlock_door_bt,
					 ]

observable_bthreads_unlock = [
								unlock_env_level_bt,
							  unlock_env_distance_from_objective_bt,
							  ]

bthreads = {
	"MiniGrid-Unlock-v0": (internal_bthreads_unlock, observable_bthreads_unlock),
	# "MiniGrid-UnlockPickup-v0": (internal_bthreads_unlock_pickup, observable_bthreads_unlock_pickup),
}

def create_strategies(observation_shape, env_name, add_general_bthreads=True):
	bt_obs_shape = (observation_shape[0], observation_shape[1])

	bthreads_observations = BThreadsObservations()
	internal_bthreads, observable_bthreads = bthreads[env_name]

	all_observable_bthreads = observable_bthreads + observable_bthreads_general if add_general_bthreads else observable_bthreads

	bt_obs_arr = [BThreadObservation(x.__name__, bt_obs_shape) for x in all_observable_bthreads]
	environment_bthreads = [x() for x in internal_bthreads] + [x(bt_obs_arr[i]) for i,x in enumerate(all_observable_bthreads)]
	for bt_obs in bt_obs_arr:
		bthreads_observations.add_new_bt_observation(bt_obs)
	
	return environment_bthreads, bthreads_observations


def number_of_bthreads(env_name, add_general_bthreads=True):
	additional_bthreads = 0 if not add_general_bthreads else len(observable_bthreads_general)
	n_bts = len(bthreads[env_name][1]) + additional_bthreads
	return n_bts




	