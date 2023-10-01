from enum import Enum
from bppy import *
# from util import BOARD_SIZE as bsize, UPDATE_STATE as update_state
from itertools import product
from gymnasium import spaces, Env
import numpy as np
import math
from itertools import combinations
from bp_events import *
from observation_inference import *

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
		self.bthreads_obs = []

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
passed_door_right = EventSet(lambda event: event.name == "passed door right")
passed_door_left = EventSet(lambda event: event.name == "passed door left")
picked_up_ball = EventSet(lambda event: event.name == "picked up ball")
dropped_ball = EventSet(lambda event: event.name == "dropped ball")
blocked_door = EventSet(lambda event: event.name == "blocked door")
unblocked_door = EventSet(lambda event: event.name == "unblocked door")
reached_goal = EventSet(lambda event: event.name == "reached goal")

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

	while True:
		e = yield {waitFor: pick_up_action}
		obs, info = e.data["observation"], e.data["info"]
		if get_key_position(obs, info) is None:
			yield {request: BEvent("picked up key", {"observation":obs, "info":info})}
			yield {waitFor: dropped_key}

@b_thread
def drop_key_bt():
	yield {waitFor: picked_up_key}
	while True:
			e = yield {waitFor: drop_action}
			obs, info = e.data["observation"], e.data["info"]
			if get_key_position(obs, info) is not None: # dropped key successfully
				yield {request: BEvent("dropped key", {"observation":obs, "info":info})}
				yield {waitFor: picked_up_key}

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
			e = yield {waitFor: toggle_action}
			if not is_door_open(e.data["observation"], e.data["info"]):
				yield {request: BEvent("closed door", {"observation":e.data["observation"], "info":e.data["info"]})}
				break

@b_thread
def picked_up_box_bt():
	while True:
		e = yield {waitFor: EventList([pick_up_action, toggle_action])}
		obs, info = e.data["observation"], e.data["info"]
		if get_box_position(obs,info) is None:
			yield {request: BEvent("picked up box", {"observation":obs, "info":info})}

@b_thread
def passed_door_bt():
	left_to_door = True
	while True:
		e = yield {waitFor: forward_action}
		obs, info = e.data["observation"], e.data["info"]
		if agent_is_right_to_door(obs, info):
			left_to_door = False
			yield {request: BEvent("passed door right", {"observation":obs, "info":info})}
		elif not left_to_door:
			left_to_door = True
			yield {request: BEvent("passed door left", {"observation":obs, "info":info})}

# @b_thread
# def unlock_pickup_env_level_bt_complicated(bt_obs:BThreadObservation):
# 	has_key, open_door, right_to_door, door_is_unlocked = False, False, False, False
# 	level = 0
# 	while True:
# 		if right_to_door: # in the room of the box
# 			level = 5 if has_key else 6
# 		elif open_door: # left to door and door is open
# 			level = 3 if has_key else 4
# 		else: 
# 			level = 2 if door_is_unlocked else 1 if has_key else 0

# 		bt_obs.update_observation(level)
# 		print("Level: ", level)
# 		e = yield {waitFor: EventList([picked_up_key, dropped_key, opened_door, closed_door, passed_door_right, passed_door_left, unlocked_door ,picked_up_box])}
# 		# update the state
# 		if e in picked_up_key:
# 			has_key = True
# 		if e in dropped_key:
# 			has_key = False
# 		if e in opened_door:
# 			open_door = True
# 		if e in closed_door:
# 			open_door = False
# 		if e in passed_door_right:
# 			right_to_door = True
# 		if e in passed_door_left:
# 			right_to_door = False
# 		if e in unlocked_door:
# 			door_is_unlocked = True
# 		if e in picked_up_box:
# 			bt_obs.update_observation(7)
# 			print("Level: ", 7)
# 			return
		

@b_thread
def unlock_pickup_env_level_bt(bt_obs:BThreadObservation):
	has_key, door_is_unlocked = False, False
	level = 0
	while True:
		if door_is_unlocked:
			level = 3
			if has_key:
				level = 2
		elif has_key:
			level = 1
		else:
			level = 0

		bt_obs.update_observation(level)
		# print("Level: ", level)
		e = yield {waitFor: EventList([picked_up_key, dropped_key, unlocked_door ,picked_up_box])}
		# update the state
		if e in picked_up_key:
			has_key = True
		if e in dropped_key:
			has_key = False
		if e in unlocked_door:
			door_is_unlocked = True
		if e in picked_up_box:
			bt_obs.update_observation(4)
			# print("Level: ", 4)
			return
		
# @b_thread
# def unlock_pickup_env_distance_from_objective_bt_complicated(bt_obs: BThreadObservation):
# 	e = yield {waitFor: reset_event}
# 	has_key, open_door, right_to_door, door_is_unlocked = False, False, False, False
# 	while True:
# 		obs, info = e.data["observation"], e.data["info"]
# 		if right_to_door or open_door: # in the room of the box
# 			distance = get_distance_from_box(e.data["observation"], e.data["info"])
# 		elif has_key or door_is_unlocked:
# 			distance = get_distance_from_door(e.data["observation"], e.data["info"])
# 		else:
# 			distance = get_distance_from_key(e.data["observation"], e.data["info"])

# 		bt_obs.update_observation(distance)
# 		print("distance:", distance)
# 		e = yield {waitFor: EventList([forward_action,picked_up_key, dropped_key, opened_door, closed_door, passed_door_right, passed_door_left, unlocked_door ,picked_up_box])}
# 		# update the state
# 		if e in forward_action:
# 			continue
# 		if e in picked_up_key:
# 			has_key = True
# 		if e in dropped_key:
# 			has_key = False
# 		if e in opened_door:
# 			open_door = True
# 		if e in closed_door:
# 			open_door = False
# 		if e in passed_door_right:
# 			right_to_door = True
# 		if e in passed_door_left:
# 			right_to_door = False
# 		if e in unlocked_door:
# 			door_is_unlocked = True

# 		if e in picked_up_box:
# 			bt_obs.update_observation(0)
# 			print("distance:", 0)
# 			return


@b_thread
def unlock_pickup_env_distance_from_objective_bt(bt_obs: BThreadObservation):
	e = yield {waitFor: reset_event}
	has_key, door_is_unlocked = False, False
	while True:
		obs, info = e.data["observation"], e.data["info"]
		if door_is_unlocked:
			distance = get_distance_from_box(obs, info)
		elif has_key:
			distance = get_distance_from_door(obs, info)
		else:
			distance = get_distance_from_key(obs, info)

		bt_obs.update_observation(distance)
		e = yield {waitFor: EventList([forward_action,picked_up_key, dropped_key, unlocked_door ,picked_up_box])}
		# update the state
		if e in forward_action:
			continue
		if e in picked_up_key:
			has_key = True
		if e in dropped_key:
			has_key = False
		if e in unlocked_door:
			door_is_unlocked = True

		if e in picked_up_box:
			bt_obs.update_observation(0)
			return

####################################################################

@b_thread
def pick_up_ball_bt():
	while True:
		e = yield {waitFor: pick_up_action}
		obs, info = e.data["observation"], e.data["info"]
		if get_ball_position(obs, info) is None:
			yield {request: BEvent("picked up ball", {"observation":obs, "info":info})}
			yield {waitFor : dropped_ball}

@b_thread
def drop_ball_bt():
	yield {waitFor: picked_up_ball}
	while True:
			e = yield {waitFor: drop_action}
			obs, info = e.data["observation"], e.data["info"]
			if get_ball_position(obs, info) is not None: # dropped ball successfully
				yield {request: BEvent("dropped ball", {"observation":obs, "info":info})}
				yield {waitFor: picked_up_ball}

@b_thread
def unblocked_door_bt():
	e = yield{waitFor: reset_event}
	initial_ball_pos = get_ball_position(e.data["observation"], e.data["info"])
	while True:
		yield {waitFor: picked_up_ball}
		e = yield {waitFor: dropped_ball}
		obs, info = e.data["observation"], e.data["info"]
		if get_ball_position(obs, info) != initial_ball_pos:
			yield {request: BEvent("unblocked door", {"observation":obs, "info":info})}
			yield {waitFor: blocked_door}
		

@b_thread
def blocked_door_bt():
	e = yield{waitFor: reset_event}
	initial_ball_pos = get_ball_position(e.data["observation"], e.data["info"])
	yield {waitFor: unblocked_door}
	while True:
		yield {waitFor: picked_up_ball}
		e = yield {waitFor: drop_action}
		obs, info = e.data["observation"], e.data["info"]
		if get_ball_position(obs,info) == initial_ball_pos:
			yield {request: BEvent("blocked door", {"observation":obs, "info":info})}
			yield {waitFor: unblocked_door}

@b_thread
def bup_env_level_bt(bt_obs:BThreadObservation):
	has_key, door_is_unlocked, has_ball, door_is_blocked = False, False, False, True
	level = 0
	while True:
		if door_is_unlocked:
			level = 4 if has_key else 5
		elif door_is_blocked:
			level = 1 if has_ball else 0
		else:
			level = 3 if has_key else 2
		bt_obs.update_observation(level)
		e = yield {waitFor: EventList([picked_up_key, dropped_key, unlocked_door ,picked_up_box, unblocked_door, picked_up_ball, dropped_ball])}
		# update the state
		if e in picked_up_key:
			has_key = True
		if e in dropped_key:
			has_key = False
		if e in unlocked_door:
			door_is_unlocked = True
		if e in unblocked_door:
			door_is_blocked = False
		if e in picked_up_ball:
			has_ball = True
		if e in dropped_ball:
			has_ball = False
		if e in picked_up_box:
			bt_obs.update_observation(6)
			return
	
@b_thread
def bup_env_distance_bt(bt_obs: BThreadObservation):
	e = yield {waitFor: reset_event}
	has_key, door_is_unlocked, has_ball, door_is_blocked = False, False, False, True
	while True:
		obs, info = e.data["observation"], e.data["info"]
		if door_is_blocked:
			if has_ball:
				distance = 0
			else:
				distance = get_distance_from_ball(obs, info)
		elif door_is_unlocked:
			distance = get_distance_from_box(obs, info)
		elif has_key:
			distance = get_distance_from_door(obs, info)
		else:
			distance = get_distance_from_key(obs, info)

		bt_obs.update_observation(distance)
		e = yield {waitFor: EventList([forward_action ,picked_up_key, dropped_key, unlocked_door ,picked_up_box, unblocked_door, picked_up_ball, dropped_ball])}
		# update the state
		if e in forward_action:
			continue
		if e in picked_up_key:
			has_key = True
		if e in dropped_key:
			has_key = False
		if e in unlocked_door:
			door_is_unlocked = True
		if e in unblocked_door:
			door_is_blocked = False
		if e in picked_up_ball:
			has_ball = True
		if e in dropped_ball:
			has_ball = False

		if e in picked_up_box:
			bt_obs.update_observation(0)
			return



####################################################################

@b_thread
def reached_goal_bt():
	while True:
		e = yield {waitFor: forward_action}
		obs, info = e.data["observation"], e.data["info"]
		if get_goal_position(obs, info) == get_agent_position(obs, info):
			yield {request: BEvent("reached goal", {"observation":obs, "info":info})}
			return
		
@b_thread
def exp_doorkey_env_level_bt(bt_obs: BThreadObservation):
	while True:
		yield {waitFor: picked_up_key}
		bt_obs.update_observation(1)
		while True:
			e = yield {waitFor: EventList([dropped_key, unlocked_door])}
			if e in unlocked_door:

@b_thread
def doorkey_env_level_bt(bt_obs: BThreadObservation):
	has_key, open_door, right_to_door = False, False, False
	while True:
		if open_door or right_to_door:
			level = 2
		elif has_key:
			level = 1
		else:
			level = 0
		
		bt_obs.update_observation(level)
		e = yield {waitFor: EventList([picked_up_key, dropped_key, opened_door, closed_door, reached_goal, passed_door_left, passed_door_right])}
		if e in reached_goal:
			bt_obs.update_observation(3)
			return
		
		if e in picked_up_key:
			has_key = True
		if e in dropped_key:
			has_key = False
		if e in opened_door:
			open_door = True
		if e in closed_door:
			open_door = False
		if e in passed_door_right:
			right_to_door = True
		if e in passed_door_left:
			right_to_door = False
		
def doorkey_env_distance_bt(bt_obs: BThreadObservation):
	e = yield {waitFor: reset_event}
	distance = get_distance_from_key(e.data["observation"], e.data["info"])
	has_key, open_door, right_to_door = False, False, False
	while True:
		if open_door or right_to_door:
			distance = get_distance_from_goal(e.data["observation"], e.data["info"])
		elif has_key:
			distance = get_distance_from_door(e.data["observation"], e.data["info"])
		else:
			distance = get_distance_from_key(e.data["observation"], e.data["info"])
		
		bt_obs.update_observation(distance)
		e = yield {waitFor: EventList([forward_action, picked_up_key, dropped_key, opened_door, closed_door, reached_goal, passed_door_left, passed_door_right])}
		
		if e in forward_action:
			continue
		if e in reached_goal:
			bt_obs.update_observation(0)
			return
		
		if e in picked_up_key:
			has_key = True
		if e in dropped_key:
			has_key = False
		if e in opened_door:
			open_door = True
		if e in closed_door:
			open_door = False
		if e in passed_door_right:
			right_to_door = True
		if e in passed_door_left:
			right_to_door = False
		
		


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

internal_bthreads_unlock_pickup = [pick_up_key_bt,
									drop_key_bt,
									unlock_door_bt,
									open_door_bt,
									closed_door_bt,
									picked_up_box_bt,
									passed_door_bt,
									]
observable_bthreads_unlock_pickup = [
									unlock_pickup_env_level_bt,
									unlock_pickup_env_distance_from_objective_bt,
									]					

observable_bthreads_bup = [
							bup_env_level_bt,
							bup_env_distance_bt,
							]

internal_bthreads_bup = [pick_up_key_bt,
					 drop_key_bt,
					 unlock_door_bt,
					 pick_up_ball_bt,
					 drop_ball_bt,
					 picked_up_box_bt,
					 unblocked_door_bt,
					 ]					

bthreads = {
	"MiniGrid-Unlock-v0": (internal_bthreads_unlock, observable_bthreads_unlock),
	"MiniGrid-UnlockPickup-v0": (internal_bthreads_unlock_pickup, observable_bthreads_unlock_pickup),
	"MiniGrid-BlockedUnlockPickup-v0": (internal_bthreads_bup, observable_bthreads_bup),
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




	