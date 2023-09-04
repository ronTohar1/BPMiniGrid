from enum import Enum
from bppy import *
# from util import BOARD_SIZE as bsize, UPDATE_STATE as update_state
from itertools import product
from gymnasium import spaces
import numpy as np
import math
from itertools import combinations

# shared data - maybe something like {num_ships_found, num_2_size_found, found_5_size:True\False....} 
# And then bthreads can listen to those events (wait for update data event) and than look at the data.
# e.g found the 2 size ship, then no need to hit in distances of 2 only at 3 and more.

bthreads_progress = {}
state = None
info = None

ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
}

def set_state(s, information=None):
	global state
	global info
	state = s
	info = information


def get_new_space():
	if state is None:
		raise Exception("State is None")
	
	h,w,_ = state.shape
	return np.zeros((h,w))
	
	# return np.zeros((11,11))
	# return []

def reset_strategy(name):
	bthreads_progress[name] = get_new_space()

def update_strategy(name, space):
	bthreads_progress[name] = space

def get_strategy(name):
	return bthreads_progress[name]

def reset_all_strategies():
	for name in bthreads_progress:
		reset_strategy(name)

def kill_strategy(name):
	reset_strategy(name)


pred_all_events = EventSet(lambda event: True)

def bevent_wrapper(event_list):
	return [BEvent(str(event)) for event in event_list]

# We need this so we can choose any event, because bppy checks if event is selectable
@b_thread
def request_all_moves():
	name = "requester"
	moves = list(ACTIONS_ALL.keys())
	moves = bevent_wrapper(moves)
	while True:
		event = yield {request: moves, waitFor: pred_all_events}

####################################################################
####################################################################


def get_level(info, initial_ball_pos):
	ball_pos = info["objects_location"]["ball"]
	key_pos = info["objects_location"]["key"]
	door_pos = info["objects_location"]["door"]
	door_state = info["objects_location"]["door_state"]
	agent_pos = info["objects_location"]["agent"]

	if door_state is None or door_state!= 2: # Door was already unlocked
		if door_pos is None or door_pos[0] <= agent_pos[0]: # I am right to the door or at the door
			level = 7
		elif ball_pos and ball_pos == initial_ball_pos: # I am not in the box room and ball is blocking the door
			level = 0
		elif door_state == 0: #Open door
			if ball_pos is None or key_pos is None: # I am not in the box room and I have not dropped the key or the ball
				level = 5
			else:
				level = 6
		elif door_state == 1: #Unlocked but not open door
			level = 4

		if level == 7 and key_pos and ball_pos: # I am in the box room and I dropped the key and the ball
			level = 8
		return level

	if ball_pos:
		if ball_pos == initial_ball_pos:
			level = 0
		else:
			level = 2
	else:
		level = 1

	if level == 2 and not key_pos:
		level = 3

	return level

def get_distance(pos1, pos2):
	return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])

@b_thread
def level():
	name = "level"
	level = 0
	space = get_new_space()
	initial_ball_pos = info["objects_location"]["ball"]
	while True:
		space.fill(level)
		update_strategy(name, space)

		action = yield {waitFor: pred_all_events}
		# action = int(action.name)
		level =	get_level(info, initial_ball_pos)
	
			



		
		

@b_thread
def how_far_from_next_objective():
	name = "objective_distance"
	level = 0
	space = get_new_space()
	initial_ball_pos = info["objects_location"]["ball"]
	agent_pos = info["objects_location"]["agent"]

	distance = get_distance(agent_pos, initial_ball_pos)
	while True:
		space.fill(distance)
		update_strategy(name, space)

		action = yield {waitFor: pred_all_events}
		# action = int(action.name)
		level =	get_level(info, initial_ball_pos)

		ball_pos = info["objects_location"]["ball"]
		key_pos = info["objects_location"]["key"]
		door_pos = info["objects_location"]["door"]
		door_state = info["objects_location"]["door_state"]
		agent_pos = info["objects_location"]["agent"]
		box_pos = info["objects_location"]["box"]

		if level ==8 or level == 7 or level == 6: # I am right to the door or the door is open
			if box_pos:
				distance = get_distance(agent_pos, box_pos)
			else:
				distance = 0
		if level == 5: # I am not in the box room and I have not dropped the key or the ball, but door is open
			distance = get_distance(agent_pos, box_pos)
		if level == 4: # I am not in the box and door is unlocked, but not open.
			distance = get_distance(agent_pos, door_pos)
		if level == 3: # I am not in the box and I have the key
			distance = get_distance(agent_pos, door_pos)

		if level == 2: # I am not in the box and the ball is not blocking the door
			distance = get_distance(agent_pos, key_pos)
		if level == 1: # I am not in the box and I have the ball
			distance = get_distance(agent_pos, key_pos)
		if level == 0: # I am not in the box and the ball is blocking the door
			distance = get_distance(agent_pos, initial_ball_pos)




#################################################################### 
# Door key problem

def door_key_get_level(info):
	# ball_pos = info["objects_location"]["ball"]
	key_pos = info["objects_location"]["key"]
	door_pos = info["objects_location"]["door"]
	door_state = info["objects_location"]["door_state"]
	agent_pos = info["objects_location"]["agent"]
	level = 0

	if door_state!= 2: # Door was already unlocked
		if door_state == 0 or door_pos is None: # Open door (if None it means the agent is at the door)
			level = 3
		elif agent_pos[0] >= door_pos[0]: # I am right to the door or at the door
			level = 3
		elif door_state == 1 and agent_pos[0] < door_pos[0]: # Unlocked but not open door (and agent is left to the door)
			level = 2
	else:
		if key_pos is None: # Holding Key
			level = 1
		else:
			level = 0

	return level

@b_thread
def door_key_level():
	name = "door_key_level"
	space = get_new_space()
	level = 0
	while True:
		space.fill(level)
		update_strategy(name, space)

		action = yield {waitFor: pred_all_events}
		level = door_key_get_level(info)

@b_thread
def door_key_how_far_from_next_objective():
	name = "door_key_objective_distance"
	space = get_new_space()
	distance = 0
	while True:
		
		level = door_key_get_level(info)

		# ball_pos = info["objects_location"]["ball"]
		key_pos = info["objects_location"]["key"]
		door_pos = info["objects_location"]["door"]
		door_state = info["objects_location"]["door_state"]
		agent_pos = info["objects_location"]["agent"]
		goal_pos = info["objects_location"]["goal"]
		if level == 3:
			if goal_pos:
				distance = get_distance(agent_pos, goal_pos)
			else:
				distance = 0
		elif level == 2:
			distance = get_distance(agent_pos, door_pos)
		elif level == 1:
			distance = get_distance(agent_pos, door_pos)
		elif level == 0:
			distance = get_distance(agent_pos, key_pos)

		space.fill(distance)
		update_strategy(name, space)

		action = yield {waitFor: pred_all_events}


		
@b_thread
def count_turns_in_direction_left():
	name = "count_turns_in_direction_left"
	space = get_new_space()
	turned_left = 0
	while True:
		space.fill(turned_left)
		update_strategy(name, space)

		action = yield {waitFor: pred_all_events}
		action = int(action.name)

		if action == 0:
			turned_left += 1
		else:
			turned_left = 0


@b_thread
def count_turns_in_direction_right():
	name = "count_turns_in_direction_right"
	space = get_new_space()
	turned_right = 0
	while True:
		space.fill(turned_right)
		update_strategy(name, space)

		action = yield {waitFor: pred_all_events}
		action = int(action.name)

		if action == 1:
			turned_right += 1
		else:
			turned_right = 0


#################################################################### 
# Key Corridor problem


def get_door_state(door_loc, doors_states, doors_loc):
	if door_loc is None:
		return None
	if door_loc not in doors_loc:
		return None
	index = doors_loc.index(door_loc)
	return doors_states[index]	


@b_thread
def key_corridor_level():
	name = "key_corridor_level"
	space = get_new_space()
	level = 0

	last_round_agent_is_on_door = False
	in_room = False
	
	doors_state = info["objects_location"]["door_state"]
	# where door_state is 2 - the only locked door
	ball_door_index = [i for i, x in enumerate(doors_state) if x == 2][0]
	ball_door_loc = info["objects_location"]["door"][ball_door_index]

	while True:

		key_pos = info["objects_location"]["key"]
		doors_pos = info["objects_location"]["door"]
		doors_state = info["objects_location"]["door_state"]
		agent_pos = info["objects_location"]["agent"]
		ball_pos = info["objects_location"]["ball"]
		level = 0

		ball_door_pos = ball_door_loc
		ball_door_state = get_door_state(ball_door_pos, doors_state, doors_pos)

		if ball_door_state == None: # agent is on the door
			level = 4
			last_round_agent_is_on_door = True
			in_room = False
		elif ball_door_state!= 2: # Door was already unlocked
			if (agent_pos[0] > ball_door_pos[0] and last_round_agent_is_on_door) or in_room: # I am right to the door and I was on the door last round
				level = 4
				in_room=True
			elif ball_door_state == 0: # Open door (if None it means the agent is at the door)
				level = 3
			elif ball_door_state == 1: # Unlocked but not open door (and agent is left to the door)
				level = 2
		else:
			if key_pos is None: # Holding Key
				level = 1
			else:
				level = 0


		# Check if holding key in levels:
		if level >= 2:
			if key_pos is not None:
				level +=1

		space.fill(level)
		update_strategy(name, space)

		action = yield {waitFor: pred_all_events}




@b_thread
def key_corridor_how_far_from_objective():
	name = "key_corridor_how_far_from_objective"
	space = get_new_space()
	level = 0
	distance = 0

	last_round_agent_is_on_door = False
	in_room = False
	
	doors_state = info["objects_location"]["door_state"]
	# where door_state is 2 - the only locked door
	ball_door_index = [i for i, x in enumerate(doors_state) if x == 2][0]
	ball_door_loc = info["objects_location"]["door"][ball_door_index]

	while True:

		key_pos = info["objects_location"]["key"]
		doors_pos = info["objects_location"]["door"]
		doors_state = info["objects_location"]["door_state"]
		agent_pos = info["objects_location"]["agent"]
		ball_pos = info["objects_location"]["ball"]
		level = 0

		ball_door_pos = ball_door_loc
		ball_door_state = get_door_state(ball_door_pos, doors_state, doors_pos)

		if ball_door_state == None: # agent is on the door
			level = 4
			last_round_agent_is_on_door = True
			in_room = False
		elif ball_door_state!= 2: # Door was already unlocked
			if (agent_pos[0] > ball_door_pos[0] and last_round_agent_is_on_door) or in_room: # I am right to the door and I was on the door last round
				level = 4
				in_room=True
			elif ball_door_state == 0: # Open door (if None it means the agent is at the door)
				level = 3
			elif ball_door_state == 1: # Unlocked but not open door (and agent is left to the door)
				level = 2
		else:
			if key_pos is None: # Holding Key
				level = 1
			else:
				level = 0

		if level == 0:
			distance = get_distance(agent_pos, key_pos)
		elif level == 1:
			distance = get_distance(agent_pos, ball_door_loc)
		elif level == 2:# Unlocked but not open door (and agent is left to the door)
			distance = get_distance(agent_pos, ball_door_loc)
		elif level >= 3: # Open door or in the room
			if ball_pos is None:
				distance = 0
			else:
				distance = get_distance(agent_pos, ball_pos)

		space.fill(distance)
		update_strategy(name, space)

		action = yield {waitFor: pred_all_events}

####################################################################
####################################################################

def unlock_get_level(info):
	key_pos = info["objects_location"]["key"]
	door_pos = info["objects_location"]["door"]
	door_state = info["objects_location"]["door_state"]

	level = 0 
	if not key_pos:
		level = 1

	if door_state is not None and door_state == 0: #Open door
		level = 2 

	return level

@b_thread
def unlock_level():
	name = "unlock_level"
	space = get_new_space()
	level = 0
	while True:
		space.fill(level)
		update_strategy(name, space)

		action = yield {waitFor: pred_all_events}
		level = unlock_get_level(info)

@b_thread
def unlock_how_far_from_next_objective():
	name = "unlock_how_far_from_next_objective"
	key_pos = info["objects_location"]["key"]
	door_pos = info["objects_location"]["door"]
	agent_pos = info["objects_location"]["agent"]
	space = get_new_space()
	level = 0
	distance = get_distance(key_pos, agent_pos)
	while True:
		space.fill(distance)
		update_strategy(name, space)

		action = yield {waitFor: pred_all_events}
		key_pos = info["objects_location"]["key"]
		door_pos = info["objects_location"]["door"]
		agent_pos = info["objects_location"]["agent"]
		level = unlock_get_level(info)

		if level == 0:
			distance = get_distance(key_pos, agent_pos)
		elif level == 1:
			distance = get_distance(door_pos, agent_pos)
		elif level == 2:
			distance = 0


####################################################################
####################################################################

def unlock_pickup_get_level(info):
	key_pos = info["objects_location"]["key"]
	door_pos = info["objects_location"]["door"]
	door_state = info["objects_location"]["door_state"]
	agent_pos = info["objects_location"]["agent"]
	box_pos = info["objects_location"]["box"]
	level = 0

	if box_pos is None:
		level = 5

	elif door_state!= 2: # Door was already unlocked
		if door_state == 0 or door_pos is None: # Open door (if None it means the agent is at the door)
			level = 3
		elif agent_pos[0] >= door_pos[0]: # I am right to the door or at the door
			level = 3
		elif door_state == 1 and agent_pos[0] < door_pos[0]: # Unlocked but not open door (and agent is left to the door)
			level = 2

		if level == 3 and key_pos is not None and door_pos is not None and agent_pos[0] > door_pos[0]: # I am right to the door without the key
			level = 4
	else:
		if key_pos is None: # Holding Key
			level = 1
		else:
			level = 0

	return level


@b_thread
def unlock_pickup_level():
	name = "unlock_level"
	space = get_new_space()
	level = 0
	while True:
		space.fill(level)
		update_strategy(name, space)

		action = yield {waitFor: pred_all_events}
		level = unlock_pickup_get_level(info)

@b_thread
def unlock_pickup_how_far_from_next_objective():
	name = "unlock_how_far_from_next_objective"
	key_pos = info["objects_location"]["key"]
	door_pos = info["objects_location"]["door"]
	agent_pos = info["objects_location"]["agent"]
	space = get_new_space()
	level = 0
	distance = get_distance(key_pos, agent_pos)
	while True:
		space.fill(distance)
		update_strategy(name, space)

		action = yield {waitFor: pred_all_events}
		key_pos = info["objects_location"]["key"]
		door_pos = info["objects_location"]["door"]
		agent_pos = info["objects_location"]["agent"]
		box_pos = info["objects_location"]["box"]
		level = unlock_pickup_get_level(info)

		if level == 0:
			distance = get_distance(key_pos, agent_pos)
		elif level == 1 or level == 2:
			distance = get_distance(door_pos, agent_pos)
		elif level == 3 or level == 4:
			distance = get_distance(box_pos, agent_pos)
		else:
			distance = 0

		
		



strategies_doorkey = [
					door_key_level,
					door_key_how_far_from_next_objective,
					count_turns_in_direction_left,
					count_turns_in_direction_right,
]

strategies_blockedunlock = [ 
					level,	
					how_far_from_next_objective
					]

strategies_keycorridor = [
					key_corridor_level,
					key_corridor_how_far_from_objective,
					count_turns_in_direction_left,
					count_turns_in_direction_right,
					]

strategies_unlock = [unlock_level,
					 unlock_how_far_from_next_objective,
					 count_turns_in_direction_left,
					 count_turns_in_direction_right,]

strategies_unlock_pickup = [unlock_pickup_level,
							unlock_pickup_how_far_from_next_objective,
							count_turns_in_direction_left,
							count_turns_in_direction_right,]


strategies_bts = strategies_unlock_pickup

def create_strategies():
	# bthreads = [x() for x in strategies_bts + [request_all_moves()]]
    # bthreads = [request_all_moves()]
    bthreads = [x() for x in strategies_bts] + [request_all_moves()]
    return bthreads

def number_of_bthreads():
    # because of the request_all_moves which is not a strategy, but just there so we can choose any move
    return len(strategies_bts)
