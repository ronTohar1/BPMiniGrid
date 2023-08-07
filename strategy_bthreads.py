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

	if door_state!= 2: # Door was already unlocked
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

		# print("distance", distance)
		# print("level", level)


strategies_bts = [ 
					level,	
					how_far_from_next_objective
					]

def create_strategies():
	# bthreads = [x() for x in strategies_bts + [request_all_moves()]]
    # bthreads = [request_all_moves()]
    bthreads = [x() for x in strategies_bts] + [request_all_moves()]
    return bthreads

def number_of_bthreads():
    # because of the request_all_moves which is not a strategy, but just there so we can choose any move
    return len(strategies_bts)
