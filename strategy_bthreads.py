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

# @b_thread
# def slow_when_blocked():
# 	name = "slow_when_blocked"
# 	space = get_new_space()

# 	update_strategy(name,space)
# 	threshold = 2
# 	while True:
# 		action = yield {waitFor: pred_all_events}
# 		action = int(action.name)
# 		presence = state[0]
# 		ego_index = (len(presence)//2, len(presence)//2)
# 		other_vehicles = []
# 		for i in range(len(presence)):
# 			for j in range(len(presence)):
# 				if presence[i][j] == 1 and (i,j) != ego_index:
# 					other_vehicles.append((i,j))
# 		is_close_vehicle_front = False


# 		for vehicle in other_vehicles:
# 			# check if vehicle is on our lane and in front of us and close enough
# 			if vehicle[0] == ego_index[0] and  vehicle[1] > ego_index[1] and np.abs(vehicle[1] - ego_index[1]) <= threshold:
# 				is_close_vehicle_front = True
# 				break		

# 		if is_close_vehicle_front:
# 			space.fill(1)
# 		else:
# 			space.fill(0)
			
# 		update_strategy(name, space)


@b_thread
def level():
	"""This bthread just updates the level of the game
	level 0 - start of episode
	level 1 - after picking up ball
	level 2 - after dropping ball in a different location
	level 3 - after picking up the key
	level 4 - after toggle the door
	level 5 - after drop the key
	"""

	name = "level"
	space = get_new_space()
	update_strategy(name,space)
	print("level bt info: ", info)
	while True:
		action = yield {waitFor: pred_all_events}
		action = int(action.name)
		
		print("bt info: ", info)


strategies_bts = [ 
					level,	
					]

def create_strategies():
	# bthreads = [x() for x in strategies_bts + [request_all_moves()]]
    # bthreads = [request_all_moves()]
    bthreads = [x() for x in strategies_bts] + [request_all_moves()]
    return bthreads

def number_of_bthreads():
    # because of the request_all_moves which is not a strategy, but just there so we can choose any move
    return len(strategies_bts)
