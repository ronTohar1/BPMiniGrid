import numpy as np

DIRECTIONS = {
	"right": 0,
	"down": 1,
	"left": 2,
	"up": 3
}

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

def get_distance(pos1, pos2):
	return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])

def get_cell_in_front_of_agent(agent_pos, agent_dir):
	if agent_dir == DIRECTIONS["right"]:
		return (agent_pos[0] + 1, agent_pos[1])
	elif agent_dir == DIRECTIONS["down"]:
		return (agent_pos[0], agent_pos[1] + 1)
	elif agent_dir == DIRECTIONS["left"]:
		return (agent_pos[0] - 1, agent_pos[1])
	elif agent_dir == DIRECTIONS["up"]:
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
	return door_state == 0 # 0 is open

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

def get_distance_from_box(obs, info):
	return get_distance_from(obs, info, "box")

def get_distance_from_ball(obs, info):
	return get_distance_from(obs, info, "ball")

def get_distance_from_goal(obs, info):
	return get_distance_from(obs, info, "goal")

def get_key_position(obs,info):
	return info["objects_location"]["key"]

def get_box_position(obs,info):
	return info["objects_location"]["box"]

def get_ball_position(obs,info):
	return info["objects_location"]["ball"]

def get_goal_position(obs,info):
	return info["objects_location"]["goal"]

def get_agent_position(obs,info):
    return info["objects_location"]["agent"]
