import bppy as bp
from bppy.gym import *
import numpy as np
ACTIONS_INDICES = {
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
	"done": 6 # Should I include this action?
}



class Orientation:
    RIGHT = "right"
    LEFT = "left"
    DOWN = "down"
    UP = "up"

ORIENTATION_POS = {
    Orientation.RIGHT: [0,1],
    Orientation.LEFT: [0,-1],
    Orientation.DOWN: [-1,0],
    Orientation.UP: [1,0],
}

ROWS = 3
COLS = 3

MAX_STEPS = 200

# defining the agent actions

# defining the internal events for the environment
move_event = bp.EventSet(lambda e: e.name.startswith("Move"))
open_event = bp.EventSet(lambda e: e.name.startswith("Open"))
rotate_event = bp.EventSet(lambda e: e.name.startswith("Rotate"))
close_event = bp.EventSet(lambda e: e.name.startswith("Close"))
pickup_event = bp.EventSet(lambda e: e.name.startswith("Pickup"))
drop_event = bp.EventSet(lambda e: e.name.startswith("Drop"))

agent_actions_event_set = bp.EventSetList([move_event,
                                           open_event,
                                           rotate_event,
                                           close_event,
                                           pickup_event,
                                           drop_event,])

class Open(bp.BEvent):
    def __init__(self, i, j):
        super().__init__("Open", {"i": i, "j": j})

class Close(bp.BEvent):
    def __init__(self, i, j):
        super().__init__("Close", {"i": i, "j": j})

class Move(bp.BEvent):
    def __init__(self, i, j):
        super().__init__("Move", {"i": i, "j": j})

class Rotate(bp.BEvent):
    def __init__(self, i, j, orientation):
        super().__init__("Rotate", {"i": i, "j": j, "orientation": orientation})

class Pickup(bp.BEvent):
    def __init__(self, i, j):
        super().__init__("Pickup", {"i": i, "j": j})

class Drop(bp.BEvent):
    def __init__(self, i, j):
        super().__init__("Drop", {"i": i, "j": j})


def get_forward_location(current_location, orientation):
    return [a+b for a,b in zip(list(current_location), ORIENTATION_POS[orientation])]

def get_allowed_orientations(current_orientation):
    if current_orientation == Orientation.UP or current_orientation == Orientation.DOWN:
        return [Orientation.LEFT, Orientation.RIGHT]
    else:
        return [Orientation.UP, Orientation.DOWN]

#################
##### Map  ######
#################

# b-thread representing wall locations, blocking moves to this wall
@bp.thread
def wall(i, j):  # block moves to this wall
    yield bp.sync(block=[Move(i, j), Drop(i,j)])

# b-thread for the start of the environment run, triggering the initial location of the agent
@bp.thread
def start(i,j):
    yield bp.sync(request=Move(i,j), block=bp.AllExcept(Move(i,j)))


# b-thread representing the goal of the environment, providing a terminal state with reward 1
@bp.thread
def goal(i,j):
    steps = 0
    e = yield bp.sync(waitFor=agent_actions_event_set)
    while e != Move(i,j):
        steps+=1
        e = yield bp.sync(waitFor=agent_actions_event_set)
    
    print("Finished with reward ", 1 - 0.9 * ((steps+1) / MAX_STEPS))
    yield bp.sync(block=bp.All(), localReward= 1 - 0.9 * ((steps+1) / MAX_STEPS))  # reached goal - terminate the program with a reward

# bthread for limiting the number of steps in the environment
@bp.thread
def limit_steps():
    steps = 0
    while steps < MAX_STEPS:
        yield bp.sync(waitFor=agent_actions_event_set)  # reached goal - terminate the program with a reward of 1
        steps+=1
    yield bp.sync(block=bp.All(), localReward=0)

# Can rotate only 90 degrees
@bp.thread
def rotate_90_deg_only(initial_orientation):
    curr_orientation = initial_orientation
    while True:
        possible_orientations = get_allowed_orientations(curr_orientation)
        blocked_rotate_events = bp.EventSet(lambda e: e in rotate_event and e.data["orientation"] not in possible_orientations)
        e = yield bp.sync(block=blocked_rotate_events, waitFor=rotate_event)
        curr_orientation = e.data["orientation"]

# allowed Move event to be forward only according to the orientation of the player
@bp.thread
def move_forward_only(initial_orientation):
    orientation = initial_orientation
    e = yield bp.sync(waitFor=move_event) # starting move event
    location = [e.data["i"], e.data["j"]]

    while True:
        forward_loc = get_forward_location(location, orientation)
        not_forward_event_set = bp.EventSet(lambda e: (e in move_event) and not (e == Move(*forward_loc)))
        e = yield bp.sync(block=not_forward_event_set, waitFor=bp.EventSetList([move_event, rotate_event]))
        if e in rotate_event:
            orientation = e.data["orientation"]
        location = [e.data["i"], e.data["j"]]


#################
##### Door ######
#################

# bthread describing a door - blocks objects like a wall untill opened
@bp.thread
def door(i, j):
    while True:
        yield bp.sync(block=[Move(i, j), Drop(i,j)], waitFor=Open(i,j)) 
        yield bp.sync(waitFor=Close(i,j))

# bthread for allowing open and closing of a door only at the actual door location
@bp.thread
def door_at(i, j):
    open_door_at_other_locations_event_set = bp.EventSet(lambda e: (e in open_event) and not (e == Open(i,j)))
    close_door_at_other_locations_event_set = bp.EventSet(lambda e: (e in close_event) and not (e == Close(i,j)))
    yield bp.sync(block=bp.EventSetList([open_door_at_other_locations_event_set, close_door_at_other_locations_event_set]))

# bthread for allowing opening a door only if we are facing it directly
@bp.thread
def door_open_if_infront(i,j,initial_orientation):
    orientation = initial_orientation
    door_location = [i,j]
    e = yield bp.sync(waitFor=move_event) # initial move event

    while True:
        agent_location = [e.data["i"], e.data["j"]]
        if e in rotate_event:
            orientation = e.data["orientation"]

        forward_location = get_forward_location(agent_location, orientation)
        if forward_location == door_location:
            e = yield bp.sync(waitFor=bp.EventSetList([move_event, rotate_event])) # no block as we are infront of door
        else:
            e = yield bp.sync(block=bp.EventSetList([Open(i,j), Close(i,j)]),
                                waitFor=bp.EventSetList([move_event, rotate_event])) # not infront of door


# bthread enforcing the need of a key to open a door
@bp.thread
def door_open_with_key(i, j):
    while True:
        yield bp.sync(block=Open(i, j), waitFor=pickup_event)  
        yield bp.sync(waitFor=drop_event)

# bthread for opening and closing a door alternation
@bp.thread
def door_alternate_open_close(i, j):
    while True:
        yield bp.sync(block=Close(i, j), waitFor=Open(i,j))  
        yield bp.sync(block=Open(i,j), waitFor=Close(i,j))


#################
##### key  ######
#################

# Did not finish
@bp.thread
def key(i,j):
    while True:
        yield bp.sync(block=bp.EventSetList([Move(i,j), Drop(i,j)]), waitFor=Pickup(i,j))
        e = yield bp.sync(waitFor=drop_event)
        i,j = e.data["i"], e.data["j"]

# allow drop and pickup alternatively only
@bp.thread
def key_drop_pickup_alternate():
    while True:
        yield bp.sync(block=drop_event, waitFor=pickup_event)
        yield bp.sync(block=pickup_event, waitFor=drop_event)

# allow pickup in key place only
@bp.thread
def key_pickup_only(i,j):
    def key_pickup_only_event_set(i,j):
        return  bp.EventSet(lambda e: (e in pickup_event) and not (e == Pickup(i,j)))
    while True:
        e = yield bp.sync(block=key_pickup_only_event_set(i,j), waitFor=drop_event)
        i,j = e.data["i"], e.data["j"]

# blocks pickup if we picked the key up
@bp.thread
def key_picked_up(i,j):
    while True:
        e = yield bp.sync(waitFor=Pickup(i,j))
        e = yield bp.sync(block=pickup_event, waitFor=drop_event)
        i,j = e.data["i"], e.data["j"]

# bthread for allowing to pickup a key only if we are infront
@bp.thread
def key_pickup_if_infront(i,j, initial_orientation):
    orientation = initial_orientation
    door_location = [i,j]

    e = yield bp.sync(waitFor=move_event) # initial move event

    while True:
        agent_location = [e.data["i"], e.data["j"]]
        if e in rotate_event:
            orientation = e.data["orientation"]
        forward_location = get_forward_location(agent_location, orientation)
        if forward_location == door_location:
            e = yield bp.sync(waitFor=bp.EventSetList([move_event, rotate_event])) # no block as we are infront of key
        else:
            e = yield bp.sync(block=Pickup(i,j), waitFor=bp.EventSetList([move_event, rotate_event])) # not infront of key

        


#################
##### Agent #####
#################

# b-thread for the agent, requesting actions based on the current location
@bp.thread
def random_agent(orientation):
    e = yield bp.sync(waitFor=bp.EventSetList([move_event, rotate_event]))
    while True:
        if e in move_event or e in rotate_event:
            i,j = e.data["i"], e.data["j"]
        if e in rotate_event:
            orientation = e.data["orientation"]
            
        forward_loc = get_forward_location([i,j], orientation)
        events = [ 
                #    Pickup(*forward_loc), 
                #    Drop(*forward_loc),
                   Move(*forward_loc), 
                   Rotate(i,j,np.random.choice(get_allowed_orientations(orientation))),
                #    Close(*forward_loc), Open(*forward_loc)
                ]
        e = yield bp.sync(request=events, waitFor=bp.All())     

################
## Additional ##
################


@bp.thread
def request_move(priority=1):
    while True:
        events = [Move(i,j) for i in range(ROWS) for j in range(COLS)]
        e = yield bp.sync(request=events, priority=priority)

@bp.thread
def request_open(priority=1):
    while True:
        events = [Open(i,j) for i in range(ROWS) for j in range(COLS)]
        e = yield bp.sync(request=events, priority=priority)


@bp.thread
def request_close(priority=1):
    while True:
        events = [Close(i,j) for i in range(ROWS) for j in range(COLS)]
        e = yield bp.sync(request=events, priority=priority)

@bp.thread
def scenario_1():
    # yield bp.sync(request=Pickup(2,0))
    yield bp.sync(request=Move(1,0))
    yield bp.sync(request=Pickup(2,0))
    yield bp.sync(request=Move(2,0))
    yield bp.sync(request=Rotate(2,0, Orientation.RIGHT))
    yield bp.sync(request=Open(2,1))
    yield bp.sync(request=Close(2,1))
    yield bp.sync(request=Open(2,1))
    yield bp.sync(request=Move(2,1))


@bp.thread
def scenario_2():
    yield bp.sync(request=Move(1,0))
    yield bp.sync(request=Pickup(2,0))
    yield bp.sync(request=Move(2,0))
    yield bp.sync(request=Rotate(2,0, Orientation.RIGHT))
    yield bp.sync(request=Rotate(2,0, Orientation.DOWN))
    yield bp.sync(request=Drop(1,0))
    yield bp.sync(request=Rotate(2,0, Orientation.RIGHT))
    yield bp.sync(request=Open(2,1))
    yield bp.sync(request=Move(2,1))

@bp.thread
def scenario_3():
    yield bp.sync(request=Move(1,0))
    yield bp.sync(request=Pickup(2,0))
    yield bp.sync(request=Move(2,0))
    yield bp.sync(request=Rotate(2,0, Orientation.RIGHT))
    yield bp.sync(request=Open(2,1))
    yield bp.sync(request=Move(2,1))
    yield bp.sync(request=Move(2,2))

    yield bp.sync(request=Rotate(2,2, Orientation.DOWN))
    yield bp.sync(request=Rotate(2,2, Orientation.LEFT))
    yield bp.sync(request=Drop(2,1))

    yield bp.sync(request=Rotate(2,2, Orientation.DOWN))

    yield bp.sync(request=Move(1,2))
    yield bp.sync(request=Move(0,2))


@bp.thread
def scenario_4():
    yield bp.sync(request=Move(1,0))
    yield bp.sync(request=Pickup(2,0))
    yield bp.sync(request=Move(2,0))
    yield bp.sync(request=Rotate(2,0, Orientation.RIGHT))
    yield bp.sync(request=Open(2,1))
    yield bp.sync(request=Move(2,1))
    yield bp.sync(request=Move(2,2))

    yield bp.sync(request=Rotate(2,2, Orientation.DOWN))

    yield bp.sync(request=Move(1,2))
    yield bp.sync(request=Move(0,2))


# function to initialize the b-program with the defined b-threads
def init_bprogram():
    walls_extra_locations = [(0, 1), (1,1)]
    goals_location = [(0, COLS-1)]
    key_loc = (2,0)
    door_loc = (2,1)

    initial_orientation = Orientation.UP
    key_bthread=[
                key(*key_loc), 
                 key_drop_pickup_alternate(), 
                 key_picked_up(*key_loc), 
                 key_pickup_only(*key_loc), 
                 key_pickup_if_infront(*key_loc, initial_orientation)
                ]
    door_bthread=[
                    door(*door_loc), 
                  door_at(*door_loc), 
                  door_alternate_open_close(*door_loc), 
                  door_open_if_infront(*door_loc, initial_orientation),
                  door_open_with_key(*door_loc)
                ]
    
    map_bthread= [
                    limit_steps(),
                   move_forward_only(initial_orientation),
                   rotate_90_deg_only(initial_orientation),
                   ]
    
    additional_bthreads = [
                    # random_agent(initial_orientation),
                #    request_move(6),
                #    request_open(5),
                #    request_close(7),
                # scenario_1(),
                # scenario_2(),
                # scenario_3(),
                scenario_4(),

    ]

    return bp.BProgram(bthreads=
                       [start(0,0), ] +
                                door_bthread +
                                key_bthread + 
                                map_bthread +
                                additional_bthreads +
                                [goal(i,j) for (i, j) in goals_location] +
                                [wall(i, j) for (i, j) in walls_extra_locations] +
                                [wall(-1, j) for j in range(COLS)] +
                                [wall(ROWS, j) for j in range(COLS)] +
                                [wall(i, -1) for i in range(ROWS)] +
                                [wall(i, COLS) for i in range(ROWS)],
                    #    event_selection_strategy=bp.SimpleEventSelectionStrategy(),
                       event_selection_strategy=bp.PriorityBasedEventSelectionStrategy(),
                       listener=bp.PrintBProgramRunnerListener())

bprogram = init_bprogram()
bprogram.run()



# # defining the observation space for the environment based on the current_location variable of the agent b-thread
# class FrozenLakeObservationSpace(BPObservationSpace):
#     def __init__(self, dim):
#         super().__init__([dim], np.int64, None)
#     def bp_state_to_gym_space(self, bthreads_states):
#         agent_bthread_statement = [x for x in bthreads_states if "current_location" in x.get("locals", {})][0]
#         current_location = agent_bthread_statement["locals"]["current_location"]
#         return np.asarray([current_location[0]*COLS + current_location[1]], dtype=self.dtype)


# # initialize environment with the defined b-program generator, observation space, and reward function
# env = BPEnv(bprogram_generator=init_bprogram,
#             action_list=agent_actions,  # all program events are considered as possible actions for the agent
#             observation_space=FrozenLakeObservationSpace(ROWS*COLS),
#             reward_function=lambda rewards: sum(filter(None, rewards)))

# # reset environment and print initial state
# state, _ = env.reset()
# print(state)
# terminated = False
# while not terminated:  # loop until the environment (b-program) terminates
#     action_id = env.action_space.sample()  # sample an action
#     state, reward, terminated, _, info = env.step(action_id)  # take a step with the sampled action
#     print(agent_actions[action_id].name, state, reward, terminated, info)
# print("finished sampling")

# # importing stable_baselines3 and initializing a PPO model
# from stable_baselines3 import PPO
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=100000, progress_bar=True)

# # running the environment again with the trained model
# state, _ = env.reset()
# print(state)
# terminated = False
# while not terminated:
#     action_id, _states = model.predict(state)
#     state, reward, terminated, _, info = env.step(action_id)
#     print(agent_actions[action_id].name, state, reward, terminated, info)

