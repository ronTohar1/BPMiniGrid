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
DIRECTOINS = {
	"right": 0,
	"down": 1,
	"left": 2,
	"up": 3
}



ROWS = 3
COLS = 3

MAX_STEPS = 200

# defining the agent actions
agent_actions = [bp.Event(action_name) for action_name in ACTIONS.keys()]

# defining the internal events for the environment
move_event = bp.EventSet(lambda e: e.name.startswith("Move"))


class Move(bp.BEvent):
    def __init__(self, i, j):
        super().__init__("Move", {"i": i, "j": j})


# b-thread representing wall locations, blocking moves to this wall
@bp.thread
def wall(i, j):  # block moves to this wall
    yield bp.sync(block=Move(i, j))


# b-thread for the start of the environment run, triggering the initial location of the agent
@bp.thread
def start():
    yield bp.sync(request=Move(0, 0), block=agent_actions)


# b-thread representing the goal of the environment, providing a terminal state with reward 1
@bp.thread
def goal(i,j):
    steps = 0
    e = yield bp.sync(waitFor=bp.EventSetList([Move(i,j), agent_actions]))
    while e != Move(i,j):
        steps+=1
        e = yield bp.sync(waitFor=bp.EventSetList([Move(i,j), agent_actions]))
    yield bp.sync(block=bp.All(), localReward= 1 - 0.9 ((steps+1) / MAX_STEPS))  # reached goal - terminate the program with a reward of 1

@bp.thread
def limit_steps():
    steps = 0
    while steps < MAX_STEPS:
        yield bp.sync(block=agent_actions)  # reached goal - terminate the program with a reward of 1
        steps+=1
    yield bp.sync(block=bp.All, localReward=0)


# b-thread for the agent, requesting actions based on the current location
@bp.thread
def agent():
    while True:
        e = yield bp.sync(waitFor=move_event)
        current_location = (e.data["i"], e.data["j"])
        yield bp.sync(request=agent_actions)


# function to initialize the b-program with the defined b-threads
def init_bprogram():
    """
    returning an instance for the standard 4x4 frozen lake environment:
        ["SFFF",
         "FHFH",
         "FFFH",
         "HFFG"]
    """
    walls_locations = [(1, 1), (2,1)]
    goals_location = [(2,2)]
    return bp.BProgram(bthreads=[start(), agent()] +
                                [goal(i,j) for (i, j) in goals_location ] +
                                [wall(i, j) for (i, j) in walls_locations] +
                                [wall(-1, j) for j in range(COLS)] +
                                [wall(ROWS, j) for j in range(COLS)] +
                                [wall(i, -1) for i in range(ROWS)] +
                                [wall(i, COLS) for i in range(ROWS)],
                       event_selection_strategy=bp.SimpleEventSelectionStrategy(),
                       listener=bp.PrintBProgramRunnerListener())

# listing all possible events in the b-program
all_events = [Move(i, j) for i in range(-1, ROWS+1) for j in range(-1, COLS+1)] + agent_actions + [bp.BEvent("HOLE"), bp.BEvent("GOAL")]


# defining the observation space for the environment based on the current_location variable of the agent b-thread
class FrozenLakeObservationSpace(BPObservationSpace):
    def __init__(self, dim):
        super().__init__([dim], np.int64, None)
    def bp_state_to_gym_space(self, bthreads_states):
        agent_bthread_statement = [x for x in bthreads_states if "current_location" in x.get("locals", {})][0]
        current_location = agent_bthread_statement["locals"]["current_location"]
        return np.asarray([current_location[0]*COLS + current_location[1]], dtype=self.dtype)


# initialize environment with the defined b-program generator, observation space, and reward function
env = BPEnv(bprogram_generator=init_bprogram,
            action_list=agent_actions,  # all program events are considered as possible actions for the agent
            observation_space=FrozenLakeObservationSpace(ROWS*COLS),
            reward_function=lambda rewards: sum(filter(None, rewards)))

# reset environment and print initial state
state, _ = env.reset()
print(state)
terminated = False
while not terminated:  # loop until the environment (b-program) terminates
    action_id = env.action_space.sample()  # sample an action
    state, reward, terminated, _, info = env.step(action_id)  # take a step with the sampled action
    print(agent_actions[action_id].name, state, reward, terminated, info)
print("finished sampling")

# importing stable_baselines3 and initializing a PPO model
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000, progress_bar=True)

# running the environment again with the trained model
state, _ = env.reset()
print(state)
terminated = False
while not terminated:
    action_id, _states = model.predict(state)
    state, reward, terminated, _, info = env.step(action_id)
    print(agent_actions[action_id].name, state, reward, terminated, info)

