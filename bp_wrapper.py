from bppy import *
# from util import *



# from eventsUtilities import EventHandler

# event_handler = EventHandler()


class BPwrapper():
	def __init__(self):
		self.ess: EventSelectionStrategy = None
		self.bprog: BProgram = None
		self.selectable_events = None
		self.tickets = None
		self.listener = None
		self.initialized = False

	def reset(self, bprogram: BProgram):
		self.bprog = bprogram
		self.bprog.setup()
		self.ess = self.bprog.event_selection_strategy
		self.tickets = self.bprog.tickets
		self.selectable_events = self.ess.selectable_events(self.tickets)
		self.listener = self.bprog.listener
		self.initialized = True

		# self.update_internal_state()

	# def update_internal_state(self):
	# 	while any([e.name == UPDATE_STATE for e in self.selectable_events]):
	# 		# choose the update state event
	# 		update_state_event = [e for e in self.selectable_events if e.name == UPDATE_STATE][0]
	# 		self.choose_event(update_state_event)
	# 		self.update_selectable_events()

	def update_selectable_events(self):
		self.selectable_events = self.ess.selectable_events(self.tickets)

	def advance_randomly(self) -> BEvent:
		chosen_event = self.ess.select(self.tickets)
		self.choose_event(chosen_event)
		return chosen_event

	def choose_event(self, event: BEvent):
		# if event not in self.selectable_events:
			# raise Exception("Tried to choose blocked event!")
		self.listen(event)
		self.bprog.advance_bthreads(self.bprog.tickets, event)
		# self.update_internal_state()
		self.update_selectable_events()

	# Just an idea for now, not sure if it's needed or possible.
	def choose_external_event(self, event: BEvent):
		pass

	def listen(self, event: BEvent):
		if self.listener:
			self.listener.event_selected(b_program=self.bprog, event=event)

	def get_selectable_events(self):
		return self.selectable_events

	# def is_game_finished(self):
	# 	""" Returns true if the game has finished.
	# 	If the last event available is the win event - choosing it."""

	# 	selectable = self.selectable_events
	# 	if selectable is None:
	# 		raise Exception("Selectable events is None")
	# 	if len(selectable) > 1:  # More than one possible event left.
	# 		return False
	# 	if len(selectable) == 1: # One event left - either win event or might be some other event available.
	# 		event = selectable[0]
	# 		game_finished = event_handler.is_game_finished_event(event)
	# 		if game_finished: # If game is finished - choosing the win event
	# 			self.choose_event(event)
	# 		return game_finished

	# 	return True # No events are selectable

