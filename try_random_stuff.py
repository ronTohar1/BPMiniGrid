import bppy as bp
from bppy import * 

@bp.thread
def add_hot():
    yield bp.sync(request= bp.BEvent("HOT"))
    yield bp.sync(request= bp.BEvent("HOT"))
    yield bp.sync(request= bp.BEvent("HOT"))


@bp.thread
def add_cold():
    yield bp.sync(request= bp.BEvent("COLD"))
    yield bp.sync(request= bp.BEvent("COLD"))
    yield bp.sync(request= bp.BEvent("COLD"))


@bp.thread
def control_temp():
    e = bp.BEvent("Dummy")
    while True:
        e = yield bp.sync(waitFor= bp.All(), block=e)


if __name__ == "__main__":
    b_program = BProgram(bthreads=[add_hot(), add_cold(), control_temp()],
                         event_selection_strategy=SimpleEventSelectionStrategy(),
                         listener=PrintBProgramRunnerListener())
    b_program.run()
