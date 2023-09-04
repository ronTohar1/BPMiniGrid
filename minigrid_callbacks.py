from stable_baselines3.common.callbacks import BaseCallback


class MinigridCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def _on_step(self) -> bool:
            """
            :return: If the callback returns False, training is aborted early.
            """
            return True
    
    def _on_rollout_end(self) -> None:
        # print(self.locals)
        n_bf_turns = 0
        for info in self.locals["infos"]:
            n_bf_turns += info["back_and_forth_turns"]
        n_bf_turns /= len(self.locals["infos"])
        # self.logger.record("rollout/n_back_and_forth_turns", n_bf_turns)
        print("n_back_and_forth_turns", n_bf_turns)
        # return super()._on_rollout_end()


