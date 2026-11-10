from isaaclab.managers import ActionManager
import torch
from collections.abc import Sequence

# ActionManagerD2W is a simple custom ActionManager that
# maintain _prev_prev_action for action smoothness reward computation.
class ActionManagerD2W(ActionManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_prev_action = torch.zeros_like(self._action)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        super().reset(env_ids)
        if env_ids is None:
            self._prev_prev_action.zero_()
        else:
            self._prev_prev_action[env_ids] = 0.0
        return {}

    def process_action(self, action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        # check if action dimension is valid
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        # store the input actions
        self._prev_prev_action[:] = self._prev_action
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    @property
    def prev_prev_action(self):
        return self._prev_prev_action
