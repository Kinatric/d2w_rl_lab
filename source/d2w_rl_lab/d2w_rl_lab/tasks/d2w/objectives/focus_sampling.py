"""D2W flat straight sampling; preserve zero samples and the old speed band."""
import torch
from .flat_speed_extension import MAX_FORWARD_M_S, PREVIOUS_FORWARD_M_S


def focus_commands(commands, flat, max_forward):
    """Focus 65% of eligible samples; reserve 25% of that branch for >2 m/s.

    Most focused samples retain 1.6--2 m/s practice. The extension uses the
    effective terrain/global bound, with explicit boundary samples. Standing,
    terrain distribution, command duration and all motor limits are unchanged.
    These are resampling probabilities, not fractions of total training time.
    """
    n = commands.shape[0]
    out = commands.clone()
    eligible = flat & (commands.abs().amax(dim=1) > 1.e-6) & (max_forward >= 1.6)
    selected = eligible & (torch.rand(n, device=commands.device) < .65)
    upper = max_forward.clamp(max=MAX_FORWARD_M_S)
    legacy_upper = upper.clamp(max=PREVIOUS_FORWARD_M_S)
    speeds = 1.6 + (legacy_upper - 1.6) * torch.rand(n, device=commands.device)
    speeds = torch.where(torch.rand(n, device=commands.device) < .25, legacy_upper, speeds)
    extended = (upper > PREVIOUS_FORWARD_M_S) & (torch.rand(n, device=commands.device) < .25)
    new_speed = PREVIOUS_FORWARD_M_S + (upper-PREVIOUS_FORWARD_M_S)*torch.rand(n, device=commands.device)
    new_speed = torch.where(torch.rand(n, device=commands.device) < .25, upper, new_speed)
    speeds = torch.where(extended, new_speed, speeds)
    out[selected] = 0.
    out[selected, 0] = speeds[selected]
    return out, selected
