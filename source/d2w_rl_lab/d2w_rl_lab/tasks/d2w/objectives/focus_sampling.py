"""D2W training-only sampling for zero hold and high-speed straight tracking."""
import torch


def focus_commands(commands, flat, max_forward):
    """Replace 65 percent of eligible nonzero flat samples with [1.6..2, 0, 0]."""
    n = commands.shape[0]
    out = commands.clone()
    eligible = flat & (commands.abs().amax(dim=1) > 1.e-6) & (max_forward >= 1.6)
    selected = eligible & (torch.rand(n, device=commands.device) < .65)
    upper = max_forward.clamp(max=2.)
    speeds = 1.6 + (upper - 1.6) * torch.rand(n, device=commands.device)
    speeds = torch.where(torch.rand(n, device=commands.device) < .25, upper, speeds)
    out[selected] = 0.
    out[selected, 0] = speeds[selected]
    return out, selected
