"""Pure tensor memory for qualified stair-transfer repetition.

The caller owns geometric, support, command and terrain qualification. This
helper adds no lift reward and prescribes no periodic gait.
"""
import math
import torch


def transfer_repeat_cost(events, forward_xy, gate, valid, previous, dt):
    """Return an event-only repeat score and detached per-environment memory.

    ``forward_xy`` contains actual body world XY positions, despite its name.
    Repetition is local to 2 seconds and 1 metre of displacement from the last
    eligible event. Simultaneous events clear side memory symmetrically.
    """
    if events.ndim != 2 or events.shape[1] != 2:
        raise ValueError('events must have shape (N, 2)')
    if forward_xy.shape != events.shape or gate.shape != events.shape[:1] or valid.shape != gate.shape:
        raise ValueError('position, gate and valid shapes must match events')
    if events.dtype != torch.bool or gate.dtype != torch.bool or valid.dtype != torch.bool:
        raise TypeError('events, gate and valid must be boolean')
    if not math.isfinite(float(dt)) or float(dt) <= 0.:
        raise ValueError('dt must be finite and positive')

    position = forward_xy.detach()
    active = gate & valid
    n = events.shape[0]
    if previous is None:
        last_side = torch.full((n,), -1, dtype=torch.long, device=events.device)
        elapsed = position.new_zeros(n)
        last_position = position.clone()
        inactive_age = position.new_zeros(n)
    else:
        last_side = previous['last_side'].detach()
        elapsed = previous['elapsed'].detach()
        last_position = previous['last_position'].detach()
        inactive_age = previous.get('inactive_age', position.new_zeros(n)).detach()

    # Reset first, so discontinuity cannot compare with another episode's event.
    # Invalid episode/command continuity clears immediately. A transient gate
    # dropout suspends scoring but preserves history for at most 0.2 seconds.
    inactive_age = torch.where(active, torch.zeros_like(inactive_age), inactive_age + float(dt))
    retain = valid & (inactive_age <= .2)
    last_side = torch.where(retain, last_side, torch.full_like(last_side, -1))
    elapsed = torch.where(retain, elapsed + float(dt), torch.zeros_like(elapsed))
    last_position = torch.where(retain[:, None], last_position, position)
    eligible = events & active[:, None]
    single = eligible.sum(-1) == 1
    both = eligible.all(-1)
    side = eligible[:, 1].to(torch.long)  # Only read when exactly one event exists.
    local = (elapsed <= 2.) & ((position - last_position).square().sum(-1) <= 1.)
    repeat = single & (side == last_side) & local
    cost = repeat.to(position.dtype)

    last_side = torch.where(single, side, last_side)
    last_side = torch.where(both, torch.full_like(last_side, -1), last_side)
    event = single | both
    elapsed = torch.where(event, torch.zeros_like(elapsed), elapsed)
    last_position = torch.where(event[:, None], position, last_position)
    state = dict(inactive_age=inactive_age.detach().clone(), last_side=last_side.detach().clone(), elapsed=elapsed.detach().clone(),
                 last_position=last_position.detach().clone())
    return cost, state
