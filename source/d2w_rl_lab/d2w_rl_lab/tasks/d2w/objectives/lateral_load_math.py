"""D2W load diagnostic and conservative lateral relief math (Torch only).

The reference torque and 2 s filter are shaping scales, NOT winding temperature,
firmware current, or a certified continuous-duty limit. No positive lift reward.
"""
import math
import torch


def smooth(x, low, high):
    t = ((x-low)/(high-low)).clamp(0., 1.)
    return t.square()*(3.-2.*t)


class LoadAccumulator:
    """Integrate DELIVERED torque at every physical actuator computation."""
    def __init__(self, shape, device, reference_nm=8., filter_s=2.):
        if reference_nm <= 0 or filter_s <= 0:
            raise ValueError('Positive load reference and filter time required')
        self.reference_nm, self.filter_s = reference_nm, filter_s
        self.values = {name: torch.zeros(shape, device=device) for name in (
            'elapsed_s', 'torque_sq_integral', 'mechanical_energy_abs_j',
            'above_reference_s', 'continuous_above_s', 'ratio_sq_ema', 'filtered_above_s',
            'interval_torque_sq', 'interval_s', 'interval_power_j', 'interval_above_s')}

    def update(self, delivered, velocity, dt):
        if dt <= 0:
            raise ValueError('Positive physical timestep required')
        sq = delivered.square()
        above = delivered.abs() > self.reference_nm
        v = self.values
        v['elapsed_s'].add_(dt)
        v['torque_sq_integral'].add_(sq*dt)
        v['mechanical_energy_abs_j'].add_((delivered*velocity).abs()*dt)
        v['above_reference_s'].add_(above*dt)
        v['continuous_above_s'].copy_(torch.where(above, v['continuous_above_s']+dt, 0.))
        v['ratio_sq_ema'].lerp_(sq/self.reference_nm**2, -math.expm1(-dt/self.filter_s))
        v['filtered_above_s'].copy_(torch.where(v['ratio_sq_ema'] > 1., v['filtered_above_s']+dt, 0.))
        v['interval_torque_sq'].add_(sq*dt)
        v['interval_s'].add_(dt)
        v['interval_power_j'].add_((delivered*velocity).abs()*dt)
        v['interval_above_s'].add_(above*dt)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in self.values.values():
            value[ids] = 0.

    def consume_interval(self):
        v = self.values
        duration = v['interval_s'].clone()
        ratio_sq = v['interval_torque_sq']/duration.clamp_min(1.e-9)/self.reference_nm**2
        self.last_interval_power_j = v['interval_power_j'].clone()
        self.last_interval_above_s = v['interval_above_s'].clone()
        v['interval_torque_sq'].zero_()
        v['interval_s'].zero_()
        v['interval_power_j'].zero_()
        v['interval_above_s'].zero_()
        return ratio_sq, duration


def support_margins(points_xy, com_xy):
    """Signed COM-to-edge margin of each leave-one-wheel-out triangle.

    All points must be genuine supporting ground contacts. A projected triangle
    is only a quasi-static support proxy, especially on slopes/uneven ground.
    """
    other = torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
                         device=points_xy.device)
    tri = points_xy[:, other]
    edges = tri.roll(-1, dims=2)-tri
    area = (edges[:, :, 0, 0]*(-edges[:, :, 2, 1])
            - edges[:, :, 0, 1]*(-edges[:, :, 2, 0]))
    rel = com_xy[:, None, None]-tri
    cross = edges[..., 0]*rel[..., 1]-edges[..., 1]*rel[..., 0]
    margin = (cross*area.sign()[..., None]/edges.norm(dim=-1).clamp_min(1.e-9)).amin(-1)
    valid = torch.isfinite(tri).all(dim=(-1, -2)) & torch.isfinite(com_xy).all(-1)[:, None]
    valid &= area.abs() > 1.e-4
    return torch.where(valid, margin, torch.full_like(margin, -1.e6))


def lateral_scope(command, upright, gyro_xy, stairs, terminated):
    """Lateral-dominant translation, INCLUDING combined yaw; pure yaw released."""
    lateral = smooth(command[:, 1].abs(), .08, .20)
    lateral *= 1.-smooth(command[:, 0].abs()/command[:, 1].abs().clamp_min(.08), .5, 1.5)
    healthy = smooth(upright, math.cos(.5), math.cos(.25))
    healthy *= 1.-smooth(gyro_xy.norm(dim=-1), .5, 1.5)
    return lateral*healthy*(~stairs)*(~terminated)


def local_twist_progress(command, actual_twist, wheel_xy):
    """Rigid-body twist demand at each support point, not wheel rolling speed.

    vy + yaw_rate*x differs front/rear during lateral+turn commands. Tracking
    the commanded body twist yields exactly one at each nonstationary point.
    Joint swing motion is deliberately not counted as body progress.
    """
    def point_velocity(twist):
        x=twist[:,0,None]-twist[:,2,None]*wheel_xy[...,1]
        y=twist[:,1,None]+twist[:,2,None]*wheel_xy[...,0]
        return torch.stack((x,y),-1)
    desired=point_velocity(command);actual=point_velocity(actual_twist)
    magnitude_sq=desired.square().sum(-1)
    progress=(actual*desired).sum(-1)/magnitude_sq.clamp_min(.08**2)
    return progress,magnitude_sq>.08**2


def relief(command, actual_twist, wheel_xy, gate, load, reference_load, bearing,
           margins, ema, ratio_sq, load_age, valid, old, dt):
    """A mild load cost plus single-leg relief with three stable supports.

    The load cost does not prescribe lifting a particular leg: it also falls
    with reduced sustained effort, including when tracking succeeds. Only the stronger
    single-leg term needs three other supports. No prescribed gait frequency.
    Equal top scores select neither side. A short latch avoids rapid retargeting.
    Single-leg relief vanishes when progress recovers or support becomes unsafe.
    Both costs clear when actual effort drops; neither pays a positive lift reward.
    """
    if dt <= 0:
        raise ValueError('Positive control timestep required')
    valid = valid & torch.isfinite(command).all(-1) & torch.isfinite(actual_twist).all(-1)
    valid &= torch.isfinite(wheel_xy).all(dim=(-1,-2))
    valid &= torch.isfinite(gate) & torch.isfinite(reference_load) & (reference_load > 0.)
    for field in (load, margins, ema, ratio_sq, load_age):
        valid &= torch.isfinite(field).all(-1)
    command, actual_twist, wheel_xy, gate, load, reference_load, margins, ema, ratio_sq, load_age = [
        torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        for x in (command, actual_twist, wheel_xy, gate, load, reference_load, margins, ema, ratio_sq, load_age)]
    reference_load = reference_load.clamp_min(1.e-6)
    n = len(command)
    fresh = old is None
    previous = {} if fresh else old
    support_age = previous.get('support_age', torch.zeros_like(load))
    support_age = torch.where(valid[:, None] & bearing, support_age+dt, 0.).clamp_max(.1)
    stable = bearing & (support_age >= .06-1.e-6)
    others = stable.sum(-1, keepdim=True)-stable.long()
    # Do not permit unloading when the projected COM is outside the remaining
    # three supports. Zero-margin boundary is accepted, not claimed robust.
    can_unload = (others == 3) & (margins >= 0.)
    progress,moving_point = local_twist_progress(command,actual_twist,wheel_xy)
    stalled = (1.-smooth(progress, .4, .85))*moving_point
    loaded = smooth(load, .1*reference_load[:, None], .6*reference_load[:, None])
    excess = (ema-1.).clamp_min(0.)
    stress = excess/(1.+excess)
    legal = valid & (gate > 0.)
    blocked = (ema > 1.) & (ratio_sq > 1.) & bearing & (loaded > 0.) & legal[:, None] & (stalled>0.)
    # Load history belongs to the actuator, not the command/contact gate. A
    # command change or brief recovery does not instantly cool a motor. Reward
    # eligibility still clears immediately; physical episode reset clears load.
    age = load_age
    persistence = smooth(age, .4, 1.2)
    score = stress*loaded*persistence*can_unload*blocked*stalled
    top, candidate = score.max(-1)
    unique = ((score-top[:, None]).abs() < 1.e-6).sum(-1) == 1
    candidate = torch.where((top > 0.) & unique, candidate, -1)
    choice = previous.get('choice', torch.full((n,), -1, device=load.device, dtype=torch.long))
    choice_age = previous.get('choice_age', torch.zeros(n, device=load.device))
    selected_safe = can_unload.gather(1, choice.clamp_min(0)[:, None]).squeeze(1)
    keep = (choice >= 0) & legal & selected_safe & (choice_age < .8)
    choice = torch.where(keep, choice, candidate)
    choice = torch.where(legal, choice, -1)
    choice_age = torch.where(keep, choice_age+dt, 0.)
    mask = torch.arange(4, device=load.device)[None] == choice[:, None]
    per_wheel = gate[:, None]*score*mask
    unload_cost = per_wheel.sum(-1)
    # A pure three-support gate supplies virtually no signal in the reported
    # lateral+opposite-yaw case. Start with a weak all-calf burden cost, so PPO
    # can discover force redistribution BEFORE a leg can safely be relieved.
    # Square positive excess before summing: for otherwise equal conditions,
    # concentrating the same excess into one calf cannot reduce this cost.
    # Both filtered and current effort must remain high; no charge after the
    # actuator unloads. The global monotone normalization bounds the term.
    sustained_excess = (torch.minimum(ema,ratio_sq)-1.).clamp_min(0.)
    # Successful body tracking must not hide persistent high calf effort.
    # This weak term asks for lower effort, not a particular foot movement.
    # Keep progress and three-support/COM gates on the stronger unload term.
    burden_eligible = (ema > 1.) & (ratio_sq > 1.) & bearing & legal[:, None]
    burden_per_wheel = sustained_excess.square()*loaded*persistence*burden_eligible
    burden = burden_per_wheel.mean(-1)
    load_cost = gate*burden/(1.+burden)
    value = .25*load_cost + .75*unload_cost
    state = dict(age=age, support_age=support_age, choice=choice, choice_age=choice_age)
    diagnostic = dict(value=value, per_wheel=per_wheel, gate=gate, stalled=stalled,
                      can_unload=can_unload, blocked=blocked, persistence=persistence,
                      stress=stress, margins=margins, selected=mask,
                      load_cost=load_cost, unload_cost=unload_cost,
                      burden_per_wheel=burden_per_wheel)
    return value, state, diagnostic
