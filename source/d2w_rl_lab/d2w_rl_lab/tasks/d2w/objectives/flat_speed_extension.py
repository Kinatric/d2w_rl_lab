"""D2W flat forward speed exploration within the unchanged wheel target limit.

2.6 m/s is a training command, not a demonstrated hardware operating speed.
At radius 0.0813 m it needs 31.98 rad/s, below the existing 36 rad/s target
clip. Available loaded speed still depends on drag, traction and leg motion.
"""

MAX_FORWARD_M_S = 2.6
PREVIOUS_FORWARD_M_S = 2.0


def configure(cfg):
    command = cfg.commands.base_velocity
    assert tuple(command.ranges.lin_vel_x) == (-2., 2.)
    assert not command.command_range_curriculum
    command.ranges.lin_vel_x = (-2., MAX_FORWARD_M_S)
    for terrain, limits in command.terrain_max_command_ranges.items():
        # Preserve the PREVIOUS effective intersection on every other terrain;
        # merely changing the global limit would also enlarge some rough caps.
        lower, upper = limits.get('lin_vel_x', (-2., 2.))
        limits['lin_vel_x'] = (max(-2., lower),
                              MAX_FORWARD_M_S if terrain == 'flat' else min(2., upper))
    assert 'flat' in command.terrain_max_command_ranges
    matches = [rule for rule in command.command_clip_rules
               if rule['command'] == 'lin_vel_x' and rule['limit'] == 2.]
    assert len(matches) == 1
    # Speeds above the old forward bound are exclusively straight motion.
    # This avoids combining the added rolling speed with high lateral/yaw demand.
    matches[0]['clip']['lin_vel_y'] = (0., 0.)
    matches[0]['clip']['ang_vel_yaw'] = (0., 0.)
    return cfg
