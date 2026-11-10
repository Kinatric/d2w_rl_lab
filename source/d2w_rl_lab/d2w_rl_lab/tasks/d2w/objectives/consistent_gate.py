import torch

def designated_flat(env):
    if getattr(env.cfg, 'd2w_upstream_profile', False):
        from d2w_rl_lab.utils.terrains import get_terrain_masks
        if env.scene.terrain.cfg.terrain_type == 'plane':
            return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        return get_terrain_masks(env).get('flat', torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    term=env.command_manager.get_term('base_velocity')
    index=term.terrain_type2idx.get('flat')
    assert index is not None
    return term.terrain_idxs==index

def span_excess(span,nominal):
    # A geometric envelope permits balance adjustments; not an exact wheelbase
    # setpoint or a runtime action/termination limit.
    return (.85*nominal-span).clamp_min(0)+(span-1.15*nominal).clamp_min(0)
