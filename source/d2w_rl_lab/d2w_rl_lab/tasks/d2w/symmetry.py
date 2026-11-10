"""D2W sagittal reflection using the upstream augmentation mechanism."""
import math
import torch
from d2w_rl_lab.utils.symmetry import ReferenceSymmetry, ReferenceSymmetryMapper

class D2WSymmetryMapper(ReferenceSymmetryMapper):
    JOINT_PERM = [4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11]
    JOINT_SIGN = [-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1]

    def reverse_obs_group(self, group_name, value):
        manager = self.base_env.observation_manager
        widths = [math.prod(d) for d in manager._group_obs_term_dim[group_name]]
        if sum(widths) != value.shape[-1]:
            raise ValueError(f'D2W symmetry group {group_name} layout mismatch')
        return super().reverse_obs_group(group_name, value)

    def reverse_term(self, name, cfg, value):
        if name not in self.VECTOR_SIGNS and name not in self.JOINT_TERMS and name not in {'contact_force','height_scan'}:
            raise ValueError(f'No D2W symmetry mapping for {name}')
        return super().reverse_term(name, cfg, value)

class D2WSymmetry(ReferenceSymmetry):
    def __init__(self, env):
        self.mapper = D2WSymmetryMapper(env)
        self.num_aug = 2
