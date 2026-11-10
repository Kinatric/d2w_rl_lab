"""Small independent high-speed stair-command exploration; no gait forcing."""
import torch

def add_probe(original, result, stairs, focused, draw, speed_draw):
    eligible=stairs & (original.abs().amax(-1)>1.e-6) & (~focused)
    selected=eligible & (draw < .02)
    new=result.clone()
    new[:,0]=torch.where(selected, 1.+speed_draw, new[:,0])
    new[:,1:]=torch.where(selected[:,None], torch.zeros_like(new[:,1:]), new[:,1:])
    return new,selected
