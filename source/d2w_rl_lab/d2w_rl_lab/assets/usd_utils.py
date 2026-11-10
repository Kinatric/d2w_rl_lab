"""Shared visual branch utility, derived from wty-yy/d2w_rl_lab (MIT)."""
from __future__ import annotations

def _deactivate_visual_branches(root_prim: Usd.Prim) -> int:
    """Deactivate normalized robot ``visuals`` branches before environment replication.
    We found that this can very effectively reduce cuda memory usage.
    """
    from pxr import Usd

    stage = root_prim.GetStage()
    visual_paths = [prim.GetPath() for prim in Usd.PrimRange(root_prim) if prim.GetName() == "visuals"]
    for path in visual_paths:
        stage.GetPrimAtPath(path).SetActive(False)
    return len(visual_paths)
