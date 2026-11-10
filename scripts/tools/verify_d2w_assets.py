"""Validate the external D2W runtime package without starting simulation.

Usage: python scripts/tools/verify_d2w_assets.py
"""
from d2w_rl_lab.assets.d2w_metadata import get_d2w_asset_identity

if __name__ == '__main__':
    identity = get_d2w_asset_identity()
    print(f'D2W canonical SHA-256: {identity.canonical_asset_sha256}')
    print(f'USD: {identity.isaac_usd_path}')
    print(f'MJCF: {identity.mujoco_xml_path}')
    print('Recorded asset hashes and receipt agree; this is not a new physical validation.')
