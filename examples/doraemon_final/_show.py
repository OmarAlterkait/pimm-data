"""Shared printing for the examples."""
import numpy as np


def event(x):
    for part, sub in x.items():
        if not isinstance(sub, dict):
            continue
        roles = sub.get('_roles', {})
        rows = [(k, v) for k, v in sub.items() if isinstance(v, np.ndarray)]
        print(f'[{part}]  {len(rows[0][1]) if rows else 0} rows')
        for k, v in rows:
            print(f'    {k:22s} {str(v.shape):14s} {str(v.dtype):8s} {roles.get(k, "")}')


def batch(b):
    for k, v in b.items():
        if hasattr(v, 'shape'):
            print(f'    {k:28s} {tuple(v.shape)}  {b.get("_roles", {}).get(k, "")}')
