"""
Detector-specific transforms.

PDGToSemantic: derives semantic labels from PDG codes in seg data when
no label (labl) file is available. For production training with labels,
datasets load labels from the labl file directly.
"""

import numpy as np

from .transform import TRANSFORMS
from .utils.pdg import pdg_to_semantic


@TRANSFORMS.register_module()
class PDGToSemantic:
    """Fallback: derive approximate semantic labels from PDG codes.

    Schemes
    -------
    motif_5cls : shower(0), track(1), michel(2), delta(3), led(4)
    pid_6cls : photon(0), electron(1), muon(2), pion(3), proton(4), other(5)
    custom : user-provided {pdg_code: class_index} dict
    """

    def __init__(self, scheme='motif_5cls', custom_map=None):
        if scheme not in ('motif_5cls', 'pid_6cls', 'custom', 'none'):
            raise ValueError(f"Unknown label scheme: {scheme}")
        if scheme == 'custom':
            assert custom_map is not None
        self.scheme = scheme
        self.custom_map = custom_map

    def __call__(self, data_dict):
        if self.scheme == 'none' or 'pdg' not in data_dict:
            return data_dict

        if 'segment' in data_dict or 'segment_motif' in data_dict:
            return data_dict

        pdg = data_dict['pdg']
        labels = pdg_to_semantic(pdg, scheme=self.scheme,
                                 custom_map=self.custom_map)
        data_dict['segment_motif'] = labels[:, None]

        if self.scheme == 'motif_5cls':
            pid = pdg_to_semantic(pdg, scheme='pid_6cls')
            data_dict['segment_pid'] = pid[:, None]
        elif self.scheme == 'pid_6cls':
            data_dict['segment_pid'] = labels[:, None]

        n = len(labels)

        if 'instance_particle' not in data_dict and 'track_ids' in data_dict:
            track_ids = data_dict['track_ids']
            mask = track_ids >= 0
            if mask.any():
                _, inverse = np.unique(track_ids[mask], return_inverse=True)
                out = np.full(n, -1, dtype=np.int32)
                out[mask] = inverse
                data_dict['instance_particle'] = out[:, None]
            else:
                data_dict['instance_particle'] = np.full((n, 1), -1, dtype=np.int32)

        if 'instance_interaction' not in data_dict and 'interaction_ids' in data_dict:
            iids = data_dict['interaction_ids']
            mask = iids >= 0
            if mask.any():
                _, inverse = np.unique(iids[mask], return_inverse=True)
                out = np.full(n, -1, dtype=np.int32)
                out[mask] = inverse
                data_dict['instance_interaction'] = out[:, None]
            else:
                data_dict['instance_interaction'] = np.full((n, 1), -1, dtype=np.int32)

            data_dict['segment_interaction'] = (iids[:, None] != -1).astype(np.int32)

        return data_dict
