"""
Helper file for handling MD trajectories.

Adapt the __main__ block to pre-compute all pairwise distances.
"""

import logging
import mdtraj
from mdtraj.core.trajectory import Trajectory
import pandas as pd
import numpy as np
from itertools import combinations
from os import PathLike


def load_traj_metadata(protein_id: str, base_path: PathLike) -> dict:
    _suffix = "-0-protein"
    fname = protein_id + _suffix
    path_prefix = "DESRES-Trajectory_"

    data_path = base_path + "/" + path_prefix + fname + "/" + fname + "/"
    topology = data_path + fname + ".pdb"
    times = pd.read_csv(data_path + fname + "_times.csv", header=None, names=["time", "traj_file"])
    metadata = {
        'topology_path': topology,
        'times': times,
        'trajectory_files': [data_path + traj_file for traj_file in times['traj_file'].to_list()]
    }
    logging.info(f"The trajectory is {10_000 * len(metadata['trajectory_files'])} frames long")
    return metadata


def compute_pwise_distances(traj: Trajectory) -> np.ndarray:
    sel = traj.top.select("symbol != H")
    atom_pairs = list(combinations(sel, 2))
    return mdtraj.compute_distances(traj, atom_pairs)


def pairwise_distance_generator(metadata: dict):
    for traj_file in metadata['trajectory_files']:
        traj = mdtraj.load(traj_file, top=metadata['topology_path'])
        yield compute_pwise_distances(traj)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    base_path = "/the/path/to/DESRES_folding_trajs"
    metadata = load_traj_metadata("2JOF", base_path)
    for i, pwise_distances in enumerate(pairwise_distance_generator(metadata)):
        print(f"Shape of pairwise distance at {i + 1}-th iteration is {pwise_distances.shape}")
        if i == 4:
            break
