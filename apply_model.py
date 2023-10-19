import os
import sys
import h5py
import json
import numpy as np
import torch as pt
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

from src.dataset import StructuresDataset, collate_batch_features, select_by_sid, select_by_interface_types
from src.data_encoding import encode_structure, encode_features, extract_topology, categ_to_resnames, resname_to_categ
from src.structure import data_to_structure, encode_bfactor, concatenate_chains, split_by_chain
from src.structure_io import save_pdb, read_pdb
from src.scoring import bc_scoring, bc_score_names

from model.config import config_model, config_data
from model.data_handler import Dataset
from model.model import Model
from pathlib import Path

import logging

logger = logging.getLogger('PeSTo main script')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    "%Y-%m-%d %H:%M:%S")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


def log_error(error_log: Path, error_type: str,  message: str) -> None:
    with error_log.open("a") as errlog:
        errlog.write(f"[{error_type}] {message}\n")


def check_processed(filepath: str, output_dir: Path) -> bool:
    if len(list(output_dir.glob(Path(filepath).name + "*"))) == 5:
        return True
    return False


def run(data_path: str, output_dir: Path) -> None:
    # data parameters
    # data_path = "examples/issue_19_04_2023"
    logger.info(f"output will be saved here: {output_dir=}")
    output_dir.mkdir(parents=True, exist_ok=True)

    error_log = output_dir / "error_log.txt"

    # model parameters
    # R3
    # save_path = "model/save/i_v3_0_2021-05-27_14-27"  # 89
    # save_path = "model/save/i_v3_1_2021-05-28_12-40"  # 90
    # R4
    # save_path = "model/save/i_v4_0_2021-09-07_11-20"  # 89
    save_path = "model/save/i_v4_1_2021-09-07_11-21"  # 91

    # select saved model
    model_filepath = os.path.join(save_path, 'model_ckpt.pt')
    # model_filepath = os.path.join(save_path, 'model.pt')

    # add module to path
    if save_path not in sys.path:
        sys.path.insert(0, save_path)

    # define device
    device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

    # create model
    model = Model(config_model)

    # reload model
    model.load_state_dict(pt.load(model_filepath,
                                  map_location=device))

    # set model to inference
    model = model.eval().to(device)

    # find pdb files and ignore already predicted oins
    pdb_filepaths = glob(os.path.join(data_path, "*.pdb"), recursive=True)
    # print(pdb_filepaths)
    # pdb_filepaths = [fp for fp in pdb_filepaths if "_i" not in fp]

    # print(pdb_filepaths)

    # create dataset loader with preprocessing
    dataset = StructuresDataset(pdb_filepaths, with_preprocessing=True)

    # debug print
    print(len(dataset))

    # run model on all subunits
    with pt.no_grad():
        for subunits, filepath in tqdm(dataset):
            fpath = Path(filepath)
            if check_processed(filepath, output_dir):
                logger.info(f"{filepath} already processed")
                continue
            try:
                # concatenate all chains together
                structure = concatenate_chains(subunits)
            except TypeError:
                logger.error(f"Type error on concatenate_chains {filepath=}")
                log_error(error_log, "TypeError", str(filepath))

            # encode structure and features
            X, M = encode_structure(structure)
            # q = pt.cat(encode_features(structure), dim=1)
            q = encode_features(structure)[0]

            # extract topology
            ids_topk, _, _, _, _ = extract_topology(X, 64)

            # pack data and setup sink (IMPORTANT)
            X, ids_topk, q, M = collate_batch_features([[X, ids_topk, q, M]])

            # run model
            z = model(X.to(device),
                      ids_topk.to(device),
                      q.to(device),
                      M.float().to(device))

            # for all predictions
            for i in range(z.shape[1]):
                # prediction
                p = pt.sigmoid(z[:, i])

                # encode result
                structure = encode_bfactor(structure, p.cpu().numpy())

                # save results
                output_filepath = output_dir / f"{fpath.name}-interface_{i}"
                try:
                    save_pdb(split_by_chain(structure), output_filepath)
                except IndexError:
                    logger.error(f"failed to process {filepath=}")
                    log_error(error_log, "IndexError", str(filepath))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reads Yulab's PDBs and prepares the files for BIPSPI+")
    parser.add_argument("-i", "--input-dir", required=True,
                        help="A directory containing PDF files")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="A path to write the renamed PDB files")
    args = parser.parse_args()
    run(args.input_dir, Path(args.output_dir))
