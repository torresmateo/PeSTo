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

# data parameters
data_path = "examples/issue_19_04_2023"

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

# load functions


# define device
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

# create model
model = Model(config_model)

# reload model
model.load_state_dict(pt.load(model_filepath, map_location=pt.device("cpu")))

# set model to inference
model = model.eval().to(device)

# find pdb files and ignore already predicted oins
pdb_filepaths = glob(os.path.join(data_path, "*.pdb"), recursive=True)
pdb_filepaths = [fp for fp in pdb_filepaths if "_i" not in fp]

# create dataset loader with preprocessing
dataset = StructuresDataset(pdb_filepaths, with_preprocessing=True)

# debug print
print(len(dataset))

# run model on all subunits
with pt.no_grad():
    for subunits, filepath in tqdm(dataset):
        logger.info(f"{subunits=} {filepath=}")
        # concatenate all chains together
        structure = concatenate_chains(subunits)

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
            output_filepath = filepath[:-4]+'_i{}.pdb'.format(i)
            save_pdb(split_by_chain(structure), output_filepath)
