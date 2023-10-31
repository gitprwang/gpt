import torch
import os
from util.dataloader import *

zinc_data_path = '/home/data/wd/zinc_data.txt'
uspto_data_path = '/home/data/wd/USPTO/uspto_data.txt'
rxn_data_path = '/home/data/wd/rxn.txt'

# generate_reaction_json([rxn_data_path], '/home/data/wd/rxn_reaction.json')

generate_smiles_json([zinc_data_path], '/home/data/wd/zinc_smiles/smiles', '/home/data/wd/zinc.txt') 