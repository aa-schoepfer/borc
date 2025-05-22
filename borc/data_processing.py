import json
import logging
import itertools
import warnings
import pandas as pd
import numpy as np

from sklearn.utils import check_X_y, check_array

from rdkit import Chem
from rdkit.Chem import AllChem

MF_FINGERPRINT_RADIUS = 4
MF_BIT_SIZE = 256

logger = logging.getLogger(__name__)


class OptimizationSettings:
    def __init__(self, json_file):
        data = json.load(open(json_file))

        self.csv_file = data["csvFileName"]
        self.molecular_col = data["columns"]["molecules"]
        self.tabular_col = data["columns"]["conditions"]
        self.property_col = data["columns"]["properties"]
        try:
            self.batch_size = data["batchSize"]
        except KeyError:
            self.batch_size = 1

        self._add = data["_add"]
        self._remove = data["_remove"]

    def display_settings(self):
        return (
            f"CSV File: {self.csv_file}\n"
            f"Molecular columns: {self.molecular_col}\n"
            f"Tabular columns: {self.tabular_col}\n"
            f"Property columns: {self.property_col}\n"
            f"Batch size: {self.batch_size}"
        )


def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=MF_FINGERPRINT_RADIUS, nBits=MF_BIT_SIZE
    )
    return {smiles: np.asarray(fingerprint)}

def fingerprint_from_dict(smiles, fingerprint_dict):
    if smiles in fingerprint_dict:
        return fingerprint_dict[smiles]
    else:
        raise ValueError(f"SMILES string not found in dictionary: {smiles}")

def recover_smiles_from_fingerprint(fingerprint, smiles_dict):
    all_smiles = []
    for smiles, fp in smiles_dict.items():
        if np.array_equal(fingerprint, fp):
            all_smiles.append(smiles)
    
    if len(all_smiles) > 0:
        return all_smiles
    
    raise ValueError("Fingerprint not found in dictionary")

def load_data(csv_file, molecular_cols, tabular_cols, property_cols):
    data = pd.read_csv(csv_file)
    molecular_data = data[molecular_cols]
    tabular_data = data[tabular_cols]
    property_data = data[property_cols]
    return molecular_data, tabular_data, property_data


def process_data(settings):

    raw_data = load_data(
        settings.csv_file,
        settings.molecular_col,
        settings.tabular_col,
        settings.property_col,
    )

    logger.debug("Data loaded\n" + f"Shapes: {[d.shape for d in raw_data]}")

    list_of_dicts = raw_data[0].map(smiles_to_fingerprint).values.ravel()
    
    # Merge list of dicts into a single dict
    smiles_mf_dict = {}
    for d in list_of_dicts:
        if any(k in smiles_mf_dict for k in d.keys()):
            warnings.warn(
                f"Duplicate SMILES string found: {d}. Overwriting existing fingerprint."
            )
        smiles_mf_dict.update(d)

    molecular_fingerprints = raw_data[0].map(lambda x: fingerprint_from_dict(x, smiles_mf_dict)).values

    X_tabular = raw_data[1].map(lambda x: np.array([x])).values

    logger.debug(f"molecular_fingerprints.shape: {molecular_fingerprints.shape}")
    logger.debug(f"X_tabular.shape: {X_tabular.shape}")

    X_raw = np.concatenate((molecular_fingerprints, X_tabular), axis=1)
    y_raw = raw_data[2].values

    logger.debug(f"X_raw shape: {X_raw.shape}")
    logger.debug(f"y_raw shape: {y_raw.shape}")

    X_columns = settings.molecular_col + settings.tabular_col
    y_columns = settings.property_col

    index_ranges = []
    start = 0
    for col in settings.molecular_col:
        end = start + MF_BIT_SIZE
        index_ranges.append(list(range(start, end)))
        start = end
    for col in settings.tabular_col:
        end = start + 1
        index_ranges.append([start])
        start = end

    logger.debug(f"Feature columns: {X_columns}")
    logger.debug(f"Target columns: {y_columns}")
    logger.debug(f"Index ranges: {index_ranges}")

    return X_raw, y_raw, X_columns, y_columns, index_ranges, smiles_mf_dict


def build_X_y_representation(X, y):

    X = np.array([np.concatenate(x) for x in X])
    X, y = check_X_y(X, y, multi_output=True)

    logger.debug(f"Final X shape: {X.shape}")
    logger.debug(f"Final y shape: {y.shape}")

    return X, y

def build_X_representation(X):
    X = np.array([np.concatenate(x) for x in X])
    X = check_array(X)

    logger.debug(f"Final X shape: {X.shape}")

    return X

def build_combination_space(X, X_columns, settings, smiles_mf_dict):

    unique_params = []

    for i, col in enumerate(X_columns):
        X_col = X[:, i]
        if hasattr(settings, "_add") and col in settings._add:
            logger.debug(f"Adding {col} to combination space")
            if col in settings.molecular_col:
                
                list_of_dicts = list(map(smiles_to_fingerprint, settings._add[col]))

                for d in list_of_dicts:
                    if any(k in smiles_mf_dict for k in d.keys()):
                        warnings.warn(
                            f"Duplicate SMILES string found: {d}. Overwriting existing fingerprint."
                        )
                    smiles_mf_dict.update(d)

                additional_parameters = list(
                    map(lambda x: fingerprint_from_dict(x, smiles_mf_dict), settings._add[col])
                )
                logger.debug(
                    f"Additional parameters for {col}: {additional_parameters}"
                )
            else:
                additional_parameters = list(
                    map(lambda x: np.asarray([x]), settings._add[col])
                )
                logger.debug(
                    f"Additional parameters for {col}: {additional_parameters}"
                )

            X_col = np.concatenate(
                (
                    np.array([np.asarray(x) for x in X_col]),
                    np.asarray(additional_parameters),
                )
            )

            logger.debug(f"Unique values for {col}: {len(np.unique(X_col, axis=0))}")

        unique_params.append(
            np.unique(np.array([np.asarray(x) for x in X_col]), axis=0)
        )

    logger.debug(f"Unique parameters before removal: {unique_params}")


    # Remove parameters specified in settings._remove from unique_params
    for i, col in enumerate(X_columns):
        if hasattr(settings, "_remove") and col in settings._remove:            
            to_remove = settings._remove[col]
            # Remove matching entries from unique_params[i]
            if col in settings.molecular_col:
                # For molecular columns, remove fingerprints matching those SMILES
                remove_fps = [fingerprint_from_dict(sm, smiles_mf_dict) for sm in to_remove]
                mask = [
                    not any(np.array_equal(u, rfp) for rfp in remove_fps)
                    for u in unique_params[i]
                ]
                unique_params[i] = unique_params[i][mask]
            else:
                # For tabular columns, remove values directly
                mask = [
                    u not in to_remove
                    for u in unique_params[i]
                ]
                unique_params[i] = unique_params[i][mask]

    comb_space = list(itertools.product(*unique_params))
    logger.debug(f"Unique parameters after removal: {unique_params}")
    for i, col in enumerate(X_columns):
        logger.debug(f"Number of unique values for {col}: {len(unique_params[i])}")
    
    logger.debug(f"Combinatorial space size: {len(comb_space)}")
    

    X_comb = build_X_representation(comb_space)

    return X_comb, smiles_mf_dict

def unbuild_representation(X, index_ranges):
    X_unbuilt = []
    for i, x in enumerate(X):
        x_unbuilt = []
        for j, index_range in enumerate(index_ranges):
            x_unbuilt.append(x[index_range])
        X_unbuilt.append(x_unbuilt)
    
    X_unbuilt = np.array(X_unbuilt, dtype=object)
    
    logger.debug(f"Unbuilt X shape: {X_unbuilt.shape}")
    logger.debug(f"Unbuilt X: {X_unbuilt}")

    return X_unbuilt 

def print_candidates(candidates, cands_y, molecular_col, tabular_col, smiles_mf_dict):
    for i, candidate in enumerate(candidates):
        print(f"Candidate {i + 1}:")
        print(f"  Properties: {cands_y[0][i]}, {cands_y[1][i]}")
        for j, col in enumerate(molecular_col):
            print(f"  {col}: { recover_smiles_from_fingerprint(candidate[j], smiles_mf_dict) }")
        for j, col in enumerate(tabular_col):
            print(f"  {col}: {candidate[len(molecular_col) + j]}")
        
