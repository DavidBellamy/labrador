from typing import Dict, Iterable, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from tqdm import tqdm


class mimic4_eCDFer:
    def __init__(self, ecdf_data: np.lib.npyio.NpzFile) -> None:
        """
        Maps an iterable of lab codes and and an iterable of corresponding lab values to their probabilities on the corresponding eCDF.

        Parameters:
        ecdf_data: a NumPy .npz data archive containing named arrays: {itemid}_x and {itemid}_y for all itemid's in MIMIC-IV.
            {itemid}_x contains the *unique* values of the random variable (e.g. lab values).
            {itemid}_y contains the probabilities corresponding to P(X <= x) for that itemid.

        Note: {itemid}_x, {itemid}_y are index-aligned such that:
            ecdf_data[f"{itemid}_y"][i] = P(X <= ecdf_data[f"{itemid}_x"][i]) for all i.
        """

        self.ecdf_data = ecdf_data
        self.itemids = list(set([int(itemid[:-2]) for itemid in ecdf_data.files]))

    def __call__(
        self,
        itemids: Union[Iterable[int], NDArray[np.int_]],
        lab_values: Union[Iterable[float], NDArray[np.float_]],
        null_token: Union[int, float] = np.nan,
    ) -> NDArray[np.float_]:
        """
        Returns Pr(X <= x) for all x in lab_values.
          i.e. maps all values in lab_values to their probabilities on the eCDF of the corresponding itemid

        itemids: an iterable of integer lab codes (called itemid's in MIMIC-IV).
            Missing values are not allowed because they are used to index into the eCDF database.
        lab_values: an iterable of float lab values.
            Missing values are allowed and will be mapped to null_token.
        null_token: the token to use for missing values. Default is np.nan.

        Returns an array of probabilities corresponding to the input lab_values.
        """

        assert len(itemids) == len(
            lab_values
        ), "itemids and lab_values must be the same length"

        # Find the indices of the nearest values in the compressed eCDF cut-off points
        ixs = [
            self.find_nearest_ecdf_cutoff(itemid, labval)
            for itemid, labval in zip(itemids, lab_values)
        ]

        # Return the corresponding eCDF probabilities
        return np.array(
            [
                self.ecdf_data[f"{itemid}_y"][ix].item()
                if ix is not None
                else null_token
                for itemid, ix in zip(itemids, ixs)
            ]
        )

    def find_nearest_ecdf_cutoff(
        self, itemid: int, lab_value: float
    ) -> Union[int, None]:
        """
        Finds the nearest value to `lab_value` in the eCDF for `itemid`.
        Returns the index of this nearest value or None if the lab_value is missing.
        """
        if np.isnan(lab_value):
            idx = None
        else:
            lab_value = np.array(lab_value)
            idx = (
                np.abs(self.ecdf_data[f"{itemid}_x"] - lab_value.reshape(-1, 1))
            ).argmin(axis=1)

        return idx

    def __len__(self):
        return len(self.itemids)


def map_lab_values_to_eCDF_values(
    df: pd.DataFrame, ecdfs: np.lib.npyio.NpzFile
) -> pd.DataFrame:
    """
    Maps a dataframe of lab values to a dataframe of probabilities, where each
    probability is the corresponding output of the eCDF for the lab test specifid
    in the column name and the associated lab value.

    This prepares a dataframe of lab values for either Labrador or BERT tokenization.
    """
    eCDFer = mimic4_eCDFer(ecdfs)

    # Assert that all digit column names in df (e.g. `1023`) exist in the eCDF database
    assert all(
        [int(col) in eCDFer.itemids for col in df.columns if col.isdigit()]
    ), "All digit column names in df must exist in the eCDF database"

    # Ensure column names that represent MIMIC itemid's are strings
    mimic_lab_cols = [str(col) for col in df.columns if col.isdigit()]

    # Map lab values to probabilities on the eCDFs
    for label, content in tqdm(
        df[mimic_lab_cols].items(),
        desc="Mapping lab values to eCDF probabilities",
        total=df[mimic_lab_cols].shape[1],
    ):
        df[label] = eCDFer(np.full(len(content), int(label)), content.to_numpy())

    return df


def make_labrador_inputs(
    df: pd.DataFrame,
    label_col: Union[str, None],
    codebook: pd.DataFrame,
    null_token: int,
) -> Tuple[Dict[str, np.array], Union[np.array, None], Union[pd.DataFrame, None]]:
    """
    Converts a dataframe of eCDF probabilities (not raw lab values) into Labrador's input format.

    df: dataframe containing feature columns and the label column. Missing values should only appear in digit-named
        columns (i.e. MIMIC-IV labs)
        Assumes that all digit-named columns are MIMIC-IV labs and their values are the eCDF probabilities
    label_col: name of label column
    codebook: a dataframe mapping MIMIC-IV itemid's to their tokens (i.e. frequency rankings)
    null_token: the token to use for missing values in Labrador

    Returns model inputs, labels, and non_mimic_feature_values
    """

    feature_cols = list(df.columns)
    if label_col is not None:
        feature_cols.remove(label_col)
    mimic_lab_cols = [int(col) for col in feature_cols if col.isdigit()]
    mimic_lab_cols_str = [str(col) for col in feature_cols if col.isdigit()]
    non_mimic_lab_cols = [str(col) for col in feature_cols if not col.isdigit()]

    # Fill missing lab values with Labrador's special <null_token>
    df_with_null_tokens = df[mimic_lab_cols_str].fillna(null_token, inplace=False)
    df_with_null_tokens = df_with_null_tokens.join(df[non_mimic_lab_cols])

    # Construct model inputs
    lab_codes = [
        codebook[codebook.itemid == itemid].frequency_rank.item()
        for itemid in mimic_lab_cols
    ]
    continuous_input = df_with_null_tokens[mimic_lab_cols_str].to_numpy()
    categorical_input = np.tile(lab_codes, (len(df_with_null_tokens), 1))
    model_inputs = {
        "categorical_input": categorical_input,
        "continuous_input": continuous_input,
    }

    if label_col is not None:
        labels = df.loc[:, label_col].to_numpy()
        labels = np.expand_dims(labels, axis=-1)
    else:
        labels = None

    # Gather non-mimic-lab features
    non_mimic_lab_cols = [str(col) for col in feature_cols if not col.isdigit()]
    non_mimic_feature_values = (
        None
        if not non_mimic_lab_cols
        else df.loc[:, non_mimic_lab_cols].reset_index(drop=True)
    )

    return model_inputs, labels, non_mimic_feature_values


def make_bert_inputs(
    df: pd.DataFrame,
    label_col: Union[str, None],
    codebook: pd.DataFrame,
    mask_null: bool = False,
    mask_token: Union[int, None] = None,
) -> Tuple[Dict[str, np.array], Union[np.array, None], Union[pd.DataFrame, None]]:
    """
    Converts a dataframe of eCDF probabilities (not raw lab values) into BERT's input format.

    df: dataframe containing feature columns and the label column.
        Missing values should only appear in digit-named columns (i.e. MIMIC-IV labs)
        Assumes that all digit-named columns are MIMIC-IV labs and their values are the eCDF probabilities
    label_col: name of label column
    codebook: a dataframe mapping MIMIC-IV itemid's to their tokens
    mask_null: whether to mask lab values when they are missing
    mask_token: the token to use for missing values for BERT
    """
    feature_cols = list(df.columns)
    if label_col is not None:
        feature_cols.remove(label_col)
    mimic_lab_cols = [str(col) for col in feature_cols if col.isdigit()]

    # Map eCDF values to BERT tokens
    for col in tqdm(df[mimic_lab_cols].columns, total=df[mimic_lab_cols].shape[1]):
        for row, value in enumerate(df[col]):
            if np.isnan(value) and not mask_null:
                try:
                    df.at[row, col] = codebook[
                        (codebook.itemid == int(col)) & (np.isnan(codebook.valuenum))
                    ].token.values[0]
                except:
                    print(f"1: Could not find token for {col} and {value}")
            elif np.isnan(value) and mask_null:
                df.at[row, col] = mask_token
            else:
                try:
                    df.at[row, col] = codebook[
                        (codebook.itemid == int(col)) & (value <= codebook.valuenum)
                    ].token.min()
                except:
                    print(f"2: Could not find token for {col} and {value}")

    # Assert no missingness in df
    assert (
        df[mimic_lab_cols].isna().sum().sum() == 0
    ), "No missing values should exist in the MIMIC features"

    # Construct model inputs
    input_ids = df[mimic_lab_cols].to_numpy(dtype=np.int32)
    model_inputs = {"input_ids": input_ids}

    if label_col is not None:
        labels = df.loc[:, label_col].to_numpy()
        labels = np.expand_dims(labels, axis=-1)
    else:
        labels = None

    # organizing + saving remaining columns to recreate dataframe with embeddings
    non_mimic_lab_cols = [str(col) for col in feature_cols if not col.isdigit()]
    non_mimic_feature_values = (
        None
        if not non_mimic_lab_cols
        else df.loc[:, non_mimic_lab_cols].reset_index(drop=True)
    )

    return model_inputs, labels, non_mimic_feature_values
