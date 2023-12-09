from itertools import product
import json
import os
import os.path as op
from pathlib import Path
import shutil
from typing import Any, Generator, List, Dict, Union

import numpy as np
from tqdm import tqdm


class NpEncoder(json.JSONEncoder):
    """A JSONEncoder subclass to handle Numpy integers, floats and arrays when writing JSON lines to disk.

    Usage: json.dumps(data, cls=NpEncoder)

    This function overwrites the default() method of JSONEncoder to handle additional types; specifically Numpy
    integers, floats and arrays. For all other types, the standard default() method is used for encoding.
    """

    def default(
        self, obj: Union[np.integer, np.floating, np.ndarray, Any]
    ) -> Union[int, float, List[Any], Any]:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def json_lines_loader(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads the JSON lines located at filepath and returns them as a list of flat dictionaries."""

    jsonl = []
    with open(filepath) as f:
        for line in tqdm(f):
            jsonl.append(json.loads(line))

    return jsonl


def empty_folder(folder: Union[str, Path]) -> None:
    """Deletes the contents of `folder`"""

    for filename in os.listdir(folder):
        file_path = op.join(folder, filename)
        try:
            if op.isfile(file_path) or op.islink(file_path):
                os.unlink(file_path)
            elif op.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def gen_combinations(
    d: Dict[str, Union[List, Any]]
) -> Generator[Dict[str, Any], None, None]:
    """Takes a dictionary of lists and returns a generator that yields all possible combinations of the lists.
    This is especially useful for hyperparameter tuning.
    """
    keys, values = d.keys(), d.values()

    list_keys = [k for k in keys if isinstance(d[k], list)]
    nonlist_keys = [k for k in keys if k not in list_keys]
    list_values = [v for v in values if isinstance(v, list)]
    nonlist_values = [v for v in values if v not in list_values]

    combinations = product(*list_values)

    for c in combinations:
        result = dict(zip(list_keys, c))
        result.update({k: v for k, v in zip(nonlist_keys, nonlist_values)})
        yield result
