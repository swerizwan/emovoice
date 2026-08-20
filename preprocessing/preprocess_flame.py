"""
FLAME facial representation preprocessing.

This script converts facial motion sequences into the
unified FLAME expression representation used by the
Voice-Driven Emotion Separation framework.

Processing steps:
1. Load facial motion data
2. Convert mesh parameters to FLAME representation
3. Extract 53-dimensional expression coefficients
4. Normalize temporal length
5. Fix identity and pose parameters
6. Save processed FLAME expression sequences

import os
import json
import pickle
import argparse
import logging
import numpy as np

from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor


# ==========================================================
# Configuration
# ==========================================================

FLAME_EXPR_DIM = 53

TARGET_FPS = 60

MIN_FRAMES = 10

SUPPORTED_EXTENSIONS = [
    ".npy",
    ".npz",
    ".pkl"
]


# ==========================================================
# Logger
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format=
    "[%(asctime)s] %(levelname)s: %(message)s"
)

logger = logging.getLogger(
    "FLAME_Preprocessing"
)



# ==========================================================
# File Loading
# ==========================================================

def load_file(path):

    """
    Load different facial parameter formats.

    Supported:
        npy
        npz
        pickle

    Returns:
        dictionary containing:
            expression
            pose
            shape
            vertices
    """


    suffix = Path(path).suffix


    if suffix == ".npy":

        data = np.load(
            path,
            allow_pickle=True
        )


        if isinstance(
            data,
            np.ndarray
        ):

            return {
                "expression": data
            }


        return data.item()



    elif suffix == ".npz":

        data = np.load(
            path,
            allow_pickle=True
        )


        return {
            key:data[key]
            for key in data.files
        }



    elif suffix == ".pkl":

        with open(
            path,
            "rb"
        ) as f:

            return pickle.load(f)



    else:

        raise RuntimeError(
            f"Unsupported file: {path}"
        )



# ==========================================================
# FLAME Parameter Extraction
# ==========================================================

def extract_flame_expression(
        params):

    """
    Extract FLAME expression coefficients.

    FLAME parameters:

        shape:
            identity

        pose:
            jaw/head rotation

        expression:
            dynamic facial deformation


    Only expression coefficients are used.
    """


    possible_keys = [
        "expression",
        "expr",
        "exp",
        "flame_expression"
    ]


    expression = None


    for key in possible_keys:

        if key in params:

            expression = params[key]

            break



    if expression is None:

        raise ValueError(
            "FLAME expression parameter missing"
        )


    expression = np.asarray(
        expression
    )


    if expression.ndim == 1:

        expression = expression.reshape(
            1,
            -1
        )



    if expression.shape[1] < FLAME_EXPR_DIM:

        raise ValueError(
            "Expression dimension smaller than 53"
        )


    expression = (
        expression[:, :FLAME_EXPR_DIM]
    )


    return expression.astype(
        np.float32
    )



# ==========================================================
# Pose and Identity Processing
# ==========================================================

def remove_identity_information(
        params):

    """
    Remove identity-dependent FLAME parameters.

    Identity parameters are not optimized.
    Only expression deformation is retained.
    """


    identity = None
    pose = None


    for key in [
        "shape",
        "identity",
        "betas"
    ]:

        if key in params:

            identity = params[key]

            break


    for key in [
        "pose",
        "rotation",
        "jaw_pose"
    ]:

        if key in params:

            pose = params[key]

            break


    return identity, pose



# ==========================================================
# Temporal Processing
# ==========================================================

def resample_sequence(
        sequence,
        original_fps,
        target_fps=TARGET_FPS):


    """
    Temporal interpolation to 60 FPS.
    """


    if original_fps == target_fps:

        return sequence



    old_time = np.arange(
        len(sequence)
    ) / original_fps


    new_length = int(
        len(sequence)
        *
        target_fps
        /
        original_fps
    )


    new_time = np.linspace(
        0,
        old_time[-1],
        new_length
    )


    interpolator = interp1d(
        old_time,
        sequence,
        axis=0,
        kind="linear"
    )


    return interpolator(
        new_time
    ).astype(
        np.float32
    )



# ==========================================================
# Sequence Normalization
# ==========================================================

def normalize_sequence(
        sequence):


    """
    Normalize each expression dimension.
    """


    mean = np.mean(
        sequence,
        axis=0
    )


    std = np.std(
        sequence,
        axis=0
    )


    normalized = (
        sequence - mean
    ) / (
        std + 1e-8
    )


    return (
        normalized,
        mean,
        std
    )



# ==========================================================
# Sequence Validation
# ==========================================================

def validate_sequence(
        sequence):


    if len(sequence) < MIN_FRAMES:

        return False


    if np.isnan(
        sequence
    ).any():

        return False


    if np.isinf(
        sequence
    ).any():

        return False


    return True



# ==========================================================
# Process One Sample
# ==========================================================

def process_single_file(
        input_path,
        output_dir,
        fps):


    try:

        params = load_file(
            input_path
        )


        expression = (
            extract_flame_expression(
                params
            )
        )


        if not validate_sequence(
            expression
        ):

            logger.warning(
                f"Invalid sequence: {input_path}"
            )

            return None



        expression = resample_sequence(
            expression,
            fps,
            TARGET_FPS
        )


        normalized, mean, std = (
            normalize_sequence(
                expression
            )
        )


        name = Path(
            input_path
        ).stem


        save_path = os.path.join(
            output_dir,
            name + ".npz"
        )


        np.savez(
            save_path,
            expression=normalized,
            original_expression=expression,
            mean=mean,
            std=std,
            fps=TARGET_FPS
        )


        return {
            "file":name,
            "frames":len(expression),
            "dimension":
                expression.shape[1]
        }


    except Exception as e:

        logger.error(
            f"{input_path}: {e}"
        )

        return None



# ==========================================================
# Dataset Scanner
# ==========================================================

def collect_files(
        root):


    files=[]


    for path,_,names in os.walk(
        root
    ):

        for name in names:

            if Path(name).suffix in SUPPORTED_EXTENSIONS:

                files.append(
                    os.path.join(
                        path,
                        name
                    )
                )


    return files



# ==========================================================
# Dataset Processing
# ==========================================================

def preprocess_dataset(
        input_dir,
        output_dir,
        fps,
        workers):


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    files = collect_files(
        input_dir
    )


    logger.info(
        f"Found {len(files)} sequences"
    )


    metadata=[]


    worker_args = [
        (
            file,
            output_dir,
            fps
        )
        for file in files
    ]



    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:


        results = list(
            tqdm(
                executor.map(
                    lambda x:
                    process_single_file(
                        *x
                    ),
                    worker_args
                ),
                total=len(files)
            )
        )


    for r in results:

        if r is not None:

            metadata.append(r)



    with open(
        os.path.join(
            output_dir,
            "metadata.json"
        ),
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )


    logger.info(
        "FLAME preprocessing completed."
    )



# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input_dir",
        required=True
    )


    parser.add_argument(
        "--output_dir",
        required=True
    )


    parser.add_argument(
        "--fps",
        default=60,
        type=int
    )


    parser.add_argument(
        "--workers",
        default=8,
        type=int
    )


    args = parser.parse_args()



    preprocess_dataset(
        args.input_dir,
        args.output_dir,
        args.fps,
        args.workers
    )
