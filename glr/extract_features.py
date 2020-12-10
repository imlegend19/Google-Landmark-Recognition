import os
import pickle

from tqdm import tqdm
import numpy as np
import tensorflow as tf

from glr import (DELG_IMAGE_SCALES_TENSOR, DELG_SCORE_THRESHOLD_TENSOR, GLOBAL_FEATURE_EXTRACTION_FN,
                 LOCAL_FEATURE_EXTRACTION_FN, LOCAL_FEATURE_NUM_TENSOR, MAX_NUM_EMBEDDINGS, NUM_EMBEDDING_DIMENSIONS,
                 PATH_ID)
from glr.helpers import load_image_tensor


def extract_global_features(image_paths):
    num_embeddings = len(image_paths)
    if MAX_NUM_EMBEDDINGS > 0:
        num_embeddings = min(MAX_NUM_EMBEDDINGS, num_embeddings)

    ids = num_embeddings * [None]
    embeddings = np.empty((num_embeddings, NUM_EMBEDDING_DIMENSIONS))

    for i, image_path in tqdm(enumerate(image_paths),
                              desc="Extracting global features",
                              total=len(image_paths)):
        if i >= num_embeddings:
            break

        ids[i] = PATH_ID[image_path]
        image_tensor = load_image_tensor(image_path)

        features = GLOBAL_FEATURE_EXTRACTION_FN(
            image_tensor,
            DELG_IMAGE_SCALES_TENSOR,
            DELG_SCORE_THRESHOLD_TENSOR
        )

        embeddings[i, :] = tf.nn.l2_normalize(
            tf.reduce_sum(features[0], axis=0, name='sum_pooling'),
            axis=0,
            name='final_l2_normalization'
        ).numpy()

    return ids, embeddings


def extract_local_features(image_path):
    """Extracts local features for the given `image_path`."""
    image_tensor = load_image_tensor(image_path)

    features = LOCAL_FEATURE_EXTRACTION_FN(
        image_tensor,
        DELG_IMAGE_SCALES_TENSOR,
        DELG_SCORE_THRESHOLD_TENSOR,
        LOCAL_FEATURE_NUM_TENSOR
    )

    # Shape: (N, 2)
    keypoints = tf.divide(
        tf.add(
            tf.gather(features[0], [0, 1], axis=1),
            tf.gather(features[0], [2, 3], axis=1)
        ), 2.0
    ).numpy()

    # Shape: (N, 128)
    descriptors = tf.nn.l2_normalize(
        features[1], axis=1, name='l2_normalization').numpy()

    return keypoints, descriptors


def dump_local_features(image_paths, name):
    if os.path.exists(name):
        print("Directory exists, not dumping!")
        return
    else:
        os.mkdir(name)

    for path in tqdm(image_paths, desc="Dumping local features", total=len(image_paths)):
        test_keypoints, test_descriptors = extract_local_features(path)
        filename = os.path.basename(path).split(".")[0]

        with open(f"{name}/{filename}.pkl", "wb") as fp:
            pickle.dump((test_keypoints, test_descriptors), fp)
