import os
import pickle

import PIL
import tensorflow as tf

import copy
import operator

import numpy as np
import pydegensac
from scipy import spatial
from tqdm import tqdm

from glr import HOMOGRAPHY_CONFIDENCE, ID_PATH, MAX_INLIER_SCORE, MAX_RANSAC_ITERATIONS, MAX_REPROJECTION_ERROR, TOP_K


def get_image_path(image_id):
    return ID_PATH[image_id]


def load_image_tensor(image_path):
    return tf.convert_to_tensor(np.array(PIL.Image
                                         .open(image_path)
                                         .convert('RGB')))


def load_labelmap(df):
    labelmap = dict(zip(df.path, df.landmark_id))
    return labelmap


def get_putative_matching_keypoints(test_keypoints,
                                    test_descriptors,
                                    train_keypoints,
                                    train_descriptors,
                                    max_distance=0.9):
    """Finds matches from `test_descriptors` to KD-tree of `train_descriptors`."""

    train_descriptor_tree = spatial.cKDTree(train_descriptors)
    _, matches = train_descriptor_tree.query(
        test_descriptors, distance_upper_bound=max_distance)

    test_kp_count = test_keypoints.shape[0]
    train_kp_count = train_keypoints.shape[0]

    test_matching_keypoints = np.array([
        test_keypoints[i,]
        for i in range(test_kp_count)
        if matches[i] != train_kp_count
    ])

    train_matching_keypoints = np.array([
        train_keypoints[matches[i],]
        for i in range(test_kp_count)
        if matches[i] != train_kp_count
    ])

    return test_matching_keypoints, train_matching_keypoints


def get_num_inliers(test_keypoints, test_descriptors, train_keypoints,
                    train_descriptors):
    """Returns the number of RANSAC inliers."""

    test_match_kp, train_match_kp = get_putative_matching_keypoints(
        test_keypoints, test_descriptors, train_keypoints, train_descriptors)

    if test_match_kp.shape[0] <= 4:
        return 0

    try:
        _, mask = pydegensac.findHomography(test_match_kp, train_match_kp,
                                            MAX_REPROJECTION_ERROR,
                                            HOMOGRAPHY_CONFIDENCE,
                                            MAX_RANSAC_ITERATIONS)
    except np.linalg.LinAlgError:
        return 0

    return int(copy.deepcopy(mask).astype(np.float32).sum())


def get_total_score(num_inliers, global_score):
    local_score = min(num_inliers, MAX_INLIER_SCORE) / MAX_INLIER_SCORE
    return local_score + global_score


def get_prediction_map(test_ids, train_ids_labels_and_scores):
    """Makes dict from test ids and ranked training ids, labels, scores."""

    prediction_map = dict()

    for test_index, test_id in tqdm(enumerate(test_ids),
                                    desc="Getting prediction map",
                                    total=len(test_ids)):
        image_path = ID_PATH[test_id]
        aggregate_scores = {}

        for _, label, score in train_ids_labels_and_scores[test_index][:TOP_K]:
            if label not in aggregate_scores:
                aggregate_scores[label] = 0

            aggregate_scores[label] += score

        label, score = max(aggregate_scores.items(), key=operator.itemgetter(1))
        prediction_map[image_path] = {'score': score, 'class': label}

    return prediction_map


def dump_gf(name, data):
    with open(name, 'wb') as fp:
        pickle.dump(data, fp)


def load_gf(name):
    if os.path.exists(name):
        with open(name, 'rb') as fp:
            return pickle.load(fp)
