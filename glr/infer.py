import csv
import gc
import os
import pickle

import numpy as np
from scipy import spatial
from tqdm import tqdm

from glr import ID_LABEL, ID_PATH, NUM_TO_RERANK, PATH_ID, ROOT, TEST, TEST_GF, TEST_LF, TRAIN, TRAIN_GF, TRAIN_LF
from glr.extract_features import extract_global_features, extract_local_features
from glr.helpers import (dump_gf, get_image_path, get_num_inliers, get_prediction_map, get_total_score, load_gf,
                         load_labelmap)


def rescore_and_rerank_by_num_inliers(test_image_id,
                                      train_ids_labels_and_scores):
    """Returns rescored and sorted training images by local feature extraction."""

    test_image_path = get_image_path(test_image_id)

    try:
        name = os.path.basename(test_image_path).split('.')[0]
        with open(f'{TEST_LF}/{name}.pkl', 'rb') as fp:
            test_keypoints, test_descriptors = pickle.load(fp)
    except FileNotFoundError:
        test_keypoints, test_descriptors = extract_local_features(test_image_path)

    for i in range(len(train_ids_labels_and_scores)):
        train_image_id, label, global_score = train_ids_labels_and_scores[i]

        train_image_path = get_image_path(train_image_id)
        name = os.path.basename(train_image_path).split('.')[0]

        with open(os.path.join(TRAIN_LF, f"{name}.pkl"), 'rb') as fp:
            train_keypoints, train_descriptors = pickle.load(fp)

        num_inliers = get_num_inliers(test_keypoints, test_descriptors,
                                      train_keypoints, train_descriptors)
        total_score = get_total_score(num_inliers, global_score)
        train_ids_labels_and_scores[i] = (train_image_id, label, total_score)

    train_ids_labels_and_scores.sort(key=lambda x: x[2], reverse=True)

    return train_ids_labels_and_scores


def get_predictions(labelmap, infer=None):
    """Gets predictions using embedding similarity and local feature reranking."""

    if not infer:
        test_gf = load_gf(TEST_GF)
        if not test_gf:
            test_ids, test_embeddings = extract_global_features(TEST['path'].tolist())
            dump_gf(TEST_GF, (test_ids, test_embeddings))
        else:
            test_ids, test_embeddings = test_gf
            del test_gf
    else:
        test_ids, test_embeddings = extract_global_features([infer])

    train_gf = load_gf(TRAIN_GF)
    if not train_gf:
        train_ids, train_embeddings = extract_global_features(TRAIN['path'].tolist())
        dump_gf(TRAIN_GF, (train_ids, train_embeddings))
    else:
        train_ids, train_embeddings = train_gf
        del train_gf

    train_ids_labels_and_scores = [None] * test_embeddings.shape[0]

    for test_index in tqdm(range(test_embeddings.shape[0]),
                           desc="Getting predictions map",
                           total=test_embeddings.shape[0]):
        distances = spatial.distance.cdist(
            test_embeddings[np.newaxis, test_index, :], train_embeddings,
            'cosine')[0]
        partition = np.argpartition(distances, NUM_TO_RERANK)[:NUM_TO_RERANK]

        nearest = sorted([(train_ids[p], distances[p]) for p in partition],
                         key=lambda x: x[1])

        train_ids_labels_and_scores[test_index] = [
            (train_id, labelmap[ID_PATH[train_id]], 1. - cosine_distance)
            for train_id, cosine_distance in nearest
        ]

    del test_embeddings
    del train_embeddings
    del labelmap
    gc.collect()

    pre_verification_predictions = get_prediction_map(
        test_ids, train_ids_labels_and_scores)

    if not os.path.exists(TRAIN_LF):
        os.mkdir(TRAIN_LF)

    for test_index, test_id in tqdm(enumerate(test_ids),
                                    desc="Rescoring and reranking",
                                    total=len(test_ids)):
        train_ids_labels_and_scores[test_index] = rescore_and_rerank_by_num_inliers(
            test_id, train_ids_labels_and_scores[test_index])

    post_verification_predictions = get_prediction_map(
        test_ids,
        train_ids_labels_and_scores
    )

    return pre_verification_predictions, post_verification_predictions


def save_results_csv(predictions):
    with open(os.path.join(ROOT, 'results.csv'), 'w') as result_csv:
        csv_writer = csv.DictWriter(result_csv, fieldnames=['path', 'landmark', 'score'])
        csv_writer.writeheader()
        for image_path, prediction in predictions.items():
            label = prediction['class']
            score = prediction['score']
            csv_writer.writerow({'path' : image_path, 'landmark': f'{label}',
                                 'score': f'{score}'})


def get_landmark(predictions):
    print("Getting Landmark...")
    _, prediction = list(predictions.items())[0]
    label = prediction['class']
    score = prediction['score']

    return ID_LABEL[label], score


def infer(image_path):
    img_id = max(PATH_ID.values()) + 1
    PATH_ID[image_path] = img_id
    ID_PATH[img_id] = image_path

    labelmap = load_labelmap(TRAIN)
    _, post_verification_predictions = get_predictions(labelmap, image_path)

    del PATH_ID[image_path]
    del ID_PATH[img_id]

    return get_landmark(post_verification_predictions)
