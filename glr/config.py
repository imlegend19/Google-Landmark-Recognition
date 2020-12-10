import tensorflow as tf

DATASET_DIR = "paris"

CORRUPTED = [
    'louvre/paris_louvre_000136.jpg',
    'louvre/paris_louvre_000146.jpg',
    'moulinrouge/paris_moulinrouge_000422.jpg',
    'museedorsay/paris_museedorsay_001059.jpg',
    'notredame/paris_notredame_000188.jpg',
    'pantheon/paris_pantheon_000284.jpg',
    'pantheon/paris_pantheon_000960.jpg',
    'pantheon/paris_pantheon_000974.jpg',
    'pompidou/paris_pompidou_000195.jpg',
    'pompidou/paris_pompidou_000196.jpg',
    'pompidou/paris_pompidou_000201.jpg',
    'pompidou/paris_pompidou_000467.jpg',
    'pompidou/paris_pompidou_000640.jpg',
    'sacrecoeur/paris_sacrecoeur_000299.jpg',
    'sacrecoeur/paris_sacrecoeur_000330.jpg',
    'sacrecoeur/paris_sacrecoeur_000353.jpg',
    'triomphe/paris_triomphe_000662.jpg',
    'triomphe/paris_triomphe_000833.jpg',
    'triomphe/paris_triomphe_000863.jpg',
    'triomphe/paris_triomphe_000867.jpg'
]

# Dataset parameters:
INPUT_DIR = 'content'

# DEBUGGING PARAMS:
MAX_NUM_EMBEDDINGS = -1

# Retrieval & re-ranking parameters:
NUM_TO_RERANK = 6
TOP_K = 3

# RANSAC parameters:
MAX_INLIER_SCORE = 26
MAX_REPROJECTION_ERROR = 6.0
MAX_RANSAC_ITERATIONS = 100000
HOMOGRAPHY_CONFIDENCE = 0.95

# DELG model:
SAVED_MODEL_DIR = 'delg-saved-models/local_and_global'
DELG_MODEL = tf.saved_model.load(SAVED_MODEL_DIR)
DELG_IMAGE_SCALES_TENSOR = tf.convert_to_tensor([0.70710677, 1.0, 1.4142135])
DELG_SCORE_THRESHOLD_TENSOR = tf.constant(175.)
DELG_INPUT_TENSOR_NAMES = [
    'input_image:0', 'input_scales:0', 'input_abs_thres:0'
]

# Global feature extraction:
NUM_EMBEDDING_DIMENSIONS = 2048
GLOBAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(
    DELG_INPUT_TENSOR_NAMES,
    ['global_descriptors:0']
)

# Local feature extraction:
LOCAL_FEATURE_NUM_TENSOR = tf.constant(1000)
LOCAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(
    DELG_INPUT_TENSOR_NAMES + ['input_max_feature_num:0'],
    ['boxes:0', 'features:0']
)
