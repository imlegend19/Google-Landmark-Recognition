import pathlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from glr.config import *
import pandas as pd


def consider_img(path):
    if path in CORRUPTED:
        return False
    else:
        return True


df = pd.DataFrame()
df['path'] = [str(_.absolute()) for _ in pathlib.Path(DATASET_DIR).rglob('*.jpg') if consider_img(str(_))]

le = LabelEncoder()
labels = [os.path.basename(s).split('.')[0].split('_')[1] for s in df['path'].tolist()]
le.fit(labels)

ID_LABEL = dict(zip(le.transform(le.classes_), le.classes_))

df['landmark_id'] = le.transform(labels)

TRAIN, TEST = train_test_split(df, test_size=0.3)

paths = df['path'].tolist()

ID_PATH = {}
PATH_ID = {}
for i in range(len(paths)):
    ID_PATH[i] = paths[i]
    PATH_ID[paths[i]] = i

del paths
del df
