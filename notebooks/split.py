# -*- coding: utf-8 -*-
"""split.ipynb

Original file is located at
    https://github.com/paddydoc/paddy-doctor-dataset/blob/main/notebooks/split.ipynb

## Split the final dataset  into train/test sets
"""

import glob
import os
import shutil
import random
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

SEED = 1234
def set_seed(seed=SEED):
    np.random.seed(seed) 
    random.seed(seed)
set_seed()

final_dir = './diseases_final_small/'
split_dir = './diseases_final_small_split_400/'
# final_dir = './diseases_final_augmented_2k/'
# split_dir = './diseases_final_augmented_2k_split/'

metadata = pd.read_csv(os.path.join(final_dir, 'metadata.csv'))
metadata.info()

metadata = metadata.groupby(['label']).apply(lambda x: x.sample(n=400, random_state=SEED)).reset_index(drop=True)
metadata.label.value_counts()

metadata

## test_size= 0.2793 to get 10k samples in train
train, test = train_test_split(metadata, random_state=SEED, test_size= 0.2,
                               stratify = metadata[['label']])
                               #stratify = metadata[['label', 'variety']])
train = train.sort_values(by=['image_id', 'label'])
test  = test.sort_values(by=['image_id', 'label'])
print(metadata.shape, train.shape, test.shape)

"""#### copy train images"""

for index, row in tqdm(train.iterrows()):
    from_path = os.path.join(final_dir, row['label'], row['image_id'])
    to_dir    = os.path.join(split_dir, 'train',  row['label'])
    to_path   = os.path.join(to_dir, row['image_id'])
    
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    shutil.copy(from_path, to_path)
    #print(from_path, to_path, os.path.exists(from_path))

"""#### copy test images"""

for index, row in tqdm(test.iterrows()):
    from_path = os.path.join(final_dir, row['label'], row['image_id'])
    to_dir    = os.path.join(split_dir, 'test',  row['label'])
    to_path   = os.path.join(to_dir, row['image_id'])
    
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    shutil.copy(from_path, to_path)
    #print(from_path, to_path, os.path.exists(from_path))

"""#### save the metadata files"""

train.to_csv(os.path.join(split_dir, 'metadata-train.csv'), index=False)
test.to_csv(os.path.join(split_dir, 'metadata-test.csv'), index=False)

#combined metadata
train['split'] = 'train'
test['split'] = 'test'
metadata_split = pd.concat([train, test])
metadata_split = metadata_split.sort_values(by=['image_id', 'label'])
metadata_split.to_csv(os.path.join(split_dir, 'metadata.csv'), index=False)
metadata_split.split.value_counts()

#!pip install seedir
#!pip install emoji
#!pip install seedir[emoji]
import emoji
#https://github.com/earnestt1234/seedir
import seedir as sd
sd.seedir(split_dir, style='lines', itemlimit=13, depthlimit=1, beyond='content')
#sd.seedir(split_dir, style='lines', itemlimit=15, depthlimit=2)

#sd.seedir(final_dir, style='lines', itemlimit=13, depthlimit=2)
sd.seedir(final_dir, style='lines', itemlimit=13, depthlimit=2)

"""### descriptive stats"""

print(metadata.variety.value_counts())
metadata.variety.value_counts().plot(kind='bar')

print(metadata.age.value_counts())
metadata.age.value_counts().plot(kind='bar')



metadata['label'].value_counts() / metadata['label'].shape[0]

train['label'].value_counts() / train['label'].shape[0]
#test['label'].value_counts() / train['label'].shape[0]

# ## https://stackoverflow.com/questions/53997862/pandas-groupby-two-columns-and-plot
# train.groupby(['label', 'variety'])['image_id'].count().plot(kind='bar', figsize=(16,6))

# cols = ['label', 'variety']
# #train[['label', 'variety', 'age']].groupby(cols).count().plot.bar()
# train.groupby(cols).count().unstack('variety').plot.bar()
# #.plot(kind='bar', figsize=(16,6))

#pd.crosstab(train['label'],train['variety']).plot(kind='barh', figsize=(8,12))