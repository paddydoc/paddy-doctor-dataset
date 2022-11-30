# -*- coding: utf-8 -*-
"""augmentation.ipynb

Original file is located at
    https://github.com/paddydoc/paddy-doctor-dataset/blob/main/notebooks/augmentation.ipynb

## Create the augmented datasets
"""

import glob
import os
import random
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 1234
def set_seed(seed=SEED):
    np.random.seed(seed) 
    tf.set_random_seed(seed) 
    random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    os.environ['TF_CUDNN_DETERMINISM'] = "1"
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed()

final_dir = './diseases_final/'
metadata_file = 'metadata.csv'
metadata = pd.read_csv(os.path.join(final_dir, metadata_file))

classes = metadata.label.value_counts().to_dict()
print(classes)
metadata.head()

"""## Image augmentation #1 : 5 times of original size"""

final_aug_dir = './diseases_final_augmented_5x/'
target_img_size = (256,256)

datagen = ImageDataGenerator(
    rotation_range = 5,
    shear_range = 0.2,
    zoom_range = 0.2,
    width_shift_range = 0.0,
    height_shift_range = 0.0,
    fill_mode = 'nearest',
    horizontal_flip = True,
    vertical_flip = False    
)

for cls_name, count in sorted(classes.items()):
    
    if cls_name != 'bacterial_leaf_blight':
        next
    
    to_dir = os.path.join(final_aug_dir, cls_name)
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
        
    image = datagen.flow_from_directory(
        final_dir,
        classes = [cls_name],
        target_size = target_img_size,
        save_to_dir = to_dir,
        save_prefix = '',
        save_format = 'jpg',
        seed = SEED,
        batch_size = count)
    
    total = 5
    for ix in range(total):
        print(datetime.now(), 'Augmenting', cls_name, ' #iteration = ', ix)
        im = image.next()
    print('')

"""#### Rename augmented files and create new meta data file"""

aug_meta = []
for cls_name, count in sorted(classes.items()):
    meta1 = metadata[metadata.label == cls_name]
    meta1['file_seq'] = range(meta1.shape[0])
    
    cls_dir = os.path.join(final_aug_dir, cls_name)    
    
    all_files = [Path(filename).name for filename in glob.glob(cls_dir + '/*.jpg')]
    
    files_df = pd.DataFrame({'filename': all_files})
    files_df['file_seq'] = files_df.filename.apply(lambda x: int(x.split('_')[1]))
    
    meta2 = pd.merge(meta1, files_df, on="file_seq")
    meta2['seq'] = meta2.groupby(['image_id']).cumcount()
    meta2['new_filename'] = meta2['image_id'].str[:8] + '_' + meta2['seq'].apply(lambda x: str(x+1).zfill(3)) + '.jpg'
    aug_meta.append(meta2)
    print(cls_dir, meta1.shape[0], len(all_files), meta2.shape[0])    
    
    ## rename files
    for index, row in meta2.iterrows():
        cls_dir = os.path.join(final_aug_dir, cls_name)
        from_path = os.path.join(cls_dir, row['filename'])
        to_path = os.path.join(cls_dir, row['new_filename'])
        os.rename(from_path, to_path)

aug_meta = pd.concat(aug_meta)
aug_meta['image_id'] = aug_meta['new_filename']
aug_meta = aug_meta[['image_id', 'label', 'variety', 'age']]
aug_meta.to_csv(os.path.join(final_aug_dir, 'metadata.csv'), index=False)
print(aug_meta.shape)
aug_meta

"""### Image augmentation #2 : 2k samples for each class"""

final_dir = './diseases_final/'
final_aug_dir = './diseases_final_augmented_2k/'
target_img_size = (256,256)

datagen = ImageDataGenerator(
    rotation_range = 5,
    shear_range = 0.2,
    zoom_range = 0.2,
    width_shift_range = 0.0,
    height_shift_range = 0.0,
    fill_mode = 'nearest',
    horizontal_flip = True,
    vertical_flip = False    
)

for cls_name, count in sorted(classes.items()):
    
    print(datetime.now(), 'Augmenting', cls_name)    
    to_dir = os.path.join(final_aug_dir, cls_name)
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
        
    image = datagen.flow_from_directory(
        final_dir,
        classes = [cls_name],
        target_size = target_img_size,
        save_to_dir = to_dir,
        save_prefix = '',
        save_format = 'jpg',
        seed = SEED,
        batch_size = 1)
    
    total = 2000
    for ix in range(total):
        #print(datetime.now(), 'Augmenting for', cls_name, ' #iteration = ', ix)
        im = image.next()
    print('')

"""#### Rename augmented files and create new meta data file"""

aug_meta = []
for cls_name, count in sorted(classes.items()):
    meta1 = metadata[metadata.label == cls_name]
    meta1['file_seq'] = range(meta1.shape[0])
    
    cls_dir = os.path.join(final_aug_dir, cls_name)    
    
    all_files = [Path(filename).name for filename in glob.glob(cls_dir + '/*.jpg')]
    
    files_df = pd.DataFrame({'filename': all_files})
    files_df['file_seq'] = files_df.filename.apply(lambda x: int(x.split('_')[1]))
    
    meta2 = pd.merge(meta1, files_df, on="file_seq")
    meta2['seq'] = meta2.groupby(['image_id']).cumcount()
    meta2['new_filename'] = meta2['image_id'].str[:8] + '_' + meta2['seq'].apply(lambda x: str(x+1).zfill(3)) + '.jpg'
    aug_meta.append(meta2)
    print(cls_dir, meta1.shape[0], len(all_files), meta2.shape[0])    
    
    ## rename files
    for index, row in meta2.iterrows():
        cls_dir = os.path.join(final_aug_dir, cls_name)
        from_path = os.path.join(cls_dir, row['filename'])
        to_path = os.path.join(cls_dir, row['new_filename'])
        os.rename(from_path, to_path)

aug_meta = pd.concat(aug_meta)
aug_meta['image_id'] = aug_meta['new_filename']
aug_meta = aug_meta[['image_id', 'label', 'variety', 'age']]
aug_meta.to_csv(os.path.join(final_aug_dir, 'metadata.csv'), index=False)
print(aug_meta.shape)
aug_meta

"""### Image augmentation #3 : 5k samples for each class"""

final_aug_dir = './diseases_final_augmented_5k/'
target_img_size = (256,256)

datagen = ImageDataGenerator(
    rotation_range = 5,
    shear_range = 0.2,
    zoom_range = 0.2,
    width_shift_range = 0.0,
    height_shift_range = 0.0,
    fill_mode = 'nearest',
    horizontal_flip = True,
    vertical_flip = False    
)

for cls_name, count in sorted(classes.items()):
    
    print(datetime.now(), 'Augmenting', cls_name)    
#     if cls_name != 'bacterial_leaf_blight':
#         continue
    
    to_dir = os.path.join(final_aug_dir, cls_name)
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
        
    image = datagen.flow_from_directory(
        final_dir,
        classes = [cls_name],
        target_size = target_img_size,
        save_to_dir = to_dir,
        save_prefix = '',
        save_format = 'jpg',
        seed = SEED,
        batch_size = 1)
    
    total = 5000
    for ix in range(total):
        #print(datetime.now(), 'Augmenting for', cls_name, ' #iteration = ', ix)
        im = image.next()
    print('')

"""#### Rename augmented files and create new meta data file"""

aug_meta = []
for cls_name, count in sorted(classes.items()):
    meta1 = metadata[metadata.label == cls_name]
    meta1['file_seq'] = range(meta1.shape[0])
    
    cls_dir = os.path.join(final_aug_dir, cls_name)    
    
    all_files = [Path(filename).name for filename in glob.glob(cls_dir + '/*.jpg')]
    
    files_df = pd.DataFrame({'filename': all_files})
    files_df['file_seq'] = files_df.filename.apply(lambda x: int(x.split('_')[1]))
    
    meta2 = pd.merge(meta1, files_df, on="file_seq")
    meta2['seq'] = meta2.groupby(['image_id']).cumcount()
    meta2['new_filename'] = meta2['image_id'].str[:8] + '_' + meta2['seq'].apply(lambda x: str(x+1).zfill(3)) + '.jpg'
    aug_meta.append(meta2)
    print(cls_dir, meta1.shape[0], len(all_files), meta2.shape[0])    
    
    ## rename files
    for index, row in meta2.iterrows():
        cls_dir = os.path.join(final_aug_dir, cls_name)
        from_path = os.path.join(cls_dir, row['filename'])
        to_path = os.path.join(cls_dir, row['new_filename'])
        os.rename(from_path, to_path)

aug_meta = pd.concat(aug_meta)
aug_meta['image_id'] = aug_meta['new_filename']
aug_meta = aug_meta[['image_id', 'label', 'variety', 'age']]
aug_meta.to_csv(os.path.join(final_aug_dir, 'metadata.csv'), index=False)
print(aug_meta.shape)
aug_meta

# aug_meta = pd.concat(aug_meta)
# aug_meta.to_csv(os.path.join(final_aug_dir, 'metadata.csv'), index=False)

# meta2 = pd.merge(meta1, df1, on="file_seq")
# meta2['seq'] = meta2.groupby(['image_id']).cumcount()
# meta2['new_filename'] = meta2['image_id'].str[:8] + '_' + meta2['seq'].apply(lambda x: str(x+1).zfill(3)) + '.jpg'
# meta2.to_csv('test.csv')
# meta2