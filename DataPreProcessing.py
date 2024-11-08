import os
from collections import defaultdict
import numpy as np
import torch
import os
import numpy as np
from PIL import Image
import pandas as pd
from torchvision import transforms
import pickle
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset


PATH_MIMIC = "here will be the path to the MIMIC cxr dataset"

PATH_CXP =   "here will be the path to the CXP dataset"

CXP_details = "The path towards CXP_details"
cache_dir = 'The path towards chexpert '

def prepare_image_data_paths():


    image_paths = {
        'MIMIC': PATH_MIMIC, # MIMIC-CXR
        'CXP': PATH_CXP, # CheXpert
    }


    #A dictionary that creates paths to preprocessed CSV files for each dataset (MIMIC and CXP) by appending the subdirectory and file name to the base paths in image_paths.
    df_paths = {
        dataset: os.path.join(image_paths[dataset], 'cxr_fairness', f'preprocessed.csv')
        for dataset in image_paths 
    }


    take_labels = ['No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema']
    take_labels_all = take_labels + ['Enlarged Cardiomediastinum', 'Airspace Opacity', 'Lung Lesion', 'Pleural Other', 'Fracture', 'Support Devices']

    #Check for the origin of these constants 
    IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
    IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)

    # Create a dictionary that maps White ethnicity to 0, Black ethnicity to 1, and other ethnicities to 2.

    ethnicity_mapping = defaultdict(lambda: 2)
    ethnicity_mapping.update({
        'WHITE': 0,
        'BLACK/AFRICAN AMERICAN': 1
    })

    #A dictionary defining group values for sex, ethnicity, and age. This is used to standardize and categorize these attributes within the data.
    # Age classification appears to be arbitrary, allowing for the manipulation of age cohorts to test the hypothesis of the paper and potentially gain clearer insights.
    group_vals = { 'sex': ['M', 'F'], 'ethnicity': [0, 1, 2], 'age': ["18-40", "40-60", "60-80", "80-"] }


    return { 'image_paths': image_paths, 'df_paths': df_paths, 'take_labels': take_labels, 'take_labels_all': take_labels_all, 'IMAGENET_MEAN': IMAGENET_MEAN, 'IMAGENET_STD': IMAGENET_STD, 'ethnicity_mapping': ethnicity_mapping, 'group_vals': group_vals }


#process_MIMIC, is designed to preprocess a dataset (presumably from the MIMIC-CXR dataset) by cleaning, transforming, and filtering it according to specific criteria.



'''
split: The input DataFrame containing the dataset to be processed.

only_frontal: A boolean indicating whether to filter the dataset to include only frontal X-rays.

return_all_labels: A boolean indicating whether to return all labels or a subset.

The replacement is done for the entire DataFrame (all columns), and it maps certain values to new standardized values. Here’s what each replacement does:

[None] to 0: Any cell with a value of None is replaced with 0.

-1 to 0: Any cell with a value of -1 is replaced with 0.

"[False]" to 0 and "[True]" and "[ True]" to 1: Cells with these string representations of boolean values are standardized to 0 or 1.

'UNABLE TO OBTAIN' and 'UNKNOWN' to 0: These values are considered as unknown or not available and are replaced with 0.

'MARRIED' and 'LIFE PARTNER' to 'MARRIED/LIFE PARTNER': Combines these statuses into a single category.

'DIVORCED' and 'SEPARATED' to 'DIVORCED/SEPARATED': Combines these statuses into a single category.

Age ranges like '0-10' to '10-20' and others are grouped into broader categories like '18-40', '40-60', '60-80', and '80-'.
'''
# This function standardizes the data and merges some categories to create a processed MIMIC dataset.


def process_MIMIC(split, only_frontal, return_all_labels=True):
    prepareDict = prepare_image_data_paths()
    image_paths = prepareDict['image_paths']
    ethnicity_mapping = prepareDict['ethnicity_mapping']
    take_labels = prepareDict['take_labels']
    # Standardize and clean the dataset
    copy_subjectid = split['subject_id']
    split = split.drop(columns=['subject_id']).replace(
        [[None], -1, "[False]", "[True]", "[ True]", 'UNABLE TO OBTAIN', 'UNKNOWN', 'MARRIED', 'LIFE PARTNER',
         'DIVORCED', 'SEPARATED', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
         '90-100'],
        [0, 0, 0, 1, 1, 0, 0, 'MARRIED/LIFE PARTNER', 'MARRIED/LIFE PARTNER', 'DIVORCED/SEPARATED', 
         'DIVORCED/SEPARATED', '18-40', '18-40', '18-40', '18-40', '40-60', '40-60', '60-80', '60-80', '80-', '80-']
    )
    
    split['subject_id'] = copy_subjectid.astype(str)
    split['study_id'] = split['study_id'].astype(str)
    split['age'] = split["age_decile"]
    split['sex'] = split["gender"]
    split = split.rename(columns={
        'Pleural Effusion': 'Effusion',  
        'Lung Opacity': 'Airspace Opacity' 
    })
    
    split['path'] = split['path'].astype(str).apply(lambda x: os.path.join(image_paths['MIMIC'], x))
    
    if only_frontal:
        split = split[split.frontal]
    
    split['ethnicity'] = split['ethnicity'].map(ethnicity_mapping)
    split['env'] = 'MIMIC'
    split = split[split.age != 0]
    
    columns_to_return = [
        'subject_id', 'path', 'sex', 'age', 'ethnicity', 'env', 'frontal', 'study_id', 'fold_id'
    ] + take_labels

    if return_all_labels:
        columns_to_return += ['Enlarged Cardiomediastinum', 'Airspace Opacity', 'Lung Lesion', 'Pleural Other', 'Fracture', 'Support Devices']
    
    return split[columns_to_return]

#The process_CXP function is designed to preprocess the CheXpert (CXP) dataset by standardizing and transforming data

'''
Bin Age: Converts continuous age values into age range categories ('18-40', '40-60', '60-80', '80-').

Backup subject_id: Stores subject_id in copy_subjectid.

Drop subject_id Column and Replace Values: Removes subject_id and standardizes other values:

Replace [None], -1, "[False]", "[True]", "[ True]" with [0, 0, 0, 1, 1].

'''


def process_CXP(split, only_frontal, return_all_labels=True):
    prepareDict = prepare_image_data_paths()
    image_paths = prepareDict['image_paths']
    take_labels = prepareDict['take_labels']
    def bin_age(x):
        if 0 <= x < 40: return '18-40'
        elif 40 <= x < 60: return '40-60'
        elif 60 <= x < 80: return '60-80'
        else: return '80-'

    split['Age'] = split['Age'].apply(bin_age)

    copy_subjectid = split['subject_id']
    split = split.drop(columns=['subject_id']).replace(
        [[None], -1, "[False]", "[True]", "[ True]"], [0, 0, 0, 1, 1]
    )
    split['subject_id'] = copy_subjectid.astype(str)
    
    split['Sex'] = np.where(split['Sex'] == 'Female', 'F', split['Sex'])
    split['Sex'] = np.where(split['Sex'] == 'Male', 'M', split['Sex'])
    split = split.rename(columns={
        'Pleural Effusion': 'Effusion',
        'Lung Opacity': 'Airspace Opacity',
        'Sex': 'sex',
        'Age': 'age'
    })
    
    split['path'] = split['Path'].astype(str).apply(lambda x: os.path.join(image_paths['CXP'], x))
    split['frontal'] = split['Frontal/Lateral'] == 'Frontal'
    
    if only_frontal:
        split = split[split['frontal']]
    
    split['env'] = 'CXP'
    split['study_id'] = split['path'].apply(lambda x: x[x.index('patient'):x.rindex('/')])
    
    columns_to_return = [
        'subject_id', 'path', 'sex', 'age', 'env', 'frontal', 'study_id', 'fold_id', 'ethnicity'
    ] + take_labels

    if return_all_labels:
        columns_to_return += [
            'Enlarged Cardiomediastinum', 'Airspace Opacity', 'Lung Lesion', 'Pleural Other', 'Fracture', 'Support Devices'
        ]
    
    return split[columns_to_return]


# The function below returns a specific function based on the provided string value.

def get_process_func(env):
    
    
    if env == 'MIMIC':
        return process_MIMIC
    if env == 'CXP':
        return process_CXP
    raise NotImplementedError


import pandas as pd

def load_df(env, val_fold, only_frontal=False, query_str=None):
    assert isinstance(val_fold, str)
    prepareDict = prepare_image_data_paths()
    df_paths = prepareDict['df_paths']
   
    # Load the dataset
    df = pd.read_csv(df_paths[env])
    
    # Process the dataset based on the environment
    func = get_process_func(env)
    df = func(df, only_frontal)
    
    # Apply query if provided
    if query_str:
        df = df.query(query_str)
    
    # Determine train, validation, and test folds
    train_folds = [i for i in df.fold_id.unique() if i not in ['test', val_fold]]
    ans = {
        'train': df[df.fold_id.isin(train_folds)].reset_index(drop=True),
        'val': df[df.fold_id == val_fold].reset_index(drop=True),
        'test': df[df.fold_id == 'test'].reset_index(drop=True)
    }
    
    # Ensure each split has at least one row
    assert all(len(ans[i]) > 0 for i in ans)
    
    return ans

class ConcatWrapper(Dataset):
    pass
    
class AllDatasetsShared(Dataset):
    pass

def get_dataset(dfs_all, env, split=None, concat_group=False, protected_attr=None, 
                imagenet_norm=True, augment=0, use_cache=False, subset_label=None, smaller_label_set=False):
    prepareDict = prepare_image_data_paths()
    # Image transformations based on augmentation type
    if augment == 1:
        image_transforms = [transforms.RandomHorizontalFlip(), 
                            transforms.RandomRotation(10),     
                            transforms.RandomResizedCrop(size=224, scale=(0.75, 1.0)),
                            transforms.ToTensor()]
    elif augment == 0:
        image_transforms = [transforms.ToTensor()]
    elif augment == -1:
        image_transforms = []

    if imagenet_norm and augment != -1:
        image_transforms.append(transforms.Normalize(prepareDict["IMAGENET_MEAN"], prepareDict["IMAGENET_STD"]))             
    
    # Determine splits
    splits = [split] if split else ['train', 'val', 'test']
    
    # Create datasets for each split
    datasets = []
    for s in splits:
        cache_dir = Path(cache_dir) / f'{env}/'
        cache_dir.mkdir(parents=True, exist_ok=True)
        dfs = dfs_all[s]
        dataset = ConcatWrapper(
            AllDatasetsShared(dfs, label_set= prepareDict["take_labels"] if smaller_label_set else prepareDict["take_labels_all"],
                              transform=transforms.Compose(image_transforms), split=s, 
                              cache=use_cache, cache_dir=cache_dir, subset_label=subset_label),
            concat_group=concat_group, protected_attr=protected_attr)
        datasets.append(dataset)
    
    # Return the appropriate dataset
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        return datasets[0]
    else:
        ds = ConcatDataset(datasets)
        ds.dataframe = pd.concat([d.dataframe for d in datasets])
        return ds

