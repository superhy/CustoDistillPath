import os
import random

import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from wsi.process import recovery_tiles_list_from_pkl


def safe_random_sample(pickpool, K):
    
    if len(pickpool) > K:
        return random.sample(pickpool, K)
    else:
        return pickpool

def load_slides_tileslist(ENV_task):
    """
    a simplify version for above function, only return the dict of slide -> tiles' objects
    
    need to prepare parmes:
        _env_process_slide_tile_pkl_train_dir,
        _env_process_slide_tile_pkl_test_dir,
        _env_process_slide_tumor_tile_pkl_train_dir,
        _env_process_slide_tumor_tile_pkl_test_dir,
        (label_type)
        
    Args:
        ENV_task: the task environment object
        for_train:
        
    Return:
        slides_tiles_dict: dict of slide -> tiles' objects
    
    """
    
    pkl_dir = ENV_task.TILE_PKL_DIR
    
    pkl_files = os.listdir(pkl_dir)
    
    slides_tiles_dict = {}
    for pkl_f in pkl_files:
        # each slide each pkl
        tiles_list = recovery_tiles_list_from_pkl(os.path.join(pkl_dir, pkl_f))
        slide_id = tiles_list[0].query_slideid()
        slides_tiles_dict[slide_id] = tiles_list
        
    return slides_tiles_dict

class SlideTilesDataset(Dataset):
    """
    The dataset for all tiles of one slide
    """
    
    def __init__(self, tiles_list, transform: transforms):
        self.tiles_list = tiles_list
        
        self.transform = transform
        ''' make slide cache in memory '''
        self.cache_slide = ('none', None)

    def __len__(self):
        return len(self.tiles_list)

    def __getitem__(self, idx):
        '''
        get_ihc_dab: if load the ihc_dab stain, only return the image with brown channel
        '''
        tile = self.tiles_list[idx]
        
        ''' using slide cache '''
        loading_slide_id = tile.query_slideid()
        if loading_slide_id == self.cache_slide[0]:
            preload_slide = self.cache_slide[1]
        else:
            _, preload_slide = tile.get_pil_scaled_slide()
            self.cache_slide = (loading_slide_id, preload_slide)
            
        image = tile.get_pil_tile(preload_slide)
        image = self.transform(image)
        
        return image

class FeatureClsDataset(Dataset):
    def __init__(self, features_paths, labels):
        """
        Args:
            features_paths: List of paths to pre-stored features from different foundation models
            labels: Ground truth labels
        """
        self.features_list = []
        for path in features_paths:
            self.features_list.append(torch.load(path))
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Concatenate features from different models
        combined_features = torch.cat([features[idx] for features in self.features_list], dim=0)
        return combined_features, self.labels[idx]