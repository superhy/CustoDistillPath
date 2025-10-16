'''
@author: Yang Hu
'''

import gc
import os
import pickle
import random
import sys

import numpy as np
from support.env import ENV
from support.files import clear_dir, parse_slide_caseid_from_filepath, \
    parse_slideid_from_filepath
from support.metadata import query_task_label_dict_fromcsv
from wsi import filter_tools
from wsi import slide_tools
from wsi import tiles_tools


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

# from support import env_monuseg, env_gtex_seg


sys.path.append("..")       
def generate_tiles_list_pkl_filepath(slide_filepath, tiles_list_pkl_dir):
    """
    generate the filepath of pickle 
    """
    
    slide_id = parse_slideid_from_filepath(slide_filepath)
    tiles_list_pkl_filename = slide_id + '-tiles.pkl'
    if not os.path.exists(tiles_list_pkl_dir):
        os.makedirs(tiles_list_pkl_dir)
    
    pkl_filepath = os.path.join(tiles_list_pkl_dir, tiles_list_pkl_filename)
    
    return pkl_filepath

 
def recovery_tiles_list_from_pkl(pkl_filepath):
    """
    load tiles list from [.pkl] file on disk
    (this function is for some other module)
    """
    with open(pkl_filepath, 'rb') as f_pkl:
        tiles_list = pickle.load(f_pkl)
    return tiles_list


def parse_filesystem_slide(slide_dir):
    slide_path_list = []
    for root, dirs, files in os.walk(slide_dir):
        for f in files:
            if f.endswith('.svs') or f.endswith('.tiff') or f.endswith('.tif') or f.endswith('.ndpi'):
                slide_path = os.path.join(root, f)
                slide_path_list.append(slide_path)
                
    return slide_path_list

        
def slide_tiles_split_keep_object(ENV_task):
    """
    conduct the whole pipeline of slide's tiles split, by Sequential process
    store the tiles Object [.pkl] on disk
    
    without train/test separation
    
    Args:
        slides_folder: the folder path of slides ready for segmentation
    """
    
    _env_slide_dir = ENV_task.SLIDE_DIR
    _env_tile_pkl_train_dir = ENV_task.TILE_PKL_DIR
    
    ''' load all slides '''
    slide_path_list = parse_filesystem_slide(_env_slide_dir)
    for i, slide_path in enumerate(slide_path_list):
        np_small_img, large_w, large_h, small_w, small_h = slide_tools.slide_to_scaled_np_image(slide_path)
        np_small_filtered_img = filter_tools.apply_image_filters_he(np_small_img)
        
        shape_set_img = (large_w, large_h, small_w, small_h)
        tiles_list = tiles_tools.get_slide_tiles(np_small_filtered_img, shape_set_img, slide_path,
                                                 ENV.TILE_W_SIZE, ENV.TILE_H_SIZE,
                                                 t_p_threshold=ENV.TP_TILES_THRESHOLD, load_small_tile=False)
        
        print('generate tiles for slide: %s, keep [%d] tile objects in (.pkl) list.' % (slide_path, len(tiles_list)))
        if len(tiles_list) == 0:
            continue
        pkl_path = generate_tiles_list_pkl_filepath(slide_path, _env_tile_pkl_train_dir)
        print('store the [.pkl] in {}'.format(pkl_path))
        with open(pkl_path, 'wb') as f_pkl:
            pickle.dump(tiles_list, f_pkl)
        
        gc.collect()
        
    
if __name__ == '__main__': 
    pass
    
    
