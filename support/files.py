'''
@author: yang hu

'''

from _collections import defaultdict
import csv
import glob
import math
import os
import pickle
import random
import shutil

import openslide

import pandas as pd
from support import env_ov_bev, env_tcga_lung, env_colitis_marsh, env_focus_kras,\
    env_camelyon_tumor
from support.env import ENV
from support.metadata import query_task_label_dict_fromcsv
from support.tools import Time


def parse_filesystem_slide(slide_dir, original_download=False):
    slide_path_list = []
    for root, dirs, files in os.walk(slide_dir):
        for f in files:
            if f.endswith('.svs') or f.endswith('.tiff') or f.endswith('.tif') or f.endswith('.ndpi'):
                slide_path = os.path.join(root, f)
                if original_download == True:
                    print('found slide file from TCGA_original: ' + slide_path)
                slide_path_list.append(slide_path)
                
    return slide_path_list

def parse_filenames_tilespkl(tilespkl_dir):
    tilepkl_filename_list = []
    for root, dirs, files in os.walk(tilespkl_dir):
        for f in files:
            if f.endswith('.pkl'):
                print('read slide tiles pkl file: ' + f)
                tilepkl_filename_list.append(f)
                
    return tilepkl_filename_list


def clear_dir(abandon_dirs):
    """
    """
    print('remove the old dirs: {}.'.format(abandon_dirs))
    for dir in abandon_dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)

def move_file(src_path, dst_path, mode='move'):
    """
    move single file from src_path to dst_dir
    
    Args -> mode: 
        'move' -> move the file
        'copy' or other string -> copy the file (for test)
    """
    if mode == 'move':
        shutil.move(src_path, dst_path)
    else:
        shutil.copy(src_path, dst_path)
    
def move_TCGA_download_file_rename_batch(ENV_task, tcga_slide_path_list,
                                         mode='move', filter_annotated_slide=True):
    """
    Move the svs or tif file first,
    and rename according to the case id
    
    Args:
        ENV_task: the task environment object
    
    """
    
    label_dict = query_task_label_dict_fromcsv(ENV_task)
    
    _env_parse_data_slide_dir = ENV_task.PARSE_REPO_DATA_SLIDE_DIR
    if not os.path.exists(_env_parse_data_slide_dir):
        os.makedirs(_env_parse_data_slide_dir)
    else:
        clear_dir([_env_parse_data_slide_dir])
        os.makedirs(_env_parse_data_slide_dir)  
    
    count, filted = 0, 0
    for slide_path in tcga_slide_path_list:
        # parse case id from original slide_path
        case_id = parse_slide_caseid_from_filepath(slide_path)
        if filter_annotated_slide == True and case_id not in label_dict.keys():
            filted += 1
            continue
        
        # parse slide type and id
        slide_type_id = parse_slide_typeid_from_filepath(slide_path)
        suffix = slide_path[len(slide_path) - 4 : ]
        slide_new_name = case_id + slide_type_id + suffix
        print('move slide from: ' + slide_path + ' -> ' + os.path.join(_env_parse_data_slide_dir, slide_new_name))
        move_file(slide_path, os.path.join(_env_parse_data_slide_dir, slide_new_name), mode)
        count += 1
    print('moved {} slide files, filted {} slides.'.format(count, filted))
   
    
''' ---------- id parse functions for different datasets ---------- '''    
    
def parse_TCGA_slide_typeid_from_filepath(slide_filepath):
    """
    get the type id from slide's filepath, for TCGA dataset
    
    PS: with '_' + string of typeid
    """
    # slide_type_id = '_' + slide_filepath[slide_filepath.find('.') - 3: slide_filepath.find('.')]
    
    slide_filename = slide_filepath.split(os.sep)[-1]
    slide_type_id = '.' + slide_filename.split('.')[1]
    return slide_type_id

def parse_TCGA_slide_caseid_from_filepath(slide_filepath):
    """
    get the case id from slide's filepath, for TCGA dataset
    
    Args:
        slide_filepath: as name
        cut_range: filepath string cut range to get the TCGA case_id
    """
    # cut_range = 12
    # case_id = slide_filepath[slide_filepath.find('TCGA-'): slide_filepath.find('TCGA-') + cut_range]
    
    slide_filename = slide_filepath.split(os.sep)[-1]
    case_id = slide_filename.split('.')[0]
    return case_id

def parse_LVI_slide_bioid_from_filepath(slide_filepath):
    """
    get the type id from slide's filepath, for LVI dataset
    
    PS: with '_' + string of typeid
    """
    slide_name = slide_filepath.split('slides')[-1][1:]
    id_comps = slide_name.split('_', 2)
    slide_bio_id = id_comps[-1][: id_comps[-1].find('.')]
    return slide_bio_id

def parse_FOCUS_slide_typeid_from_filepath(slide_filepath):
    """
    get the type id from slide's filepath, for LVI dataset
    
    PS: with '_' + string of typeid
    """
    slide_name = slide_filepath.split('slides')[-1][1:]
    id_parts = slide_name.split('_')
    slide_type_id = id_parts[1]
    slide_type_id = '_' + slide_type_id[: slide_type_id.find('.svs')]
    return slide_type_id

def parse_LVI_slide_caseid_from_filepath(slide_filepath):
    """
    get the case id from slide's filepath, for TCGA dataset
    
    Args:
        slide_filepath: as name
    """
    slide_name = slide_filepath.split('slides')[-1][1:]
    id_comps = slide_name.split('_', 2)
    case_id = '_'.join([id_comps[0], id_comps[1]]) + '_'
    return case_id

def parse_GTEX_slide_caseid_from_filepath(slide_filepath):
    """
    get the case id from slide's filepath, for LIVER dataset
    """
    slide_name = slide_filepath.split('slides')[-1][1:]
    case_id = slide_name[:slide_name.find('.svs')]
    return case_id

def parse_IBD_slide_bioid_from_filepath(slide_filepath):
    """
    get the clinic bio_id from slide's filepath, for IBD dataset
    """
    slide_name = slide_filepath.split('slides')[-1][1:]
    slide_tail = slide_name.split('_')[-1]
    bio_id = '_' + slide_tail
    return bio_id
    
def parse_IBD_slide_caseid_from_filepath(slide_filepath):
    """
    get the caseid from slide's filepath, for IBD dataset
    """
    slide_name = slide_filepath.split('slides')[-1][1:]
    id_parts = slide_name.split('_')[:2]
    case_id = '{}_{}'.format(id_parts[0], id_parts[1])
    return case_id

def parse_FOCUS_slide_caseid_from_filepath(slide_filepath):
    """
    get the caseid from slide's filepath, for FOCUS dataset
    """
    slide_name = slide_filepath.split('slides')[-1][1:]
    id_parts = slide_name.split('_')
    case_id = id_parts[0]
    return case_id

def parse_OVBEV_slide_caseid_from_filepath(slide_filepath):
    """
    get the caseid from slide's filepath, for OV-Bev_Resp dataset
    """
    slide_name = slide_filepath.split(os.sep)[-1]
    case_id = slide_name.split('.')[0]
    return case_id

def parse_COLITIS_slide_caseid_from_filepath(slide_filepath):
    """
    get the caseid from slide's filepath, for OV-Bev_Resp dataset
    """
    slide_name = slide_filepath.split(os.sep)[-1]
    case_id = slide_name.split('.')[0]
    return case_id

def parse_CAMELYON_slide_caseid_from_filepath(slide_filepath):
    """
    get the caseid from slide's filepath, for OV-Bev_Resp dataset
    """
    slide_name = slide_filepath.split(os.sep)[-1]
    case_id = slide_name.split('.')[0]
    return case_id


''' ------------ uniform id parse functions, automatically switch for different datasets ------------ '''
def parse_slide_typeid_from_filepath(slide_filepath):
    """
    get the type id from slide's filepath, for all available dataset
    """
    if slide_filepath.find('TCGA') != -1:
        slide_type_id = parse_TCGA_slide_typeid_from_filepath(slide_filepath)
    elif slide_filepath.find('LVI') != -1:
        slide_type_id = parse_LVI_slide_bioid_from_filepath(slide_filepath)
    elif slide_filepath.find('GTEX') != -1:
        # there is no type_id in GTEX tissues
        slide_type_id = ''
    elif slide_filepath.find('IBD-Matthias') != -1:
        slide_type_id = parse_IBD_slide_bioid_from_filepath(slide_filepath)
    elif slide_filepath.find('FOCUS') != -1:
        slide_type_id = parse_FOCUS_slide_typeid_from_filepath(slide_filepath)
    elif slide_filepath.find('OV-Bev_Resp') != -1:
        slide_type_id = ''
    elif slide_filepath.find('A_TAP_AbCD_cohort') != -1:
        slide_type_id = ''
    elif slide_filepath.find('CAMELYON') != -1:
        slide_type_id = ''
    else:
        raise NameError('cannot detect right dataset indicator!')
    
    return slide_type_id
        
def parse_slide_caseid_from_filepath(slide_filepath):
    """
    get the case id from slide's filepath
    
    Args:
        slide_filepath: as name
    """
    if slide_filepath.find('TCGA') != -1:
        case_id = parse_TCGA_slide_caseid_from_filepath(slide_filepath)
    elif slide_filepath.find('LVI') != -1:
        case_id = parse_LVI_slide_caseid_from_filepath(slide_filepath)
    elif slide_filepath.find('GTEX') != -1:
        case_id = parse_GTEX_slide_caseid_from_filepath(slide_filepath)
    elif slide_filepath.find('IBD-Matthias') != -1:
        case_id = parse_IBD_slide_caseid_from_filepath(slide_filepath)
    elif slide_filepath.find('FOCUS') != -1:
        case_id = parse_FOCUS_slide_caseid_from_filepath(slide_filepath)
    elif slide_filepath.find('OV-Bev_Resp') != -1:
        case_id = parse_OVBEV_slide_caseid_from_filepath(slide_filepath)
    elif slide_filepath.find('A_TAP_AbCD_cohort') != -1:
        case_id = parse_COLITIS_slide_caseid_from_filepath(slide_filepath)
    elif slide_filepath.find('CAMELYON') != -1:
        case_id = parse_CAMELYON_slide_caseid_from_filepath(slide_filepath)
    else:
        raise NameError('cannot detect right dataset indicator!')
    
    return case_id

def parse_caseid_from_slideid(slide_id):
    '''
    get the case_id from slide_id
    '''
    if slide_id.startswith('TCGA'):
        case_id = slide_id[:slide_id.find('_') ]
    elif slide_id.startswith('SC'): # focus
        case_id = slide_id.split('_')[0]
    else:
        case_id = slide_id
        
    return case_id

def parse_slideid_from_filepath(slide_filepath):
    """
    get the whole slideid from slide's filepath
    
    PS combine the previous 2 functions
    """
    return parse_slide_caseid_from_filepath(slide_filepath) + parse_slide_typeid_from_filepath(slide_filepath)

def parse_slideid_from_tilepklname(tilepkl_filename):
    """
    get the slide_id from previous made tiles pkl's filename
    """
    slide_id = tilepkl_filename.split('-(')[0]
    return slide_id


''' ----------- producing label file functions ----------- '''

def produce_labels_csv_for_tcgalung(ENV_task):
    """
    Process the metadata CSV file and generate a new CSV file with the required format.
    
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
    """
    
    input_csv = os.path.join(ENV_task.PROJECT_META_DIR, 'classes(old metadata).csv')
    output_csv = os.path.join(ENV_task.PROJECT_META_DIR, f'{ENV_task.ROOT_NAME}-annotations.csv')
    
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header for the output CSV
        # writer.writerow(['slide_id', 'label'])
        
        for row in reader:
            label, slide_name = row
            # Extract slide_id by removing the '.svs' suffix
            slide_id = slide_name.replace('.svs', '')
            # Convert label to binary format
            binary_label = '1' if label == 'LUAD' else '0'
            # Write the processed row to the output CSV
            writer.writerow([slide_id, binary_label])
            
def produce_labels_csv_for_ovbev(ENV_task):
    """
    Function to generate a CSV file with slide_id and label for all .svs files in subfolders.
    
    Args:
    root_folder (str): Path to the root folder containing subfolders with .svs files.
    output_csv (str): Path to the output CSV file.
    
    Returns:
    None
    """
    
    root_folder = ENV_task.SLIDE_DIR
    output_csv = os.path.join(ENV_task.PROJECT_META_DIR, 'OV-Bev_Resp-annotations.csv')
    
    # List to hold the rows for the CSV
    rows = []
    
    slide_ids = []
    # Iterate through each subfolder in the root folder
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        
        # Ensure we are only processing directories
        if os.path.isdir(subfolder_path):
            # Determine the label based on the subfolder name
            if subfolder.startswith('e'):
                label = 1
            elif subfolder.startswith('in'):
                label = 0
            else:
                # Skip folders that do not match the criteria
                continue
            
            # Iterate through each file in the subfolder
            for file in os.listdir(subfolder_path):
                if file.endswith('.svs'):
                    # Extract slide_id (file name without extension)
                    slide_id = os.path.splitext(file)[0]
                    slide_ids.append(slide_id)
                    
                    # Append the slide_id and label to the rows list
                    rows.append([slide_id, label])
                    print(f'found slide and label: {[slide_id, label]}')
    
    # Write the rows to the CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        # csvwriter.writerow(['slide_id', 'label'])
        
        # Write the rows
        csvwriter.writerows(rows)
    
    print(f'row number: {len(rows)}, no repeat cases: {len(set(slide_ids))}')
    print(f"CSV annotations file has been generated at {output_csv}")
            
def produce_labels_csv_for_colitis(ENV_task):
    """
    the label of colitis dataset is already prepared in the dataset folder by the folder names
    just need to parse the folder names and allocate the label for each file in them
    so we can produce the label csv file locally
    """
    label_asign_folder = os.path.join(ENV_task.DATA_ROOT_DIR, 'dataset')
    output_csv = os.path.join(ENV_task.PROJECT_META_DIR, 'A_TAP_AbCD_cohort-annotations.csv')
    label_dir_names = ['label-0', 'label-2', 'label-3', 'label-4', 'label-5']
    
    # Map label folder names to numerical labels
    label_mapping = {name: idx for idx, name in enumerate(label_dir_names)}
    
    # Initialize a list to store the slide_id and label pairs
    slide_labels = []
    
    # Iterate through each label folder
    for label_dir in label_dir_names:
        current_folder = os.path.join(label_asign_folder, label_dir)
        label = label_mapping[label_dir]
        
        # Get all .tiff files in the current folder
        for slide_file in glob.glob(os.path.join(current_folder, '*.tiff')):
            # Extract the slide_id
            slide_id = os.path.basename(slide_file).replace('.tiff', '')
            slide_labels.append([slide_id, label])
    
    # Write the slide_labels to the output CSV file
    with open(output_csv, 'w') as f:
        for slide_id, label in slide_labels:
            f.write(f'{slide_id},{label}\n')
    
    print(f'Labels have been written to {output_csv}')
    
def produce_labels_csv_for_focus(ENV_task):
    """
    Parse the label of focus dataset from case_id <-> label to slide_id <-> label
    """
    tilelist_folder = ENV_task.TILE_PKL_DIR
    enric_meta_csv = os.path.join(ENV_task.PROJECT_META_DIR, 'Focus_KRAS_PrimaryResections.csv')
    output_csv = os.path.join(ENV_task.PROJECT_META_DIR, 'FOCUS-colorectal-annotations.csv')

    # read enric_meta_csv
    meta_df = pd.read_csv(enric_meta_csv)
    # creat mapping from scort_id to label
    id_to_label = {row['scort_id']: 1 if row['KRAS'] == 'Mut' else 0 for index, row in meta_df.iterrows()}

    # initialize the list for slide_id and label
    slide_labels = []

    for tile_file in glob.glob(os.path.join(tilelist_folder, '*-tiles.pkl')):
        # get slide_id
        slide_id = os.path.basename(tile_file).replace('-tiles.pkl', '')
        # get its scort_id
        scort_id = slide_id.split('_')[0]
        # parse scort_id for label
        if scort_id in id_to_label:
            label = id_to_label[scort_id]
            slide_labels.append([slide_id, label])

    # write slide_labels into output_csv
    with open(output_csv, 'w') as f:
        for slide_id, label in slide_labels:
            f.write(f'{slide_id},{label}\n')

    print(f'Labels have been written to {output_csv}')
    
    
def restore_labels_csv_for_camelyon(ENV_task):
    """
    camelyon's label is naturely exist.
    just parse them and restore to csv file
    """
    train_norm_slide_dir = os.path.join(ENV_task.SLIDE_DIR, 'training/normal')
    train_tum_slide_dir = os.path.join(ENV_task.SLIDE_DIR, 'training/tumor')
    test_slide_dir = os.path.join(ENV_task.SLIDE_DIR, 'testing/images')
    test_anno_ref_dir = os.path.join(ENV_task.SLIDE_DIR, 'testing/lesion_annotations')
    output_csv = os.path.join(ENV_task.PROJECT_META_DIR, 'Feature_camelyon16-annotations.csv')
    
    slide_labels = []

    # Process training slides with normal label (0)
    for slide_file in glob.glob(os.path.join(train_norm_slide_dir, '*.tif')):
        slide_id = os.path.basename(slide_file).replace('.tif', '')
        slide_labels.append([slide_id, 0])
    
    # Process training slides with tumor label (1)
    for slide_file in glob.glob(os.path.join(train_tum_slide_dir, '*.tif')):
        slide_id = os.path.basename(slide_file).replace('.tif', '')
        slide_labels.append([slide_id, 1])

    # Process testing slides
    test_anno_files = set(os.path.basename(f).replace('.xml', '') for f in glob.glob(os.path.join(test_anno_ref_dir, '*.xml')))
    
    for slide_file in glob.glob(os.path.join(test_slide_dir, '*.tif')):
        slide_id = os.path.basename(slide_file).replace('.tif', '')
        label = 1 if slide_id in test_anno_files else 0
        slide_labels.append([slide_id, label])

    # Write the slide_labels to the output CSV file
    with open(output_csv, 'w') as f:
        for slide_id, label in slide_labels:
            f.write(f'{slide_id},{label}\n')
    
    print(f'Labels have been written to {output_csv}')
    
    
def discard_redundant_files_focus(ENV_task, check_folders=['data/tilelists',
                                                           'data/tensors-resnet-xn512',
                                                           'data/tensors-uni-xn1024',
                                                           'data/tensors-giga-xn1536'
                                                           ]):
    '''
    '''
    data_dir = ENV_task.DATA_ROOT_DIR
    label_csv = os.path.join(ENV_task.PROJECT_META_DIR, 'FOCUS-colorectal-annotations.csv')
    
    # Read the label CSV to get the list of slide IDs with labels
    labeled_slides = set()
    with open(label_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            slide_id = row[0]
            labeled_slides.add(slide_id)
    
    # Iterate over each check folder
    for folder in check_folders:
        full_folder_path = os.path.join(data_dir, folder)
        
        # Get all files in the folder
        for file_path in glob.glob(os.path.join(full_folder_path, '*')):
            file_name = os.path.basename(file_path)
            # slide_id = os.path.splitext(file_name)[0]  # Get slide_id by removing the file extension
            slide_id = file_name.split('-')[0]
            
            # If slide_id is not in the labeled slides, delete the file
            if slide_id not in labeled_slides:
                os.remove(file_path)
                print(f'Removed file: {file_path}')

    print('Redundant files have been removed.')
    

''' ------ split train/validation/test set for slide_id ------ '''
def split_train_val_test_to_pkl_ovbev(ENV_task, f, train_ratio=0.8, val_ratio=0.0):
    '''
    Function to split dataset into train, val, and test sets.
    
    Args:
        f: fold number

    PS: train_ratio=0.7, val_ratio=0.1, test_ratio=0.2
    '''
    
    csv_file = os.path.join(ENV_task.PROJECT_META_DIR, f'{ENV_task.ROOT_NAME}-annotations.csv')
    
    # Read the slide_id and label from the CSV file
    data = defaultdict(list)
    
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            slide_id, label = row[0], int(row[1])
            data[label].append(slide_id)
    
    # Split the data by label
    train_data, val_data, test_data = [], [], []
    
    for label, slide_ids in data.items():
        print(len(slide_ids))
        random.shuffle(slide_ids)
        
        total = len(slide_ids)
        train_end = math.floor(total * train_ratio)
        val_end = train_end + math.floor(total * val_ratio)
        
        train_data.extend(slide_ids[:train_end])
        val_data.extend(slide_ids[train_end:val_end])
        test_data.extend(slide_ids[val_end:])
    
    # Create the dictionary
    split_dict = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    print(split_dict)
    print(len(split_dict['train']), len(split_dict['val']), len(split_dict['test']))
    
    # Save the dictionary as a .pkl file
    output_file = os.path.join(ENV_task.PROJECT_META_DIR, f'fold{f}.pkl')
    with open(output_file, 'wb') as file:
        pickle.dump(split_dict, file)
    
    print(f"Dataset split saved to {output_file}") 
    
def split_train_val_test_to_pkl_tcga(ENV_task, f, train_ratio=0.8, val_ratio=0.0):
    '''
    Function to split dataset into train, val, and test sets.
    keep sample (with same prefix, before '-DX') from same case be in the same set (train, val or test)
    
    Args:
        f: fold number

    PS: train_ratio=0.7, val_ratio=0.1, test_ratio=0.2
    '''
    
    csv_file = os.path.join(ENV_task.PROJECT_META_DIR, f'{ENV_task.ROOT_NAME}-annotations.csv')
    
    # Read the slide_id and label from the CSV file
    data = defaultdict(list)
    
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            slide_id, label = row[0], int(row[1])
            # Extract prefix from slide_id
            prefix = slide_id.split('-DX')[0]
            data[label].append((prefix, slide_id))
    
    # Split the data by label
    train_data, val_data, test_data = [], [], []
    
    print(f'The number of slide: 0: {len(data[0])}, 1: {len(data[1])}')
    for label, prefix_slide_ids in data.items():
        # Group prefix_slide_ids by prefix
        prefix_dict = defaultdict(list)
        for prefix, slide_id in prefix_slide_ids:
            prefix_dict[prefix].append(slide_id)
        
        # Shuffle the prefixes
        prefixes = list(prefix_dict.keys())
        random.shuffle(prefixes)
        print(f'The number of prefixes for label{label}: {len(prefixes)}')
        
        # Split prefixes into train, val, and test
        total = len(prefixes)
        train_end = math.floor(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_prefixes = prefixes[:train_end]
        val_prefixes = prefixes[train_end:val_end]
        test_prefixes = prefixes[val_end:]
        
        # Collect prefix_slide_ids based on prefix split
        for prefix in train_prefixes:
            train_data.extend(prefix_dict[prefix])
        for prefix in val_prefixes:
            val_data.extend(prefix_dict[prefix])
        for prefix in test_prefixes:
            test_data.extend(prefix_dict[prefix])
    
    # Create the dictionary
    split_dict = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    # print(split_dict)
    print(len(split_dict['train']), len(split_dict['val']), len(split_dict['test']))
    
    # Save the dictionary as a .pkl file
    output_file = os.path.join(ENV_task.PROJECT_META_DIR, f'fold{f}.pkl')
    with open(output_file, 'wb') as file:
        pickle.dump(split_dict, file)
    
    print(f"Dataset split saved to {output_file}")
    
def split_train_val_test_to_pkl_new_colitis(ENV_task, f, train_ratio=0.8, val_ratio=0.0):
    '''
    Function to split dataset into train, val, and test sets.
    Keep sample (with same prefix, before '_') from same case be in the same set (train, val or test).
    
    Args:
        f: fold number

    PS: train_ratio=0.8, val_ratio=0.0, test_ratio=0.2
    '''
    
    csv_file = os.path.join(ENV_task.PROJECT_META_DIR, f'{ENV_task.ROOT_NAME}-annotations.csv')
    
    # Read the slide_id and label from the CSV file
    data = defaultdict(list)
    
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            slide_id, label = row[0], int(row[1])
            # Extract prefix from slide_id
            prefix = slide_id.split('_')[0]
            data[label].append((prefix, slide_id))
    
    # Split the data by label
    train_data, val_data, test_data = [], [], []
    
    print(f'The number of slides: 0: {len(data[0])}, 1: {len(data[1])}')
    for label, prefix_slide_ids in data.items():
        # Group prefix_slide_ids by prefix
        prefix_dict = defaultdict(list)
        for prefix, slide_id in prefix_slide_ids:
            prefix_dict[prefix].append(slide_id)
        
        # Shuffle the prefixes
        prefixes = list(prefix_dict.keys())
        random.shuffle(prefixes)
        print(f'The number of prefixes for label {label}: {len(prefixes)}')
        
        # Split prefixes into train, val, and test
        total = len(prefixes)
        train_end = math.floor(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_prefixes = prefixes[:train_end]
        val_prefixes = prefixes[train_end:val_end]
        test_prefixes = prefixes[val_end:]
        
        # Collect prefix_slide_ids based on prefix split
        for prefix in train_prefixes:
            train_data.extend(prefix_dict[prefix])
        for prefix in val_prefixes:
            val_data.extend(prefix_dict[prefix])
        for prefix in test_prefixes:
            test_data.extend(prefix_dict[prefix])
    
    # Create the dictionary
    split_dict = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    print(len(split_dict['train']), len(split_dict['val']), len(split_dict['test']))
    
    # Save the dictionary as a .pkl file
    output_file = os.path.join(ENV_task.PROJECT_META_DIR, f'fold{f}.pkl')
    with open(output_file, 'wb') as file:
        pickle.dump(split_dict, file)
    
    print(f"Dataset split saved to {output_file}")
    
def split_train_val_test_to_pkl_focus(ENV_task, f, train_ratio=0.8, val_ratio=0.0):
    '''
    Function to split dataset into train, val, and test sets.
    Keep sample (with same prefix, before '_HE') from same case be in the same set (train, val or test).
    
    Args:
        f: fold number

    PS: train_ratio=0.8, val_ratio=0.0, test_ratio=0.2
    '''
    
    csv_file = os.path.join(ENV_task.PROJECT_META_DIR, f'{ENV_task.ROOT_NAME}-annotations.csv')
    
    # Read the slide_id and label from the CSV file
    data = defaultdict(list)
    
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            slide_id, label = row[0], int(row[1])
            # Extract prefix from slide_id
            prefix = slide_id.split('_HE')[0]
            data[label].append((prefix, slide_id))
    
    # Split the data by label
    train_data, val_data, test_data = [], [], []
    
    print(f'The number of slides: 0: {len(data[0])}, 1: {len(data[1])}')
    for label, prefix_slide_ids in data.items():
        # Group prefix_slide_ids by prefix
        prefix_dict = defaultdict(list)
        for prefix, slide_id in prefix_slide_ids:
            prefix_dict[prefix].append(slide_id)
        
        # Shuffle the prefixes
        prefixes = list(prefix_dict.keys())
        random.shuffle(prefixes)
        print(f'The number of prefixes for label {label}: {len(prefixes)}')
        
        # Split prefixes into train, val, and test
        total = len(prefixes)
        train_end = math.floor(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_prefixes = prefixes[:train_end]
        val_prefixes = prefixes[train_end:val_end]
        test_prefixes = prefixes[val_end:]
        
        # Collect prefix_slide_ids based on prefix split
        for prefix in train_prefixes:
            train_data.extend(prefix_dict[prefix])
        for prefix in val_prefixes:
            val_data.extend(prefix_dict[prefix])
        for prefix in test_prefixes:
            test_data.extend(prefix_dict[prefix])
    
    # Create the dictionary
    split_dict = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    print(len(split_dict['train']), len(split_dict['val']), len(split_dict['test']))
    
    # Save the dictionary as a .pkl file
    output_file = os.path.join(ENV_task.PROJECT_META_DIR, f'fold{f}.pkl')
    with open(output_file, 'wb') as file:
        pickle.dump(split_dict, file)
    
    print(f"Dataset split saved to {output_file}")  
    
    
def split_train_val_test_to_pkl_camelyon(ENV_task, f):
    '''
    Function to split dataset into predefined train and test sets.
    slide_id prefix before '_' determines the set (train or test).
    
    Args:
        f: fold number
    '''
    
    csv_file = os.path.join(ENV_task.PROJECT_META_DIR, f'{ENV_task.ROOT_NAME}-annotations.csv')
    
    # Read the slide_id and label from the CSV file
    train_data, test_data = [], []
    
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            slide_id, label = row[0], int(row[1])
            # Extract prefix from slide_id
            prefix = slide_id.split('_')[0]
            if prefix.lower() == 'test':
                test_data.append(slide_id)
            else:
                train_data.append(slide_id)
    
    # Create the dictionary
    split_dict = {
        'train': train_data,
        'val': [],  # Empty list for validation set
        'test': test_data
    }
    
    print(f'Train set size: {len(split_dict["train"])}, Test set size: {len(split_dict["test"])}')
    
    # Save the dictionary as a .pkl file
    output_file = os.path.join(ENV_task.PROJECT_META_DIR, f'fold{f}.pkl')
    with open(output_file, 'wb') as file:
        pickle.dump(split_dict, file)
    
    print(f"Dataset split saved to {output_file}")  


''' ----------- downloading dataset functions ---------- '''

def download_gtex_tissues(ENV_task, gtex_meta_csv_name):
    '''
    warning: only running on linux
    '''
    csv_liver_filepath = os.path.join(ENV_task.METADATA_REPO_DIR, gtex_meta_csv_name)
    liver_slides_download_dir = ENV_task.PARSE_REPO_DATA_SLIDE_DIR
    df_liver_meta = pd.read_csv(csv_liver_filepath)
    
    tissue_ids = []
    for i in range(len(df_liver_meta)):
        tissue_ids.append(df_liver_meta.loc[i]['Tissue Sample ID'])
    
    tissue_ids = tissue_ids[:3]
    for i, t_id in enumerate(tissue_ids):
        download_time = Time()
        print('>>> downloading image: {}'.format(t_id))
        
        # like: wget -P /well/rittscher/users/lec468/LIVER_NASH/example_dx/slides/ https://brd.nci.nih.gov/brd/imagedownload/GTEX-111VG-0826
        os.system('wget -P {}/ https://brd.nci.nih.gov/brd/imagedownload/{} --no-check-certificate'.format(liver_slides_download_dir, t_id))
        # like: mv /well/rittscher/projects/LIVER_NASH_project/example_dx/slides/GTEX-11DYG-1726 /well/rittscher/projects/LIVER_NASH_project/example_dx/slides/GTEX-11DYG-1726.svs
        os.system('mv {}/{} {}/{}.svs'.format(liver_slides_download_dir, t_id, liver_slides_download_dir, t_id))
        print("### download image to: {}/{}.svs, time: {}".format(liver_slides_download_dir, t_id, str(download_time.elapsed() ) ) )
        
''' check damaged slides and re-download it until it's successfully transmitted '''
def check_slides_damaged(slide_filepath):
    ''' check if a tissue slides is damaged '''
    slide_damaged = False
    try:
        _ = openslide.open_slide(slide_filepath)
    except:
        slide_damaged = True
        print('found damaged slide: %s' % slide_filepath)
    return slide_damaged
def list_all_damaged_slides(ENV_task):
    ''' return a list of names of all damaged  '''
    slide_filenames = os.listdir(ENV_task.PARSE_REPO_DATA_SLIDE_DIR)
    damaged_slide_names = []
    for slide_name in slide_filenames:
        if check_slides_damaged(os.path.join(ENV_task.PARSE_REPO_DATA_SLIDE_DIR, slide_name)):
            damaged_slide_names.append(slide_name)
    print('collect damaged slides list:', damaged_slide_names)
    
    return damaged_slide_names

def forced_download_gtex_tissues(ENV_task, gtex_meta_csv_name, redownload_try=10):
    '''
    '''
    csv_liver_filepath = os.path.join(ENV_task.METADATA_REPO_DIR, gtex_meta_csv_name)
    liver_slides_download_dir = ENV_task.PARSE_REPO_DATA_SLIDE_DIR
    df_liver_meta = pd.read_csv(csv_liver_filepath)
    
    tissue_ids = []
    for i in range(len(df_liver_meta)):
        tissue_ids.append(df_liver_meta.loc[i]['Tissue Sample ID'])
        
    cannot_download_slide_list = []
    for i, t_id in enumerate(tissue_ids):
        download_time = Time()
        
        tissue_slide_path = '{}/{}.svs'.format(liver_slides_download_dir, t_id)
        
        need_redownload = True
        if os.path.exists(tissue_slide_path):
            if check_slides_damaged(tissue_slide_path) == False:
                print('### slide: {} successfully download already.'.format(t_id + '.svs'))
                need_redownload = False
            else:
                os.system('rm {}'.format(tissue_slide_path))
                print('> remove damaged slide: {}'.format(t_id + '.svs'))
        else:
            print('! not found slide: {}'.format(t_id + '.svs'))
            
        if need_redownload == False:
            continue
        
        redownload_try_times, redownload_success = 0, False
        while redownload_try_times < redownload_try and redownload_success == False:
            os.system('wget -P {}/ https://brd.nci.nih.gov/brd/imagedownload/{} --no-check-certificate'.format(liver_slides_download_dir, t_id))
            os.system('mv {}/{} {}/{}.svs'.format(liver_slides_download_dir, t_id, liver_slides_download_dir, t_id))
            if check_slides_damaged(tissue_slide_path) is False:
                redownload_success = True
                print("### successfully repair image to: {}/{}.svs".format(liver_slides_download_dir, t_id) )
            else:
                os.system('rm {}'.format(tissue_slide_path) )
                print('! redownload unsuccessful on slide: {}'.format(t_id + '.svs'))
            redownload_try_times += 1
        if redownload_success == False:
            print('!!! tried many times, still cannot download slide: {}'.format(t_id + '.svs'))
            cannot_download_slide_list.append(t_id)
            
        print('used time: {}'.format(str(download_time.elapsed() ) ) )
    
    return cannot_download_slide_list
            
    
def _run_parsedir_move_TCGA_slide(ENV_task):
    
    _env_original_data_dir = ENV_task.ORIGINAL_REPO_DATA_DIR
    _env_parse_data_slide_dir = ENV_task.PARSE_REPO_DATA_SLIDE_DIR
    _env_task_name = ENV_task.TASK_NAME
    
    slide_path_list = parse_filesystem_slide(_env_original_data_dir)
    move_TCGA_download_file_rename_batch(_env_parse_data_slide_dir,
                                         _env_task_name,
                                         slide_path_list, mode='copy')
    
def _run_download_gtex_liver_tissues(ENV_task):
    
    gtex_meta_csv_name = 'GTEx_liver_samples.csv'
    download_gtex_tissues(ENV_task, gtex_meta_csv_name)
    
def _run_download_gtex_pancreas_tissues(ENV_task):
    
    gtex_meta_csv_name = 'GTEx-Pancreas_Saponification.csv'
    download_gtex_tissues(ENV_task, gtex_meta_csv_name)
    
def _run_forced_repair_gtex_liver_tissues(ENV_task):
    
    gtex_meta_csv_name = 'GTEx_liver_samples.csv'
    cannot_download_slide_list = forced_download_gtex_tissues(ENV_task, gtex_meta_csv_name)
    print(cannot_download_slide_list)
    
def _run_forced_repair_gtex_pancreas_tissues(ENV_task):
    
    gtex_meta_csv_name = 'GTEx-Pancreas_Saponification.csv'
    cannot_download_slide_list = forced_download_gtex_tissues(ENV_task, gtex_meta_csv_name)
    print(cannot_download_slide_list)
    
        

if __name__ == '__main__':
    # case_id = parse_slide_caseid_from_filepath('F:\\OV-Bev_Resp\\data\\slides\\e1\\1427159G-Y.svs')
    # print(case_id)
    
    # produce_labels_csv_for_ovbev(env_ov_bev.ENV_OV_BEV)
    # for f in range(10):
    #     split_train_val_test_to_pkl_ovbev(env_ov_bev.ENV_OV_BEV_RES_RES64, f)
        
    # produce_labels_csv_for_tcgalung(env_tcga_lung.ENV_TCGA_LUNG)
    # for f in range(5):
    #     split_train_val_test_to_pkl_ovbev(env_tcga_lung.ENV_TCGA_LUNG_RES, f)
        # split_train_val_test_to_pkl_tcga(env_tcga_lung.ENV_TCGA_LUNG_RES, f)
        
    for f in range(10):
        split_train_val_test_to_pkl_new_colitis(env_colitis_marsh.ENV_COLITIS_ORG_RES512, f)
    for f in range(5):
        split_train_val_test_to_pkl_focus(env_focus_kras.ENV_FOCUS_ORG_RES512, f)
        split_train_val_test_to_pkl_camelyon(env_camelyon_tumor.ENV_CAMELYON_ORG_RES512, f)
    
    pass
    
    
    
    
    
    
    