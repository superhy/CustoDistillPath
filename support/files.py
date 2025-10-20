'''
@author: yang hu

'''

import csv
import glob
import math
import os
import pickle
import random
import shutil

import openslide

import pandas as pd
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
    
def move_TCGA_download_file_rename_batch(tcga_slide_path_list, parse_data_slide_dir,
                                         mode='move', metadata_dir=None, label_fname_list=[]):
    """
    Move the svs or tif file first,
    and rename according to the case id
    
    Args:
        ENV_task: the task environment object
    
    """
    label_dict_list = []
    for label_fname in label_fname_list:
        label_dict_list.append(query_task_label_dict_fromcsv(metadata_dir, None, label_fname))
    
    if not os.path.exists(parse_data_slide_dir):
        os.makedirs(parse_data_slide_dir)
    else:
        clear_dir([parse_data_slide_dir])
        os.makedirs(parse_data_slide_dir)  
    
    count, filted = 0, 0
    for slide_path in tcga_slide_path_list:
        # parse case id from original slide_path
        case_id = parse_slide_caseid_from_filepath(slide_path)
        # filter the slide if not in all label dicts
        for label_dict in label_dict_list:
            if case_id not in label_dict:
                filted += 1
                print('filted slide: ' + slide_path + ', case id: ' + case_id)
                continue
        
        # parse slide type and id
        slide_type_id = parse_slide_typeid_from_filepath(slide_path)
        suffix = slide_path[len(slide_path) - 4 : ]
        slide_new_name = case_id + slide_type_id + suffix
        print('move slide from: ' + slide_path + ' -> ' + os.path.join(parse_data_slide_dir, slide_new_name))
        move_file(slide_path, os.path.join(parse_data_slide_dir, slide_new_name), mode)
        count += 1
    print('moved {} slide files, filted {} slides.'.format(count, filted))


def move_TCGA_download_file_rename_batch_from_barcode_table(
        tcga_slide_path_list,
        parse_data_slide_dir,
        barcode_table_path,
        mode='move',
        barcode_column='bcr_patient_barcode',
        table_sep=None):
    """
    Move or copy TCGA slide files that match the case ids listed in an external table.

    Args:
        tcga_slide_path_list: iterable of slide file paths to filter.
        parse_data_slide_dir: destination directory for renamed slides.
        barcode_table_path: path to the table containing case ids.  
        mode: 'move' or 'copy', passed to move_file.
        barcode_column: column in the table that stores case ids.
        table_sep: optional delimiter override for delimited text files.
    """
    if not os.path.exists(barcode_table_path):
        raise FileNotFoundError('Barcode table not found: {}'.format(barcode_table_path))

    table_ext = os.path.splitext(barcode_table_path)[1].lower()
    if table_ext in ('.xls', '.xlsx'):
        barcode_df = pd.read_excel(barcode_table_path)
    else:
        if table_sep is not None:
            sep = table_sep
        elif table_ext in ('.tsv', '.txt'):
            sep = '\t'
        else:
            sep = ','
        barcode_df = pd.read_csv(barcode_table_path, sep=sep)

    if barcode_column not in barcode_df.columns:
        raise ValueError('Column "{}" not found in barcode table {}'.format(
            barcode_column, barcode_table_path))

    barcode_values = barcode_df[barcode_column].dropna().astype(str)
    barcode_set = {
        value.strip().upper()
        for value in barcode_values
        if value.strip()
    }

    if not barcode_set:
        print('No valid barcodes found in {}.'.format(barcode_table_path))

    filtered_slide_path_list = []
    for slide_path in tcga_slide_path_list:
        case_id = parse_slide_caseid_from_filepath(slide_path)
        if case_id.upper() in barcode_set:
            filtered_slide_path_list.append(slide_path)

    print('Filtered {} of {} slides using {} unique barcodes from {}.'.format(
        len(filtered_slide_path_list),
        len(tcga_slide_path_list),
        len(barcode_set),
        barcode_table_path))

    move_TCGA_download_file_rename_batch(
        filtered_slide_path_list,
        parse_data_slide_dir,
        mode=mode)

   
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

    
def _run_parsedir_move_TCGA_slide(original_data_dir, parse_data_slide_dir, task_name):
    """
    move and rename the TCGA downloaded slide files in batch
    """
    slide_path_list = parse_filesystem_slide(original_data_dir)
    move_TCGA_download_file_rename_batch(parse_data_slide_dir,
                                         task_name,
                                         slide_path_list, mode='copy')
    

def _run_parsedir_move_TCGA_slide_with_barcode(original_data_dir,
                                               parse_data_slide_dir,
                                               barcode_table_path,
                                               mode='copy',
                                               filter_annotated_slide=True,
                                               barcode_column='bcr_patient_barcode',
                                               table_sep=None):
    """
    Move and rename the TCGA downloaded slide files filtered by a barcode table.
    """
    slide_path_list = parse_filesystem_slide(original_data_dir)
    move_TCGA_download_file_rename_batch_from_barcode_table(
        slide_path_list,
        parse_data_slide_dir,
        barcode_table_path,
        mode=mode,
        barcode_column=barcode_column,
        table_sep=table_sep)


if __name__ == '__main__':
    pass
    
    
    
    
    
    
    
