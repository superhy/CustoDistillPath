'''

@author: yang hu
'''

import csv
import json
import os

def load_json_file(json_filepath):
    with open(json_filepath, 'r') as json_file:
        json_text = json_file.read()
        json_data = json.loads(json_text)
        
        return json_data

        
def trans_slide_meta_dict2csv(csv_filepath, slide_meta_dict):
    """
    some slide meta data parsing
    only for review
    """ 
    with open(csv_filepath, 'w', newline='') as csv_file:
        """
        case id in this function is uu_id
        without prefix 'TCGA-'
        """
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['case uu id', 'file_id', 'file_name', 'data_format', 'file_size'])
        
        for i, case_uu_id in enumerate(slide_meta_dict):
            for j, slide in enumerate(slide_meta_dict[case_uu_id]):
                csv_writer.writerow([case_uu_id, slide['file_id'],
                                     slide['file_name'],
                                     slide['data_format'],
                                     slide['file_size']])


    
"""-----------------------------------------------------------------------------------------"""
        
def query_task_label_dict_fromcsv(ENV_task, task_csv_filename=None):
    f_start_string = ENV_task.TASK_NAME + '-dx'
    if task_csv_filename == None:
        for f in os.listdir(ENV_task.METADATA_REPO_DIR):
            if f.startswith(f_start_string) and f.endswith('.csv'):
                print('automatically find CSV file: {}'.format(f))
                task_csv_filename = f
                break
            
    task_csv_filepath = os.path.join(ENV_task.METADATA_REPO_DIR, task_csv_filename)
    task_label_dict = {}
    with open(task_csv_filepath, 'r', newline='') as task_csv_file:
        csv_reader = csv.reader(task_csv_file)
        for csv_line in csv_reader:
            task_label_dict[csv_line[0]] = int(csv_line[1])
            
    return task_label_dict


if __name__ == '__main__':
    pass
 
    
