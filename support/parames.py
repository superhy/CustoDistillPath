'''
@author: Yang Hu
'''
import os
import platform
from tkinter.constants import SEL


class parames_basic():
    
    def __init__(self, 
                 project_name,
                 scale_factor=32,
                 tile_size=256,
                 tile_threshold_pct=50,
                 transform_resize=224,
                 pil_image_file_format='.png'):
        """
        Args:
            project_name:, 
            project_dir: use project_name construct the project dir path,
            slide_type: dx or tx, default dx,
            scale_factor: scale ratio when visualization,
            tile_h_size: patch size to separate the whole slide image,
            tile_w_size,
            transforms_resize,
            tp_tiles_threshold,
            pil_image_file_format,
            debug_mode
        """
        
        self.OS_NAME = platform.system()
        self.PROJECT_NAME = project_name
        
        ''' some default dirs '''
        if self.OS_NAME == 'Windows':
            if os.environ.get('USERNAME') == 'laengs2304':
                self.PROJECT_DIR = os.path.join(f'D:{os.sep}eclipse-workspace', self.PROJECT_NAME)
            else:
                self.PROJECT_DIR = os.path.join(f'D:{os.sep}/workspace', self.PROJECT_NAME)
        elif self.OS_NAME == 'Darwin':
            self.PROJECT_DIR = os.path.join('/Users/superhy/Documents/workspace', self.PROJECT_NAME)
        else:
            self.PROJECT_DIR = os.path.join('/exafs1/well/rittscher/users/lec468/workspace',
                                            self.PROJECT_NAME)
            
        if self.OS_NAME == 'Windows':
            if os.environ.get('USERNAME') == 'laengs2304':
                self.DATA_DISK = f'D:{os.sep}' # local
                # self.DATA_DISK = 'E:' # SSD
            else:
                self.DATA_DISK = f'D:{os.sep}' # local
                # self.DATA_DISK = 'E:' # STAT
                # self.DATA_DISK = 'F:' # SSD
        elif self.OS_NAME == 'Darwin':
            self.DATA_DISK = '/Volumes/Extreme SSD'
        else:
            self.DATA_DISK = '/exafs1/well/rittscher/projects' # on Linux servers
            
        self.WEIGHTS_ROOT = '/well/rittscher/shared/weights'
            
        self.SCALE_FACTOR = scale_factor
        self.TILE_H_SIZE = tile_size
        self.TILE_W_SIZE = self.TILE_H_SIZE
        self.TP_TILES_THRESHOLD = tile_threshold_pct
        self.TRANSFORMS_RESIZE = transform_resize
        self.PIL_IMAGE_FILE_FORMAT = pil_image_file_format
        
        
class parame_st_task(parames_basic):
    
    def __init__(self,
                 project_name,
                 scale_factor,
                 tile_size,
                 tile_threshold_pct,
                 transform_resize,
                 pil_image_file_format,
                 data_root_name,
                 slide_dir=None,
                 ):
        
        super(parame_st_task, self).__init__(project_name, 
                                             scale_factor,
                                             tile_size,
                                             tile_threshold_pct,
                                             transform_resize,
                                             pil_image_file_format)
        
        self.ROOT_NAME = data_root_name
        self.DATA_ROOT_DIR = os.path.join(self.DATA_DISK, self.ROOT_NAME)
        self.THUMBNAIL_DIR = os.path.join(self.DATA_ROOT_DIR, 'data/thumbnails')
        self.TILE_PKL_DIR = os.path.join(self.DATA_ROOT_DIR, 'data/tilelists')
        if slide_dir is None:
            self.SLIDE_DIR = os.path.join(self.DATA_ROOT_DIR, 'data/slides')
        else:
            self.SLIDE_DIR = slide_dir
        self.TENSOR_PT_DIR = os.path.join(self.DATA_ROOT_DIR, 'data/tensors')
        self.MODEL_DIR = os.path.join(self.DATA_ROOT_DIR, 'models')
        self.VIS_HEATMAP_DIR = os.path.join(self.DATA_ROOT_DIR, 'visualization/heatmap')
        self.VIS_STAT_DIR = os.path.join(self.DATA_ROOT_DIR, 'visualization/statistic')
        
        self.PROJECT_LOGS_DIR = os.path.join(f'{self.PROJECT_DIR}/data/{data_root_name}', 'logs')
        self.PROJECT_META_DIR = os.path.join(f'{self.PROJECT_DIR}/data/{data_root_name}', 'meta')
                

if __name__ == '__main__':
    pass