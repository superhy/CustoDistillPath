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
                 tissue_stain,
                 data_root_name,
                 slide_dir=None,
                 tile_en_dim=64,
                 tile_exp_dim=64,
                 en_batch_size=128,
                 en_num_loader=8,
                 slide_nb_labels=2,
                 epochs_slide_net=20,
                 lr_slide_net=1e-4,
                 batch_size_slide_net=16,
                 num_loader_slide_net=6,
                 loss_weights=[0.5, 0.5],
                 slide_net_warmup=0,
                 parallel_mode='basic',
                 nb_random_tiles=-1,
                 multi_test=False,
                 lora_r=4,
                 lora_alpha=8,
                 lr_boost=1.0,
                 lr_gamma=0.95,
                 milestones_slide_net=[0.5, 1.0],
                 ):
        
        super(parame_st_task, self).__init__(project_name, 
                                             scale_factor,
                                             tile_size,
                                             tile_threshold_pct,
                                             transform_resize,
                                             pil_image_file_format)
        
        self.TISSUE_STAIN = tissue_stain
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
        
        self.TILE_EN_DIM = tile_en_dim
        self.TILE_EXP_DIM = tile_exp_dim
        self.EN_BATCH_SIZE = en_batch_size # batch size for tile feature encoder
        self.EN_NUM_LOADER = en_num_loader
        if self.OS_NAME == 'Windows':
            self.EN_BATCH_SIZE = int(self.EN_BATCH_SIZE / 2)
            self.EN_NUM_LOADER = int(self.EN_NUM_LOADER / 3)
        
        self.SLIDE_NB_LABELS = slide_nb_labels
        self.EPOCHS_SLIDE_NET = epochs_slide_net
        self.LR_SLIDE_NET = lr_slide_net
        self.BATCH_SIZE_SLIDE_NET = batch_size_slide_net if self.TILE_EXP_DIM < 256 else int(batch_size_slide_net / 2)
        self.NUM_LOADER_SLIDE_NET = num_loader_slide_net
        self.LOSS_WEIGHTS = loss_weights
        # self.SLIDE_NUM_LOADER = slide_num_loader
        # if self.OS_NAME == 'Windows':
        #     # self.BATCH_SIZE_SLIDE_NET = int(self.BATCH_SIZE_SLIDE_NET / 2)
        #     self.SLIDE_NUM_LOADER = int(self.SLIDE_NUM_LOADER / 3)
        self.SLIDE_NET_WARMUP = slide_net_warmup
        self.TORCH_PARALLEL_MODE = parallel_mode # only support for longnet running
        self.NB_RANDOM_TILES = nb_random_tiles # select a part of tiles for training on big ViT (longnet)
        self.MULTI_TEST = multi_test # if multi-time tests for random selecting tiles
        self.LORA_R = lora_r
        self.LORA_ALPHA = lora_alpha
        self.SLIDE_NET_LR_BOOST = lr_boost
        self.SLIDE_NET_LR_GAMMA = lr_gamma
        self.MILESTONES_SLIDE_NET = milestones_slide_net
                

if __name__ == '__main__':
    pass