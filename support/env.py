'''
Created on 2 Apr 2024

@author: laengs2304
'''
import platform

import torch

from support.parames import parames_basic


nb_gpu = 0
if torch.cuda.is_available():
    device = torch.device("cuda")
    nb_gpu = torch.cuda.device_count()
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
        
        
ENV = parames_basic(
        project_name='FlattenPath-V01',
        scale_factor=32,
        tile_size=256,
        tile_threshold_pct=50,
        transform_resize=224,
        pil_image_file_format='.png'
    )

if __name__ == '__main__':
    pass