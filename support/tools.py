'''
@author: Yang Hu
'''

import datetime
import warnings
import functools
import numpy as np


class Time:
    """
    Class for displaying elapsed time.
    """
    
    def __init__(self):
        self.date = str(datetime.date.today())
        self.start = datetime.datetime.now()
    
    def elapsed_display(self):
        time_elapsed = self.elapsed()
        print("Time elapsed: " + str(time_elapsed))
    
    def elapsed(self):
        self.end = datetime.datetime.now()
        time_elapsed = self.end - self.start
        return time_elapsed
    



def deprecated(reason=""):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in a future version. {reason}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

'''
normalization family
'''

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data) + 1e-9) / (_range + 1e-9)

def log_normalization(data):
    return np.log1p(data - np.min(data) + 1e-9) / np.log1p(np.max(data) - np.min(data) + 1e-9)

def quantile_normalization(data, lower_quantile=0.01, upper_quantile=0.99):
    lower = np.quantile(data, lower_quantile)
    upper = np.quantile(data, upper_quantile)
    data_clipped = np.clip(data, lower, upper)
    return (data_clipped - np.min(data_clipped)) / (np.max(data_clipped) - np.min(data_clipped) + 1e-9)

def zscore_normalization(data):
    mean = np.mean(data)
    std = np.std(data) + 1e-9
    zscore_data = (data - mean) / std
    zscore_min, zscore_max = np.min(zscore_data), np.max(zscore_data)
    return (zscore_data - zscore_min) / (zscore_max - zscore_min + 1e-9)

def sigmoid_normalization(data):
    normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-9)
    return 1 / (1 + np.exp(-10 * (normalized - 0.5)))

def normalization_n1_p1(data):
    # Create copies to avoid modifying the original data
    data = np.array(data)
    
    # Initialize normalized data with the same shape
    normalized_data = np.zeros_like(data, dtype=np.float64)
    
    # Normalize positive values
    positive_mask = data > 0
    if np.any(positive_mask):
        pos_data = data[positive_mask]
        pos_min, pos_max = np.min(pos_data), np.max(pos_data)
        # print(pos_min, pos_max)
        normalized_data[positive_mask] = (pos_data - pos_min + 1e-9) / (pos_max - pos_min + 1e-9)
    
    # Normalize negative values
    negative_mask = data < 0
    if np.any(negative_mask):
        neg_data = data[negative_mask]
        neg_min, neg_max = np.min(neg_data), np.max(neg_data) # neg_max is closest to 0, neg_min is the most negative
        # print(neg_min, neg_max)
        normalized_data[negative_mask] = -(np.abs(neg_data) - np.abs(neg_max) + 1e-9) / (np.abs(neg_min) - np.abs(neg_max) + 1e-9)
        
    return normalized_data

''' normalization with a lot of zeros '''

def normalization_ignore_zeros(tensor):
    nonzero_mask = tensor != 0
    nonzero_values = tensor[nonzero_mask]
    if nonzero_values.size == 0:
        return tensor
    _range = np.max(nonzero_values) - np.min(nonzero_values)
    normalized_tensor = np.zeros_like(tensor, dtype=np.float32)
    normalized_tensor[nonzero_mask] = (nonzero_values - np.min(nonzero_values)) / (_range + 1e-9)
    return normalized_tensor

def log_normalization_with_zeros(tensor):
    nonzero_mask = tensor != 0
    nonzero_values = tensor[nonzero_mask]
    if nonzero_values.size == 0:
        return tensor
    log_values = np.log1p(nonzero_values)
    log_min = np.min(log_values)
    log_range = np.max(log_values) - log_min
    normalized_tensor = np.zeros_like(tensor, dtype=np.float32)
    normalized_tensor[nonzero_mask] = (log_values - log_min) / (log_range + 1e-9)
    return normalized_tensor

def quantile_normalization_with_zeros(tensor, lower_quantile=0.01, upper_quantile=0.99):
    nonzero_mask = tensor != 0
    nonzero_values = tensor[nonzero_mask]
    if nonzero_values.size == 0:
        return tensor
    lower = np.quantile(nonzero_values, lower_quantile)
    upper = np.quantile(nonzero_values, upper_quantile)
    clipped_values = np.clip(nonzero_values, lower, upper)
    _range = np.max(clipped_values) - np.min(clipped_values)
    normalized_tensor = np.zeros_like(tensor, dtype=np.float32)
    normalized_tensor[nonzero_mask] = (clipped_values - np.min(clipped_values)) / (_range + 1e-9)
    return normalized_tensor

def sparse_tensor_normalization(tensor):
    '''
    similar with normalization_ignore_zeros, faster
    '''
    nonzero_mask = tensor != 0
    if np.sum(nonzero_mask) == 0:
        return tensor
    max_value = np.max(tensor[nonzero_mask])
    min_value = np.min(tensor[nonzero_mask])
    normalized_tensor = np.where(
        nonzero_mask,
        (tensor - min_value) / (max_value - min_value + 1e-9),
        0
    )
    return normalized_tensor

def sigmoid_gate(data, ratio=20, bias=-0.4):
    '''
    Args:
        data: a float or a nparray of floats
    '''
    return 1 / (1 + np.exp(-(data + bias) * ratio))


def np_info(np_arr, name=None, elapsed=None, full_np_info=False):
    """
    Display information (shape, type, max, min, etc) about a NumPy array.
    *reference the code from: https://github.com/deroneriksson/python-wsi-preprocessing
    
    Args:
      np_arr: The NumPy array.
      name: The (optional) name of the array.
      elapsed: The (optional) time elapsed to perform a filtering operation.
    """
    
    if name is None:
        name = "NumPy Array"
    if elapsed is None:
        elapsed = "---"
    
    if full_np_info is False:
        print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
    else:
        # np_arr = np.asarray(np_arr)
        max = np_arr.max()
        min = np_arr.min()
        mean = np_arr.mean()
        is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
        print("%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
          name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))


if __name__ == '__main__':
    
    
    p1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
#     p1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(sigmoid_gate(p1))
