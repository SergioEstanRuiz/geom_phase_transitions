# Provides a Python interface to GPU management and monitoring functions. 
# This is a wrapper around the NVML library.
import pynvml
import torch
import random
import numpy as np
import os

def get_free_gpu():
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return None

    pynvml.nvmlInit() # Initializes the NVML library, preparing it to interact with NVIDIA GPUs on your system.
    num_devices = pynvml.nvmlDeviceGetCount() # Retrieves the number of NVIDIA GPUs available on the system.
    free_memory = []

    for i in range(num_devices):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i) # Gets a handle to the GPU device at index i.
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle) # Retrieves memory information for the GPU device.
        free_memory.append((i, meminfo.free)) # Appends a tuple of the GPU index and its free memory to the list.

    pynvml.nvmlShutdown() # Shuts down the NVML library, releasing resources and handles associated with it.
    
    return max(free_memory, key=lambda x: x[1])[0] # Returns the index of the GPU with the most free memory.


def set_seed(seed: int = 42):
    random.seed(seed)                      # Python built-in RNG
    np.random.seed(seed)                   # NumPy RNG
    torch.manual_seed(seed)                # PyTorch CPU RNG
    torch.cuda.manual_seed(seed)           # Current GPU RNG
    torch.cuda.manual_seed_all(seed)       # All GPU devices (if multi-GPU)

    # Ensures deterministic algorithms
    torch.backends.cudnn.deterministic = False # Need to change to True for full reproducibility, but may slow down training
    torch.backends.cudnn.benchmark = True # Need to change to False for full reproducibility, but may slow down training

    os.environ["PYTHONHASHSEED"] = str(seed)