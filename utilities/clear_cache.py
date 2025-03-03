import gc
import torch

def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()    # Optionally, you can force garbage collection as well 