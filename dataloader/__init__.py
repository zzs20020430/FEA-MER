# from .mdpe_dataloader import MDPEDataset, get_dataloader as get_mdpe_dataloader, load_data as load_mdpe_data
# from .real_life_dataloader import RealLifeDataset, load_data as load_real_life_data
import argparse
from .dataloader_utils import pad_batch, pad_batch_with_masks, collate_fn
from importlib import import_module
def get_dataloader(dataset_args):               #遍历args.datasets，根据dataset_args中的dataset_name，调用相应的get_dataloader函数
    module_path, class_name=dataset_args.module_path, dataset_args.dataset_class
    module = import_module(module_path)
    return getattr(module, 'get_dataloader')(dataset_args)