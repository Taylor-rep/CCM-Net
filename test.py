import torch
from torch.utils.data import DataLoader
from dataset import NPY_datasets
from tensorboardX import SummaryWriter
from lbunet import CCMNet

from engine import *
import os

from utils import *
from config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

def main(config):
  
    config.work_dir = os.getcwd() + '/'
    log_dir = os.getcwd()
    global logger
    logger = get_logger('test', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=False)
    
    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'CCMNet':
        model = CCMNet()
    else: raise Exception('network in not right!')
    model = model.cuda()
    
    input_path = config.weights_path

    if os.path.exists(input_path):
        print('#----------Testing----------#')
        best_weight = torch.load(input_path, map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        test_one_epoch(
                val_loader,
                model,
                config.criterion,
                logger,
                config,
                path = 'ultimate'
            )


if __name__ == '__main__':
    config = setting_config
    main(config)