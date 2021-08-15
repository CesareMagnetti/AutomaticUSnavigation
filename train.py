from networks.Qnetworks import setup_networks
from options.options import gather_options, print_options
import torch
import numpy as np
import random

from utils import train

if __name__=="__main__":
        # 1. gather options
        parser = gather_options(phase="train")
        config = parser.parse_args()
        config.use_cuda = torch.cuda.is_available()
        config.device = torch.device("cuda:{}".format(config.gpu_device) if config.use_cuda else "cpu")
        print_options(config, parser)

        # manual seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

        # 2. instanciate Qnetworks
        qnetwork_local, qnetwork_target = setup_networks(config)
        # 3. launch training
        train(config, qnetwork_local, qnetwork_target, name=config.name, sweep=False)

