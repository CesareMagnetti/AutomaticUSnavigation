import torch
import torch.nn as nn
import os
from .CT2UStransfer.CycleGAN.models.networks import ResnetGenerator

gpu_ids = [i for i in range(torch.cuda.device_count())]
device = torch.device('cuda:{}'.format(gpu_ids[0]) if gpu_ids else 'cpu')


def get_model(opt):
    try:
        # instanciate cycle gan architecture used in CT2UStransfer
        model = ResnetGenerator(1, 1, 64, norm_layer=nn.InstanceNorm2d, use_dropout=True, n_blocks=9)
        state_dict = torch.load(os.path.abspath("/models/%s.pth"%opt.env_model), map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(device)
    except:
        raise ValueError('unknown ``opt.env_model`` passed. possible options: <CycleGAN_standard,CycleGAN_noIdtLoss,CycleGAN_LPIPS,CycleGAN_LPIPS_noIdtLoss,CycleGAN_LPIPS_noIdtLoss_lambda_AB_1>')
    
    return model
