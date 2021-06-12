from baseEnvironment import BaseEnvironment
from utils import get_model
import torch
import torchvision.transforms as transforms 

gpu_ids = [i for i in range(torch.cuda.device_count())]
device = torch.device('cuda:{}'.format(gpu_ids[0]) if gpu_ids else 'cpu')

class CT2USEnvironment(BaseEnvironment):
    def __init__(self, dataroot, volume_id, segmentation_id, start_pos = None, scale_intensity=True, rewardID=2885,
                 model_name = 'CycleGAN_LPIPS_noIdtLoss_lambda_AB_1', **kwargs):

        BaseEnvironment.__init__(self, dataroot, volume_id, segmentation_id, start_pos, scale_intensity, rewardID, **kwargs)

        # load the queried CT2US model
        self.CT2USmodel = get_model(model_name).to(self.device)

    def CT2US(self, input):
        # add channel dimension if 2D image
        if len(input.shape)<3:
            input = input.unsqueeze(0)
        # add batch dimension if single image
        if len(input.shape)<4:
            input = input.unsqueeze(0)

        # transform to US and return
        return self.CT2USmodel(input.to(self.device)).detach()

    def sampleUS(self):

        # sample the CT plane, reward and the segmentation map
        nextStateCT, reward, segmentation = self.sample()

        # convert the state to ultrasound
        nextStateUS = self.CT2US(nextStateCT)

        return nextStateUS, reward, segmentation


    def step(self, dirA, dirB, dirC):

        # update current position
        self.pointA += dirA
        self.pointB += dirB
        self.pointC += dirC

        # sample the CT plane, reward and the segmentation map
        nextStateUS, reward, segmentation = self.sampleUS()

        return nextStateUS, reward, segmentation