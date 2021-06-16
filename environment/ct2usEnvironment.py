from .baseEnvironment import BaseEnvironment
from .utils import get_model
import torch
import torchvision.transforms as transforms 

gpu_ids = [i for i in range(torch.cuda.device_count())]
device = torch.device('cuda:{}'.format(gpu_ids[0]) if gpu_ids else 'cpu')

class CT2USEnvironment(BaseEnvironment):
    def __init__(self, parser, **kwargs):

        BaseEnvironment.__init__(self, parser, **kwargs)

        # load the queried CT2US model
        self.CT2USmodel = get_model(parser.model_name).to(self.device)

    def CT2US(self, input):
        # add channel dimension if 2D image
        if len(input.shape)<3:
            input = input.unsqueeze(0)
        # add batch dimension if single image
        if len(input.shape)<4:
            input = input.unsqueeze(0)

        # transform to US and return
        return self.CT2USmodel(input.to(self.device)).detach()

    def sample(self, return_seg=False, return_CT=False):

        # sample the CT plane, reward and the segmentation map
        sliceCT, segmentation = super().sample(state=self.state, return_seg=return_seg)

        # convert the state to ultrasound (normalize since slice is stored as a uint8)
        sliceUS = self.CT2US(sliceCT/255)

        if return_seg and return_CT:
            return sliceUS, sliceCT, segmentation
        elif return_CT:
            return sliceUS, sliceCT
        elif return_seg:
            return sliceUS, segmentation
        else:
            return sliceUS


    def step(self, increment):

        # update current position
        self.state+=increment

        # get the reward according to the segmentation
        _, segmentation = self.sample(return_seg=True)
        reward = self.get_reward(segmentation)

        return self.state, reward