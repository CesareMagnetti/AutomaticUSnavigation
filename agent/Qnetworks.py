import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, kernel_size, stride, padding, norm = nn.BatchNorm2d):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(inChannels, outChannels, kernel_size, stride, padding),
                                   norm(outChannels),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)

class HeadBlock(nn.Module):
    def __init__(self, inFeatures, actionSize, norm=nn.BatchNorm1d):
        super(HeadBlock, self).__init__()
        self.block = nn.Sequential(nn.Linear(inFeatures, inFeatures//2),
                                   norm(inFeatures//2),
                                   nn.ReLU(),
                                   nn.Linear(inFeatures//2, inFeatures//2),
                                   norm(inFeatures//2),
                                   nn.ReLU(),
                                   nn.Linear(inFeatures//2, actionSize))

    def forward(self, x):
        return self.block(x)


# build the DQN network
class SimpleQNetwork(nn.Module):
    """
    very simple CNN backbone followed by N heads, one for each agent.
    """
    def __init__(self, state_size, action_size, Nheads, seed, Nblocks=6, downsampling=2, num_features=8):
        """
        params
        ======
            state_size (list, tuple): shape of the input as CxHxW
            Nblocks (int): number of convolutional blocks to use in the backbone
            Nheads (int): number of agents sharing the convolutional backbone
            action_size (int): number of actions each agent has to take (i.e. +/- movement in each of 3 dimensions -> 6 actions)
            downsampling (int): downsampling factor of each convolutional bock
            num_features (int): number of filters in the first convolutional block, each following block
                                will go from num_features*2**i  -->  num_features*2**(i+1).
                                 
        """
        super(SimpleQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # build convolutional backbone
        cnn = [ConvBlock(state_size[0], num_features, 3, 1, 1)]
        for i in range(Nblocks-1):
            cnn.append(ConvBlock(num_features*2**i, num_features*2**(i+1), 3, downsampling, 1))
        cnn.append(nn.Conv2d(num_features*2**(i+1), num_features*2**(i+2), 3, downsampling, 1))
        self.cnn = nn.Sequential(*cnn)

        # get the shape after the conv layers
        self.num_linear_features = self.cnn_out_dim(state_size)

        # build N linear heads, one for each agent
        heads = []
        for i in range(Nheads):
            heads.append(HeadBlock(self.num_linear_features, action_size))
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        y = self.cnn(x)
        y = y.reshape(x.shape[0], -1)

        # get outputs for each head
        outs = []
        for head in self.heads:
            outs.append(head(y))
        
        return outs
    
    def cnn_out_dim(self, input_dim):
        return self.cnn(torch.zeros(1, *input_dim)).flatten().shape[0]

    def save(self, savepath):
        torch.save(self.state_dict(), savepath)
    
    def load(self, savepath):
        state_dict = torch.load(savepath, map_location='cpu')
        self.load_state_dict(state_dict)


        