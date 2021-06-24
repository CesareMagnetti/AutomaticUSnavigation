import torch, os
import torch.nn as nn

def setup_networks(config):
    # manual seed
    torch.manual_seed(config.seed)
    # 1. instanciate the Qnetworks
    if config.default_Q is None:
        params = [(1, config.load_size, config.load_size), config.action_size, config.n_agents, config.seed, config.n_blocks_Q,
                   config.downsampling_Q, config.n_features_Q, config.dropout_Q]
    elif config.default_Q.lower() == "small":
        params = [(1, config.load_size, config.load_size), config.action_size, config.n_agents, config.seed, 3,
                   4, 32, config.dropout_Q]
    elif config.default_Q.lower() == "large":
        params = [(1, config.load_size, config.load_size), config.action_size, config.n_agents, config.seed, 6,
                   2, 4, config.dropout_Q]
    else:
        raise ValueError('unknown param ``--default_Q``: {}. available options: [small, large]'.format(config.default_Q))

    qnetwork_local = SimpleQNetwork(*params).to(config.device)
    qnetwork_target = SimpleQNetwork(*params).to(config.device)
    print("Qnetwork instanciated: {} params.\n".format(qnetwork_local.count_parameters()), qnetwork_local)
    # 2. load from checkpoint if needed
    if config.load is not None:
        print("loading: {} ...".format(config.load))
        qnetwork_local.load(os.path.join(config.checkpoints_dir, config.name, config.load+".pth"))
        qnetwork_target.load(os.path.join(config.checkpoints_dir, config.name, config.load+".pth"))

    return qnetwork_local, qnetwork_target


# ===== BUILDING BLOCKS =====

class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, kernel_size, stride, padding, dropout):
        super(ConvBlock, self).__init__()
        block = [nn.Conv2d(inChannels, outChannels, kernel_size, stride, padding),
                 nn.BatchNorm2d(outChannels),
                 nn.ReLU(inplace=True)]
        if dropout:
            block+=[nn.Dropout(0.5)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class HeadBlock(nn.Module):
    def __init__(self, inFeatures, actionSize, dropout):
        super(HeadBlock, self).__init__()
        block = [nn.Linear(inFeatures, inFeatures//4),
                 nn.BatchNorm1d(inFeatures//4),
                 nn.ReLU()]
        if dropout:
            block+=[nn.Dropout(0.5)]
        block+=[nn.Linear(inFeatures//4, actionSize)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

# ===== DQN NETWORK =====

class SimpleQNetwork(nn.Module):
    """
    very simple CNN backbone followed by N heads, one for each agent.
    """
    def __init__(self, state_size, action_size, Nheads, seed, Nblocks=6, downsampling=2, num_features=4, dropout=True):
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
            dropout (bool): if you want to use dropout (p=0.5)
                                 
        """
        super(SimpleQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # build convolutional backbone
        cnn = [ConvBlock(state_size[0], num_features, 3, downsampling, 1, dropout)]
        for i in range(Nblocks-1):
            cnn.append(ConvBlock(num_features*2**i, num_features*2**(i+1), 3, downsampling, 1, dropout))
        cnn.append(nn.Conv2d(num_features*2**(i+1), num_features*2**(i+2), 3, downsampling, 1))
        self.cnn = nn.Sequential(*cnn)

        # get the shape after the conv layers
        self.num_linear_features = self.cnn_out_dim(state_size)

        # build N linear heads, one for each agent
        heads = []
        for i in range(Nheads):
            heads.append(HeadBlock(self.num_linear_features, action_size, dropout))
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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


        