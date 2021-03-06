import torch, os
import torch.nn as nn

def setup_networks(config):
    # setup correct channels if we are location aware
    nchannels = 1 if not config.location_aware else 4
    # 1. instanciate the Qnetworks
    if config.default_Q.lower() == "small":
        if config.load_size == 128:
            first_downsampling = 2
        elif config.load_size == 256:
            first_downsampling = 4
        else:
            raise ValueError('small Q-network was only thought for load_size == 128 or load_size == 256')
        params = [(nchannels, config.load_size, config.load_size), config.action_size, config.n_agents, 3,
                   4, 32, not config.no_dropout_Q, not config.no_batchnorm_Q, first_downsampling]
    elif config.default_Q.lower() == "large":
        params = [(nchannels, config.load_size, config.load_size), config.action_size, config.n_agents, 6,
                   2, 4, not config.no_dropout_Q, not config.no_batchnorm_Q]
    else:
        params = [(nchannels, config.load_size, config.load_size), config.action_size, config.n_agents, config.n_blocks_Q,
                   config.downsampling_Q, config.n_features_Q, not config.no_dropout_Q, not config.no_batchnorm_Q]
    # instanciate and send to gpu
    if config.dueling:
        if config.recurrent:
            params += [config.recurrent_history_len]
            qnetwork_local = RecurrentDuelingQNetwork(*params).to(config.device)
            qnetwork_target = RecurrentDuelingQNetwork(*params).to(config.device)
        else:
            qnetwork_local = DuelingQNetwork(*params).to(config.device)
            qnetwork_target = DuelingQNetwork(*params).to(config.device)
    else:
        if config.recurrent:
            params += [config.recurrent_history_len]
            qnetwork_local = RecurrentQnetwork(*params).to(config.device)
            qnetwork_target = RecurrentQnetwork(*params).to(config.device)
        else:
            qnetwork_local = SimpleQNetwork(*params).to(config.device)
            qnetwork_target = SimpleQNetwork(*params).to(config.device)
    # we keep networks in evaluation mode at all times, when training is needed, .train() will be called on the local network
    qnetwork_local.eval()
    qnetwork_target.eval()
    print("Qnetwork instanciated: {} params.\n".format(qnetwork_local.count_parameters()), qnetwork_local)
    # 2. load from checkpoint if needed
    if config.load is not None:
        if config.load_name is not None:
            load_name = config.load_name
        else:
            load_name = config.name
        print("loading: {}/{} model ...".format(load_name, config.load))
        qnetwork_local.load(os.path.join(config.checkpoints_dir, load_name, config.load+".pth"))
        qnetwork_target.load(os.path.join(config.checkpoints_dir, load_name, config.load+".pth"))
    return qnetwork_local, qnetwork_target


# ===== BUILDING BLOCKS =====

class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, kernel_size, stride, padding, dropout, batchnorm):
        super(ConvBlock, self).__init__()
        # conv layer
        block = [nn.Conv2d(inChannels, outChannels, kernel_size, stride, padding)]
        # batchnorm layer
        if batchnorm:
            block+=[nn.BatchNorm2d(outChannels)]
        # activation
        block+=[nn.ReLU(inplace=True)]
        # dropout
        if dropout:
            block+=[nn.Dropout(0.5)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class HeadBlock(nn.Module):
    def __init__(self, inFeatures, actionSize, dropout, batchnorm):
        super(HeadBlock, self).__init__()
        # linear layer
        block = [nn.Linear(inFeatures, inFeatures//4)]
        # batchnorm layer
        if batchnorm:
            block+=[nn.BatchNorm1d(inFeatures//4)]
        # activation
        block+=[nn.ReLU()]
        # dropout
        if dropout:
            block+=[nn.Dropout(0.5)]
        # output layer
        block+=[nn.Linear(inFeatures//4, actionSize)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

# ===== DQN NETWORK =====

class SimpleQNetwork(nn.Module):
    """
    very simple CNN backbone followed by N heads, one for each agent.
    """
    def __init__(self, state_size, action_size, Nheads, Nblocks=6, downsampling=2, num_features=4, dropout=True, batchnorm=True, first_downsampling = None):
        """
        params
        ======
            state_size (list, tuple): shape of the input as CxHxW
            Nblocks (int): number of convolutional blocks to use in the backbone
            Nheads (int): number of agents sharing the convolutional backbone
            action_size (int): number of actions each agent has to take (i.e. +/- movement in each of 3 dimensions -> 6 actions)
            downsampling (int): downsampling factor of each convolutional bock
            num_features (int): number of filters in the first convolutional block, each following block
                                will go from num_features*2**i  -->  num_features*2**(i+1)
            dropout (bool): if you want to use dropout (p=0.5)
            batchnorm (bool): if you want to use batch normalization
            first_downsampling (int): downsampling factor of the first convolutional bock (if None than downsampling is used)
                                 
        """
        super(SimpleQNetwork, self).__init__()

        # build convolutional backbone
        if first_downsampling is None:
            first_downsampling = downsampling
        cnn = [ConvBlock(state_size[0], num_features, 3, first_downsampling, 1, dropout, batchnorm)]
        for i in range(Nblocks-1):
            cnn.append(ConvBlock(num_features*2**i, num_features*2**(i+1), 3, downsampling, 1, dropout, batchnorm))
        cnn.append(nn.Conv2d(num_features*2**(i+1), num_features*2**(i+2), 3, downsampling, 1))
        self.cnn = nn.Sequential(*cnn)

        # get the shape after the conv layers
        self.num_linear_features = self.cnn_out_dim(state_size)

        # build N linear heads, one for each agent
        heads = []
        for i in range(Nheads):
            heads.append(HeadBlock(self.num_linear_features, action_size, dropout, batchnorm))
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

class DuelingQNetwork(SimpleQNetwork):
    """Implements Dueling Q learning rather than standard Q Learning, inherits most of the Q network
    """
    def __init__(self, state_size, action_size, Nheads, Nblocks=6, downsampling=2, num_features=4, dropout=True, batchnorm=True, first_downsampling=None):
        # initialize Q-network
        super(DuelingQNetwork, self).__init__(state_size, action_size, Nheads, Nblocks, downsampling, num_features, dropout, batchnorm, first_downsampling)
        
        # set the value stream as a linear head on top of the parent convolutional block
        self.value_stream = HeadBlock(self.num_linear_features, 1, dropout, batchnorm)

    def forward(self, x):
        y = self.cnn(x)
        y = y.reshape(x.shape[0], -1)

        # get advantages for each head
        advantages = []
        for head in self.heads:
            advantages.append(head(y))
        
        # get value stream output
        values = self.value_stream(y)

        # aggregate the advantages and value to get the Q values
        outs = []
        for adv in advantages:
            outs.append(values + (adv - adv.mean())) # could also use .max() but original paper uses mean
        
        return outs
    
class RecurrentQnetwork(SimpleQNetwork):
    "adds a recurrent LSTM layer after the convolutional block to consider an history of time-frames when making a decision."
    def __init__(self, state_size, action_size, Nheads, Nblocks=6, downsampling=2, num_features=4, dropout=True, batchnorm=True, first_downsampling=None, history_length=10):       
        # initialize Q-network
        super(RecurrentQnetwork, self).__init__(state_size, action_size, Nheads, Nblocks, downsampling, num_features, dropout, batchnorm, first_downsampling)
        
        # initialize the recurrent layer
        self.history_length = history_length
        self.recurrent_layer = nn.LSTM(self.num_linear_features, self.num_linear_features, batch_first=True)
    
    def forward(self, x):
        """ forward pass through the network
        Params:
        ==========
            x (tensor[B*L x C x H x W]): a sequence of batched image frames (sequence and batch are along the same dimension)
        Outputs:
        ==========
            out (list[tensor[B x action_size]]): Q values for each agent for the current frame 
        """
        #print("x:", x.shape)
        # get batch size of inputs
        B = int(x.shape[0]/self.history_length)
        #print(B)
        # pass all frames through convolutional backbone and reshape for LSTM 
        y = self.cnn(x) # B*L x self.num_linear_features (B*L can be BIG: on 15 envs with B=64 and L=10 -> 15*64*10 = 9600)
        y = y.reshape(B, self.history_length, -1) # B x L x self.num_linear_features
        #print("y:", y.shape)
        # pass these sequential features to the recurrent layer using default (h, c) initialized as zeros
        _, (h_n, _) = self.recurrent_layer(y)
        #print("h_n:", h_n.shape)
        # get outputs for each head
        outs = []
        for head in self.heads:
            outs.append(head(h_n.squeeze(0)))
        #print(["out_i: {}".format(outs[i].shape) for i in range(len(outs))])
        return outs

class RecurrentDuelingQNetwork(DuelingQNetwork):
    "adds a recurrent LSTM layer after the convolutional block to consider an history of time-frames when making a decision."
    def __init__(self, state_size, action_size, Nheads, Nblocks=6, downsampling=2, num_features=4, dropout=True, batchnorm=True, first_downsampling=None, history_length=10):       
        # initialize Q-network
        super(RecurrentDuelingQNetwork, self).__init__(state_size, action_size, Nheads, Nblocks, downsampling, num_features, dropout, batchnorm, first_downsampling)

        # initialize the recurrent layer
        self.history_length = history_length
        self.recurrent_layer = nn.LSTM(self.num_linear_features, self.num_linear_features, batch_first=True)
    
    def forward(self, x):
        """ forward pass through the network
        Params:
        ==========
            x (tensor[Seq_len x B x C x H x W]): a sequence of batched image frames
        Outputs:
        ==========
            out (list[tensor[B x action_size]]): Q values for each agent for the current frame 
        """
        # print("x:", x.shape)
        # get batch size of inputs
        B = int(x.shape[0]/self.history_length)
        # pass all frames through convolutional backbone and reshape for LSTM 
        y = self.cnn(x) # B*L x self.num_linear_features (B*L can be BIG: on 15 envs with B=64 and L=10 -> 15*64*10 = 9600)
        y = y.reshape(B, self.history_length, -1) # B x L x self.num_linear_features
        # print("y:", y.shape)
        # pass these sequential features to the recurrent layer, using default (h, c) initialized as zeros
        _, (h_n, _) = self.recurrent_layer(y)
        # print("h_n:", h_n.shape)
        # get advantages for each head
        advantages = []
        for head in self.heads:
            advantages.append(head(h_n.squeeze(0)))
        
        # get value stream output
        values = self.value_stream(h_n.squeeze(0))

        # aggregate the advantages and value to get the Q values
        outs = []
        for adv in advantages:
            outs.append(values + (adv - adv.mean())) # could also use .max() but original paper uses mean
        #print(["out_i: {}".format(outs[i].shape) for i in range(len(outs))])
        return outs