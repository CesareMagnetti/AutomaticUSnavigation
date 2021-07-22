
import torch, os, wandb, functools
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

from environment.xcatEnvironment import *
from agent.agent import Agent
from buffer.buffer import *
from visualisation.visualizers import Visualizer

# ==== THE FOLLOWING FUNCTIONS HANDLE TRAINING AND TESTING OF THE AGENTs ====
def train(config, local_model, target_model, wandb_entity="us_navigation", sweep=False, rank=0):
        """ Trains an agent on an input environment, given networks/optimizers and training criterions.
        Params:
        ==========
                config (argparse object): configuration with all training options. (see options/options.py)
                local_model (PyTorch model): pytorch network that will be trained using a particular training routine (i.e. DQN)
                                             (if more processes, the Qnetwork is shared)
                target_model (PyTorch model): pytorch network that will be used as a target to estimate future Qvalues. 
                                              (it is a hard copy or a running average of the local model, helps against diverging)
                wandb_entuty (str): which wandb workspace to save logs to. (if unsure use your main workspace i.e. your-user-name)
                sweep (bool): flag if we are performing a sweep, in which case we will not be saving checkpoints as that will occupy too much memory.
                              However we will still save the final model in .onnx format (only intermediate .pth checkpoints are not saved)
                rank (int): indicates the process number if multiple processes are queried
        """ 
        # ==== instanciate useful classes ====

        # manual seed
        torch.manual_seed(config.seed + rank) 
        # 1. instanciate environment
        if not config.location_aware:
            env = SingleVolumeEnvironment(config)
        else:
            env = LocationAwareSingleVolumeEnvironment(config)
        # 2. instanciate agent
        agent = Agent(config)
        # 3. instanciate optimizer for local_network
        optimizer = optim.Adam(local_model.parameters(), lr=config.learning_rate)
        # 4. instanciate criterion
        if "mse" in config.loss.lower():
                criterion = nn.MSELoss(reduction='none')
        elif "smooth" in config.loss.lower():
                criterion = nn.SmoothL1Loss(reduction='none')
        else:
                raise ValueError()
        # 5. instanciate replay buffer
        buffer = PrioritizedReplayBuffer(config.buffer_size, config.batch_size, config.alpha)
        # 6. instanciate visualizer
        visualizer = Visualizer(agent.results_dir)

        # ==== LAUNCH TRAINING ====

        # 1. launch exploring steps if needed
        if agent.config.exploring_steps>0:
                print("random walk to collect experience...")
                env.random_walk(config.exploring_steps, buffer)  
        # 2. initialize wandb for logging purposes
        if config.wandb in ["online", "offline"]:
                wandb.login()
        wandb.init(entity=wandb_entity, config=config, mode=config.wandb, name=config.name)
        config = wandb.config # oddly this ensures wandb works smoothly
        # 3. tell wandb to watch what the model gets up to: gradients, weights, and loss
        wandb.watch(local_model, criterion, log="all", log_freq=config.log_freq)
        # 4. start training
        for episode in tqdm(range(config.n_episodes), desc="training..."):
                logs = agent.play_episode(env, local_model, target_model, optimizer, criterion, buffer)
                # send logs to weights and biases
                if episode % config.log_freq == 0:
                        wandb.log(logs, commit=True)
                # save agent locally and test its current greedy policy
                if episode % config.save_freq == 0:
                        if not sweep:
                                print("saving latest model weights...")
                                local_model.save(os.path.join(agent.checkpoints_dir, "latest.pth"))
                                target_model.save(os.path.join(agent.checkpoints_dir, "episode%d.pth"%episode))
                        # test the greedy policy and send logs
                        out = agent.test_agent(config.n_steps_per_episode, env, local_model)
                        wandb.log(out["wandb"], commit=True)
                        # animate the trajectory followed by the agent in the current episode
                        visualizer.render_frames(out["frames"], "episode%d.gif"%episode)
                        # upload file to wandb
                        wandb.save(os.path.join(visualizer.savedir, "episode%d.gif"%episode))
        # at the end of the training session save the model as .onnx to improve the open sourceness and exchange-ability amongst different ML frameworks
        sample_inputs = torch.tensor(out["frames"][:agent.config.batch_size]) # if location aware this will be already of shape BxCxHxW otherwise this will be BxHxW.
        if len(sample_inputs.shape) == 3:
                sample_inputs = sample_inputs.unsqueeze(1)
        torch.onnx.export(local_model, sample_inputs.float().to(agent.config.device), os.path.join(agent.checkpoints_dir, "DQN.onnx"))
        # upload file to wandb
        wandb.save(os.path.join(agent.checkpoints_dir, "DQN.onnx"))

# ==== THE FOLLOWING FUNCTION HANDLE 2D PLANE SAMPLING OF A 3D VOLUME ====

def convertCoordinates3Dto2D(p1, p2, p3, origin = None):
        """ defines the 2D plane, the basis vectors of the 2D coord systems and the origin of the new coord system.
        """
        # These two vectors are in the plane
        v1 = p3 - p1
        v2 = p2 - p1
        # the cross product is a vector normal to the plane
        n = np.cross(v1, v2)
        # define first basis vector
        ex = np.cross(n, ex)/np.linalg.norm(np.cross(v1, n))
        # define the second basis vector
        ey = np.cross(n, ex)/np.linalg.norm(np.cross(n, ex))
        # we define the origin
        if origin is None:
            origin = np.array([0., 0., 0.])
        # convert the three points
        _out = []
        for point in [p1, p2, p3]:
            x = np.dot(point - origin, ex)
            y = np.dot(point - origin, ey)
            _out.append(np.array([x, y]))
        # return new 2D state representation
        return np.vstack(_out)
        
# ===== THE FOLLOWING FUNCTIONS HANDLE THE CYCLEGAN NETWORK INSTANCIATION AND WEIGHTS LOADING ====

def get_model(name, use_cuda=False):
    # instanciate cyclegan architecture used in CT2UStransfer (this is also the default architecture recommended by the authors)
    model = JohnsonResnetGenerator(1, 1, 64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9)
    state_dict = torch.load(os.path.join(os.getcwd(), "environment", "models", "%s.pth"%name), map_location='cpu')
    model.load_state_dict(state_dict)
    return model

# THE FOLLOWING MODELS WHERE TAKEN FROM THE CYCLEGAN-PIX2PIX REPO: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

class JohnsonResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(JohnsonResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                        norm_layer(ngf * mult * 2),
                        nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

