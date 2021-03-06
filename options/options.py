import argparse
from argparse import Namespace
import os

# parsing arguments
def gather_options(phase="train"):

    parser = argparse.ArgumentParser(description='train/test scripts to launch navigation experiments.')
    # I/O directories and data
    parser.add_argument('--name', '-n', type=str, help='name of the experiment.')
    parser.add_argument('--dataroot', '-r',  type=str, default="/vol/biomedic3/hjr119/DATA/XCAT_VOLUMES/", help='path to the XCAT CT volumes.')
    parser.add_argument('--volume_ids', '-vol_ids', type=str, default='samp0', help='filename(s) of the CT volume(s) comma separated.')
    parser.add_argument('--ct2us_model_name', '-model', type=str, default="bestCT2US",
                        help='filename for the state dict of the ct2us model (.pth) file.\navailable models can be found at ./models')
    parser.add_argument('--results_dir', type=str, default='./results/', help='where to save the trajectory.')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='where to save the trajectory.')
    parser.add_argument('--load', type=str, default=None, help='which model to load from.')
    parser.add_argument('--load_name', type=str, default=None, help='which experiment to load model from.')

    # logs and checkpointing 
    parser.add_argument('--wandb', type=str, default='online', help='handles weights and biases interface.\n'\
                                                                'online: launches online interface.\n'\
                                                                'offline: writes all data to disk for later syncing to a server\n'\
                                                                'disabled: completely shuts off wandb. (default = online)')
    # preprocessing
    parser.add_argument('--load_size', type=int, default=256, help="resolution to load the data. By default 256 isotropic resolution.")
    parser.add_argument('--no_preprocess', action='store_true', help="If you do not want to preprocess the CT volume. (set to uint8 and scale intensities)")
    parser.add_argument('--pmin', type=int, default=150, help="pmin value for xcatEnvironment/intensity_scaling() function.")
    parser.add_argument('--pmax', type=int, default=200, help="pmax value for xcatEnvironment/intensity_scaling() function.")
    parser.add_argument('--nmin', type=int, default=0, help="nmin value for xcatEnvironment/intensity_scaling() function.")
    parser.add_argument('--nmax', type=int, default=255, help="nmax value for xcatEnvironment/intensity_scaling() function.")

    # Qnetwork specs
    parser.add_argument('--default_Q', type=str, default="small", help="give a standard architecure: small -> 3 blocks with stride 4. large -> 6 blocks with stride 2.")
    parser.add_argument('--n_blocks_Q', type=int, default=6, help="number of convolutional blocks in the Qnetwork.")
    parser.add_argument('--downsampling_Q', type=int, default=2, help="downsampling factor of each convolutional layer of the Qnetwork.")
    parser.add_argument('--n_features_Q', type=int, default=4, help="number of features in the first convolutional layer of the Qnetwork.")
    parser.add_argument('--no_dropout_Q', action='store_true', help="if use dropout in the Qnetwork (p=0.5 after conv layers and p=0.1 after linear layers).")
    parser.add_argument('--no_batchnorm_Q', action='store_true', help="if use batch normalization in the Qnetwork.")

    # agent specs
    parser.add_argument('--action_size', type=int, default=6, help="how many action can a single agent perform.\n(i.e. up/down,left/right,forward/backwards,do nothing = 7 in a 3D volume).")
    parser.add_argument('--n_agents', type=int, default=3, help="how many RL agents (heads) will share the same CNN backbone.")

    # termination specs
    parser.add_argument('--termination', type=str, default="oscillate", help="options: <oscillate, learned> whether we terminate the episode when the agent starts to oscillate or if we learn termination with an extra action.")
    parser.add_argument('--termination_history_len', type=int, default=20, help="number of history frames to check oscillations on termination.")
    parser.add_argument('--termination_oscillation_freq', type=int, default=3, help="if in the last ``termination_history_len`` steps there are more than this number of equal planes, terminate the episode for oscillation.")
    
    # reward signal shaping
    parser.add_argument('--goal_centroids', type=str, default="2885,2897,2895", help="centroids of these anatomical tissues will define the goal plane.\n"\
                                                                                     "(default: LV+RV+RA: 2885,2897,2895). if multiple IDs separate by comma.")
    parser.add_argument('--planeDistanceRewardWeight', type=float, default=1., help='relative weight of the plane distance reward if present, see rewards/rewards.py for more info.\n')                                                                                 
    parser.add_argument('--anatomyRewardWeight', type=float, default=1., help='relative weight of the anatomy reward if present, see rewards/rewards.py for more info.\n')
    parser.add_argument('--anatomyRewardIDs', type=str, default="2885,2897,2895", help="segmentation IDs for the anatomical reward, see rewards/rewards.py for more info.\n"\
                                                                             "(default: LV+RV+RA: 2885,2897,2895). if multiple IDs separate by comma.")
    parser.add_argument('--incrementalAnatomyReward', action='store_true', default=True, help="whether the agent is rewarded on the improvement of anatomical content or on the current anatomical content.")
    parser.add_argument('--oobReward', type=float, default=0.01, help='penalize each agent if it steps outside the boundaries of the volume, see rewards/rewards.py for more info.')
    parser.add_argument('--areaRewardWeight', type=float, default=0.01, help='reward the agents if they stay far apart from each other (measuring area of spanned triangle), see rewards/rewards.py for more info.\n'\
                                                                             'This is to prevent them from clustering together, which will yield rough transitions.')
    parser.add_argument('--stopReward', type=float, default=0., help="penalize the agents when they to stop on a bad frame, see rewards/rewards.py for more info.")
    parser.add_argument('--penalize_oob_pixels', action='store_true', help="penalize the agents when they sample slices significantly out of boundary, see rewards/rewards.py for more info.\n"\
                                                                           "It will give a penalty equal to the ratio of oob pixels in a sampled slice.")

    # random seed for reproducibility
    parser.add_argument('--seed', type=int, default=1, help="random seed for reproducibility.")
    # gpu device
    parser.add_argument('--gpu_device', type=int, default=0, help="gpu ID if multiple devices available. (only single device training supported)")
    # flag for easier objective
    parser.add_argument('--easy_objective', action='store_true', help="starts the agent in a plane that should be close to a 4-chamber view.")
    # flag for location aware environment (it will give agents information about their location concatenating binary location maps to the input image through the channel dimension)
    parser.add_argument('--location_aware', action='store_true', help="will emebed the agents with information regarding their relative positions in the sampled slice."\
                                                                    "achieved by concatenating binary location maps along the channels dimension of the input image to the qnetwork.")
    # flag for CT2US pipeline (will convert slices to US before feeding them to RL agents)
    parser.add_argument('--CT2US', action='store_true', help="will launch full pipeline navigating in US domain. Else navigation will take place in the CT/XCAT volume.")
    # flag for realCT inputs. We can evaluate our agents on actual clinical data
    parser.add_argument('--realCT', action='store_true', help="tells the framework we are using actual clinical data (we have no segmentations available etc.)")
    # flag if randomizing image instensities on an episode basis, useful when navigatin in fakeCT to ensure diverse intensity range for generalization
    parser.add_argument('--randomize_intensities', action='store_true', default=False, help="whether to randomize intensities each time we reset the environment class, usefull for generalization in fakeCT navigation.")

    # training options (general)
    parser.add_argument('--starting_episode', type=int, default=0, help="what episode we start training from.")
    parser.add_argument('--n_episodes', type=int, default=2000, help="number of episodes to train the agents for.")
    parser.add_argument('--n_steps_per_episode', type=int, default=250, help="number of steps in each episode.")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size for the replay buffer.")
    parser.add_argument('--buffer_size', type=int, default=50000, help="capacity of the replay buffer.") 
    parser.add_argument('--gamma', type=int, default=0.999, help="discount factor.")
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001, help="learning rate for the q network.")
    parser.add_argument('--loss', type=str, default="SmoothL1", help='which loss to use to optimize the Qnetwork (MSE, SmoothL1).')
    parser.add_argument('--trainer', type=str, default="DoubleDeepQLearning", help='which training routine to use (DeepQLearning, DoubleDeepQLearning...).')
    parser.add_argument('--dueling', action='store_true', help="instanciates a dueling q-network architecture, everything else is the same.")
    parser.add_argument('--recurrent', action='store_true', help="instanciates a recurrent q-network architecture that keeps account of a history of frames, everything else is the same.")
    parser.add_argument('--recurrent_history_len', type=int, default=10, help="number of lookup steps when using a recurrent q network.")
    
    # training options (specific)
    parser.add_argument('--stop_decay', type=float, default=0.9, help="after what fraction of episodes we want to have eps = --eps_end or beta = --beta_end.")
    parser.add_argument('--eps_start', type=float, default=1.0, help="epsilon factor for egreedy policy, starting value.")
    parser.add_argument('--eps_end', type=float, default=0.005, help="epsilon factor for egreedy policy, starting value.")
    parser.add_argument('--alpha', type=float, default=0.6, help="alpha factor for prioritization contribution.")
    parser.add_argument('--beta_start', type=float, default=0.4, help="starting beta factor for bias correction when using a priotizied buffer.")
    parser.add_argument('--beta_end', type=float, default=1., help="ending beta factor for bias correction when using a priotizied buffer.")
    parser.add_argument('--update_every', type=int, default=100, help="how often to update the network, in steps.")
    parser.add_argument('--exploring_steps', type=int, default=25000, help="number of purely exploring steps at the beginning.")
    parser.add_argument('--target_update', type=str, default="soft", help="hard or soft update for target network. If hard specify --delay_steps. If soft specify --tau.")
    parser.add_argument('--tau', type=float, default=1e-2, help="weight for soft update of target parameters.")
    parser.add_argument('--delay_steps', type=int, default=10000, help="delay with which a hard update of the target network is conducted.")

    if phase == "train":
        parser.add_argument('--train', action='store_true', default=True, help="training flag set to true.")
        parser.add_argument('--save_freq', type=int, default=100, help="save Qnetworks every n episodes. Also tests the agent greedily for logs.")
        parser.add_argument('--log_freq', type=int, default=10, help="frequency (in episodes) with wich we store logs to weights and biases.") 
    elif phase=="test":
        parser.add_argument('--train', action='store_true', default=False, help="training flag set to False.")
        parser.add_argument('--option_file', type=str, default="train_options.txt", help=".txt file from which we load options.")
        parser.add_argument('--n_runs', type=int, default=10, help="number of test runs to do")
        parser.add_argument('--n_steps', type=int, default=250, help="number of steps to test the agent for.")
        parser.add_argument('--fname', type=str, default="sample", help="name of the file to save (gif).")
        parser.add_argument('--render', action='store_true', help="if rendering test trajectories (WARNING: takes a while).")
        parser.add_argument('--no_load', action='store_true', default=False, help="don't load any options when testing.")
    else:
        raise ValueError('unknown parameter phase: {}. expected: ("train"/"test").'.format(phase))
        
    return parser

def print_options(opt, parser, save = True):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    if save:
        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        if opt.train:
            file_name = os.path.join(expr_dir, 'train_options.txt')
        else:
            file_name = os.path.join(expr_dir, 'test_options.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

def load_options(opt, load_filename=None):
    """ loads and overrides options in opt with an existing option .txt output"""
    if opt.load_name is None:
        load_name = opt.name
    else:
        load_name = opt.load_name
    if load_filename is None:
        load_filename = 'train_options.txt'
    if "/" in load_filename: # a path was given
        load_path = load_filename
    else: # a filename was given
        load_path = os.path.join(opt.checkpoints_dir, load_name, load_filename)
        
    print("loading options from: {} ...".format(load_path))
    with open(load_path, 'r') as opt_file:
        lines = opt_file.readlines()

    # read options from the txt file 
    opt = vars(opt)
    load_opt = {}    
    for line in lines:
        if ':' in line:
            key = line.split(':')[0].strip()
            if key in opt:
                # specify which keys we do NOT want to overwrite
                if key not in ["load", "load_name", "wandb", "dataroot", "load_size", "volume_ids", "n_runs", "n_steps", "fname", "render", "option_file", "easy_objective", "train", "name", "randomize_intensities", "no_preprocess"]:
                    # get the str version of the value
                    value = line.split(':')[1].strip()
                    if "default" in value:
                        value = value.split("[default")[0].strip()
                    # if instance of bool handle explicitely to cast str to bool
                    if value.lower() in ["true", "false"]:
                        value = value == "True"
                    # otherwise cast implicitely with type()()
                    else:
                        #print(opt[key], value)
                        value = type(opt[key])(value)
                    load_opt[key] = value
    
    # overwrite overlapping opt entries with entries from load_opt
    for key in load_opt:
        opt[key] = load_opt[key]
    
    # reconvert opt to be a Namespace
    opt = Namespace(**opt)

    return opt
    
