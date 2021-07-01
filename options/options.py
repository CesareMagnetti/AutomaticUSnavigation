import argparse
import torch
import os

# parsing arguments
def gather_options(phase="train"):

    parser = argparse.ArgumentParser(description='train/test scripts to launch navigation experiments.')
    # I/O directories and data
    parser.add_argument('--dataroot', '-r',  type=str, default="/vol/biomedic3/hjr119/DATA/XCAT_VOLUMES/", help='path to the XCAT CT volumes.')
    parser.add_argument('--name', '-n', default="sample_experiment", type=str, help='name of the experiment.')
    parser.add_argument('--volume_ids', '-vol_ids', type=str, default='samp0', help='filename(s) of the CT volume(s) comma separated.')
    parser.add_argument('--ct2us_model_name', '-model', type=str, default='CycleGAN_LPIPS_noIdtLoss_lambda_AB_1',
                        help='filename for the state dict of the ct2us model (.pth) file.\navailable models can be found at ./models')
    parser.add_argument('--results_dir', type=str, default='./results/', help='where to save the trajectory.')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='where to save the trajectory.')
    parser.add_argument('--load', type=str, default=None, help='which model to load from.')

    # preprocessing
    parser.add_argument('--load_size', type=int, default=256, help="resolution to load the data. By default 256 isotropic resolution.")
    parser.add_argument('--no_preprocess', action='store_true', help="If you do not want to preprocess the CT volume. (set to uint8 and scale intensities)")
    parser.add_argument('--pmin', type=int, default=150, help="pmin value for xcatEnvironment/intensity_scaling() function.")
    parser.add_argument('--pmax', type=int, default=200, help="pmax value for xcatEnvironment/intensity_scaling() function.")
    parser.add_argument('--nmin', type=int, default=0, help="nmin value for xcatEnvironment/intensity_scaling() function.")
    parser.add_argument('--nmax', type=int, default=255, help="nmax value for xcatEnvironment/intensity_scaling() function.")

    # Qnetwork specs
    parser.add_argument('--default_Q', type=str, default=None, help="give a standard architecure: small -> 3 blocks with stride 4. large -> 6 blocks with stride 2.")
    parser.add_argument('--n_blocks_Q', type=int, default=6, help="number of convolutional blocks in the Qnetwork.")
    parser.add_argument('--downsampling_Q', type=int, default=2, help="downsampling factor of each convolutional layer of the Qnetwork.")
    parser.add_argument('--n_features_Q', type=int, default=4, help="number of features in the first convolutional layer of the Qnetwork.")
    parser.add_argument('--dropout_Q', action='store_true', help="if use dropout in the Qnetwork (p=0.5 after conv layers and p=0.1 after linear layers).")

    # agent specs
    parser.add_argument('--action_size', type=int, default=7, help="how many action can a single agent perform.\n(i.e. up/down,left/right,forward/backwards,do nothing = 7 in a 3D volume).")
    parser.add_argument('--n_agents', type=int, default=3, help="how many RL agents (heads) will share the same CNN backbone.")

    # reward signal shaping
    parser.add_argument('--anatomyRewardIDs', type=str, default="2885", help="ID of the anatomical structure of interest. (default: left ventricle, 2885). if multiple IDs separate by comma.")
    parser.add_argument('--steppingReward', type=float, default=0.1, help="give a small penalty for each step to incentivize moving towards planes of interest. (should be positive number)")
    parser.add_argument('--areaRewardWeight', type=float, default=0.01, help='how much to incentivize the agents to maximize the area of the triangle they span. (should be a positive number)\n'\
                                                                             'This is to prevent them from moving towards the edges of a volume, which are meaningless.')
    parser.add_argument('--oobReward', type=float, default=0.01, help='how much to penalis=ze each out of boundary step of an agent. (should be a positive number)')


    # random seed for reproducibility
    parser.add_argument('--seed', type=int, default=1, help="random seed for reproducibility.")
    # multi-processing
    parser.add_argument('--n_processes', type=int, default=1, help="number of processes to launch together.")
    # flag for easier objective
    parser.add_argument('--easy_objective', action='store_true', help="starts the agent in a plane that should be close to a 4-chamber view.")

    # training options (general)
    parser.add_argument('--n_episodes', type=int, default=2000, help="number of episodes to train the agents for.")
    parser.add_argument('--n_steps_per_episode', type=int, default=250, help="number of steps in each episode.")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size for the replay buffer.")
    parser.add_argument('--buffer_size', type=int, default=50000, help="capacity of the replay buffer.") 
    parser.add_argument('--gamma', type=int, default=0.999, help="discount factor.")
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0002, help="learning rate for the q network.")
    parser.add_argument('--loss', type=str, default="MSE", help='which loss to use to optimize the Qnetwork (MSE, SmoothL1).')
    parser.add_argument('--trainer', type=str, default="DoubleDeepQLearning", help='which training routine to use (DeepQLearning, DoubleDeepQLearning...).')

    # training options (specific)
    parser.add_argument('--stop_decay', type=float, default=0.9, help="after what fraction of episodes we want to have eps = --eps_end or beta = --beta_end.")
    parser.add_argument('--eps_start', type=float, default=1.0, help="epsilon factor for egreedy policy, starting value.")
    parser.add_argument('--eps_end', type=float, default=0.005, help="epsilon factor for egreedy policy, starting value.")
    parser.add_argument('--alpha', type=float, default=0.6, help="alpha factor for prioritization contribution.")
    parser.add_argument('--beta_start', type=float, default=0.4, help="starting beta factor for bias correction when using a priotizied buffer.")
    parser.add_argument('--beta_end', type=float, default=1., help="ending beta factor for bias correction when using a priotizied buffer.")
    parser.add_argument('--update_every', type=int, default=10, help="how often to update the network, in steps.")
    parser.add_argument('--exploring_steps', type=int, default=50000, help="number of purely exploring steps at the beginning.")
    parser.add_argument('--target_update', type=str, default="hard", help="hard or soft update for target network. If hard specify --delay_steps. If soft specify --tau.")
    parser.add_argument('--tau', type=float, default=1e-2, help="weight for soft update of target parameters.")
    parser.add_argument('--delay_steps', type=int, default=10000, help="delay with which a hard update of the target network is conducted.")

    if phase == "train":
        parser.add_argument('--train', action='store_true', default=True, help="training flag set to true.")
        # logs and checkpointing 
        parser.add_argument('--wandb', type=str, default='online', help='handles weights and biases interface.\n'\
                                                                    'online: launches online interface.\n'\
                                                                    'offline: writes all data to disk for later syncing to a server\n'\
                                                                    'disabled: completely shuts off wandb. (default = online)')
        parser.add_argument('--save_freq', type=int, default=100, help="save Qnetworks every n episodes. Also tests the agent greedily for logs.")
        parser.add_argument('--log_freq', type=int, default=10, help="frequency (in episodes) with wich we store logs to weights and biases.") 
    elif phase=="test":
        parser.add_argument('--train', action='store_true', default=False, help="training flag set to False.")
        parser.add_argument('--n_runs', type=int, default=5, help="number of test runs to do")
        parser.add_argument('--n_steps', type=int, default=250, help="number of steps to test the agent for.")
    else:
        raise ValueError('unknown parameter phase: {}. expected: ("train"/"test").'.format(phase))
        
    return parser

def print_options(opt, parser):
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