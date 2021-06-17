import argparse
import torch
import os

# parsing arguments
def gather_options():
    parser = argparse.ArgumentParser(description='train/test scripts to launch navigation experiments.')
    # I/O directories and data
    parser.add_argument('--dataroot', '-r',  type=str, help='path to the XCAT CT volumes.')
    parser.add_argument('--name', '-n', type=str, default='sample_experiment', help='name of the experiment.')
    parser.add_argument('--volume_ids', '-vol_ids', type=str, default='samp0,samp1,samp2,samp3,samp4,samp5,samp6,samp7', help='filename(s) of the CT volume(s) comma separated.')
    parser.add_argument('--ct2us_model_name', '-model', type=str, default='CycleGAN_LPIPS_noIdtLoss_lambda_AB_1',
                        help='filename for the state dict of the ct2us model (.pth) file.\navailable models can be found at ./models')
    parser.add_argument('--results_dir', type=str, default='./results/', help='where to save the trajectory.')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='where to save the trajectory.')

    # logs and checkpointing 
    parser.add_argument('--wandb', type=str, default='online', help='handles weights and biases interface.\n'\
                                                                'online: launches online interface.\n'\
                                                                'offline: writes all data to disk for later syncing to a server\n'\
                                                                'disabled: completely shuts off wandb. (default = online)')
    parser.add_argument('--save_every', type=int, default=100, help="save Qnetworks every n episodes. Also tests the agent greedily for logs.")
    parser.add_argument('--log_freq', type=int, default=10, help="frequency (in episodes) with wich we store logs to weights and biases.") 

    # Qnetwork specs  
    parser.add_argument('--n_blocks_Q', type=int, default=6, help="number of convolutional blocks in the Qnetwork.")
    parser.add_argument('--downsampling_Q', type=int, default=2, help="downsampling factor of each convolutional layer of the Qnetwork.")
    parser.add_argument('--n_features_Q', type=int, default=4, help="number of features in the first convolutional layer of the Qnetwork.")
    parser.add_argument('--no_dropout_Q', action='store_true', help="to not use dropout in the Qnetwork (p=0.5 after conv layers and p=0.1 after linear layers).")

    # agent specs
    parser.add_argument('--action_size', type=int, default=6, help="how many action can a single agent perform.\n(i.e. up/down,left/right,forward/backwards = 6 in a 3D volume).")
    parser.add_argument('--n_agents', type=int, default=3, help="how many RL agents (heads) will share the same CNN backbone.")

    # reward signal shaping
    parser.add_argument('--reward_id', type=int, default=2885, help="ID of the anatomical structure of interest. (default: left ventricle, 2885)")
    parser.add_argument('--penalty_per_step', type=float, default=0.1, help="give a small penalty for each step to incentivize moving towards planes of interest.")
    parser.add_argument('--area_penalty_weight', type=float, default=0.1, help='how much to incentivize the agents to maximize the area of the triangle they span.\n'\
                                                                    'This is to prevent them from moving towards the edges of a volume, which are meaningless.')

    # training options (general)
    parser.add_argument('--train', action='store_true', help='if training the agent before testing it.')
    parser.add_argument('--load', type=str, default=None, help='which model to load from.')
    parser.add_argument('--n_episodes', type=int, default=5000, help="number of episodes to train the agents for.")
    parser.add_argument('--n_steps_per_episode', type=int, default=250, help="number of steps in each episode.")
    parser.add_argument('--eps_start', type=float, default=1.0, help="epsilon factor for egreedy policy, starting value.")
    parser.add_argument('--eps_end', type=float, default=0.01, help="epsilon factor for egreedy policy, starting value.")
    parser.add_argument('--stop_eps_decay', type=float, default=0.9, help="after what fraction of episodes we want to have eps = --eps_end.")

    # training options (specific)
    parser.add_argument('--batch_size', type=int, default=64, help="batch size for the replay buffer.")
    parser.add_argument('--buffer_size', type=int, default=500000, help="capacity of the replay buffer.")
    parser.add_argument('--gamma', type=int, default=0.99, help="discount factor.")
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0002, help="learning rate for the q network.")
    parser.add_argument('--update_every', type=int, default=1, help="how often to update the network, in steps.")
    parser.add_argument('--exploring_steps', type=int, default=50000, help="number of purely exploring steps at the beginning.")
    parser.add_argument('--target_update', type=str, default="hard", help="hard or soft update for target network. If hard specify --delay_steps. If soft specify --tau.")
    parser.add_argument('--tau', type=int, default=1e-3, help="weight for soft update of target parameters.")
    parser.add_argument('--delay_steps', type=int, default=5000, help="delay with which a hard update of the target network is conducted.")

    # preprocessing
    parser.add_argument('--no_scale_intensity', action='store_true', help="If you do not want to scale the intensities of the CT volume.")

    # random seed for reproducibility
    parser.add_argument('--seed', type=int, default=1, help="random seed for reproducibility.")

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

    file_name = os.path.join(expr_dir, '{}_options.txt'.format("train" if opt.train else "test"))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')