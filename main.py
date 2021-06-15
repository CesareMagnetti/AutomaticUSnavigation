from environment.baseEnvironment import BaseEnvironment
from agent.agent import Agent
from logger.logger import Log, Logger
import argparse
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import os
import numpy as np
import torch

# parsing arguments
parser = argparse.ArgumentParser(description='test script to verify we can sample trajectories from the environment.')
# directories handling
parser.add_argument('--dataroot', '-r',  type=str, help='path to the XCAT CT volumes.')
parser.add_argument('--name', '-n', type=str, default='sample_experiment', help='name of the experiment.')
parser.add_argument('--volume_id', '-vol_id', type=str, default='samp0', help='filename of the CT volume.')
parser.add_argument('--ct2us_model_name', '-model', type=str, default='CycleGAN_LPIPS_noIdtLoss_lambda_AB_1',
                    help='filename for the state dict of the ct2us model (.pth) file.\navailable models can be found at ./models')
parser.add_argument('--results_dir', type=str, default='./results/', help='where to save the trajectory.')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='where to save the trajectory.')
# training options
parser.add_argument('--train', action='store_true', help='if training the agent before testing it.')
parser.add_argument('--batch_size', type=int, default=32, help="batch size for the replay buffer.")
parser.add_argument('--buffer_size', type=int, default=int(1e5), help="capacity of the replay buffer.")
parser.add_argument('--gamma', type=int, default=0.99, help="discount factor.")
parser.add_argument('--tau', type=int, default=1e-3, help="weight for soft update of target parameters.")
parser.add_argument('--learning_rate', '-lr', type=float, default=5e-4, help="learning rate for the q network.")
parser.add_argument('--update_every', type=int, default=4, help="how often to update the network, in steps.")
parser.add_argument('--exploring_steps', type=int, default=1000, help="number of purely exploring steps at the beginning.")
parser.add_argument('--action_size', type=int, default=6, help="how many action can a single agent perform.\n(i.e. up/down,left/right,forward/backwards = 6 in a 3D volume).")
parser.add_argument('--n_agents', type=int, default=3, help="how many RL agents (heads) will share the same CNN backbone.")
parser.add_argument('--n_episodes', type=int, default=1000, help="number of episodes to train the agents for.")
parser.add_argument('--n_steps_per_episode', type=int, default=250, help="number of steps in each episode.")
parser.add_argument('--eps_start', type=float, default=1.0, help="epsilon factor for egreedy policy, starting value.")
parser.add_argument('--eps_end', type=float, default=0.01, help="epsilon factor for egreedy policy, starting value.")
parser.add_argument('--eps_decay', type=float, default=0.995, help="epsilon factor for egreedy policy, decay factor.")
parser.add_argument('--reward_id', type=int, default=2885, help="ID of the anatomical structure of interest. (default: left ventricle, 2885)")
parser.add_argument('--no_scale_intensity', action='store_true', help="If you do not want to scale the intensities of the CT volume.")
parser.add_argument('--loss', default=torch.nn.SmoothL1Loss(), help="torch.nn instance of the loss function to use while training the Qnetwork.")
# random seed for reproducibility
parser.add_argument('--seed', type=int, default=1, help="random seed for reproducibility.")

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

def train(parser):
    """Deep Q-Learning training.
    Params
    ======
        parser.n_episodes (int): maximum number of training episodes
        parser.max_t (int): maximum number of timesteps per episode
        parser.eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        parser.eps_end (float): minimum value of epsilon
        parser.eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    rewards, TDerrors = Log("rewards", parser.n_episodes, parser.n_steps_per_episode), Log("TDerrors", parser.n_episodes, parser.n_steps_per_episode)
    logger = Logger(os.path.join(args.checkpoints_dir, args.name), rewards, TDerrors)
    # initialize epsilon
    eps = parser.eps_start 
    print_every = 10
    for episode in range(1, parser.n_episodes+1):
        env.reset()
        state = env.state
        for _ in range(parser.n_steps_per_episode):
            # get action from current state
            actions = agent.act(state, eps)
            # observe next state and reward 
            next_state, reward = env.step(actions) 
            # update Q network using Q learning algo, return the TD error
            TDerror = agent.step(state, actions, reward, next_state)
            state = next_state
            # add logs
            rewards.push(reward)
            if TDerror: # TD error can be None if agent.update_every>1
                TDerrors.push(TDerror)

        if episode % print_every == 0:
            print("[{}/{}] ({:.0f}%) eps:{:.3f}\n\t\ttotal reward collected in the last episode: {:.2f}\n\t\t"\
                  "mean reward collected in previous episodes: {:.2f}".format(episode,
                                                                              parser.n_episodes,
                                                                              int(episode/parser.n_episodes*100),
                                                                              eps,
                                                                              np.sum(rewards.current(), axis=-1),
                                                                              np.sum(rewards.mean(episodes=slice(0,episode)), axis=-1)))
            # save latest model weights
            print("saving latest model weights...")
            agent.save()
            logger.current_visuals(save=True, with_total_reward=True, title="episode%d.png"%episode)
            logger.save_current_logs_to_txt(fname="episode%d.txt"%episode)
        logger.current_visuals(save=True, with_total_reward=True)
        logger.save_current_logs_to_txt()
        
        # update eps
        eps = max(eps*parser.eps_decay, parser.eps_end)
        # step logs for next episode
        logger.step()


def test(parser):
    env.reset()
    state = env.state
    frames=[]
    for i in tqdm(range(parser.n_steps_per_episode)):
        actions = agent.act(state)
        state, reward = env.step(actions)
        frames.append(env.render(state, with_seg=True, titleText='time step: {} reward:{:.5f}'.format(i+1, reward)))
  
    # save all frames as a GIF
    if not os.path.exists(os.path.join(args.results_dir, args.name)):
        os.makedirs(os.path.join(args.results_dir, args.name))

    clip = ImageSequenceClip(frames, fps=10)
    clip.write_gif(os.path.join(args.results_dir, args.name, 'navigation.gif'), fps=10)



if __name__=="__main__":

    # instanciate environment
    env = BaseEnvironment(args)
    # instanciate agent
    agent = Agent(env, args)

    # train agent
    if args.train:
        train(args)
    
    # test agent
    test(args)
