from environment.baseEnvironment import BaseEnvironment
from agent.agent import Agent
from logger.logger import Log, Logger
from options.options import gather_options, print_options
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import os
import numpy as np
import torch

# gather options
parser = gather_options()
args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()
print_options(args, parser)

def train(parser):
    # setup logs
    rewards = Log("rewards", parser.n_episodes, parser.n_steps_per_episode)
    TDerrors = Log("TDerrors", parser.n_episodes, parser.n_steps_per_episode)
    epsilon = Log("epsilon", parser.n_episodes)
    logger = Logger(os.path.join(args.checkpoints_dir, args.name), rewards, TDerrors, epsilon)
    # initialize epsilon
    eps = parser.eps_start 
    print_every = 20
    for episode in tqdm(range(1, parser.n_episodes+1)):
        # choose a random volume and a random starting plane
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
        epsilon.push(eps)

        if episode % print_every == 0:
            print("[{}/{}] ({:.0f}%) eps:{:.3f}\n\t\ttotal reward collected in the last episode: {:.2f}\n\t\t"\
                  "mean reward collected in previous episodes: {:.2f}".format(episode,
                                                                              parser.n_episodes,
                                                                              int(episode/parser.n_episodes*100),
                                                                              eps,
                                                                              np.sum(rewards.current(), axis=-1),
                                                                              np.sum(rewards.mean(episodes=slice(0,episode)), axis=-1)))
                
            logger.current_visuals(save=True, with_total_reward=True, with_total_TDerror=True, title="episode%d.png"%episode)
        logger.current_visuals(save=True, with_total_reward=True, with_total_TDerror=True)
        logger.save_current_logs_to_txt()
        # save agent
        if episode % args.save_every == 0 and agent.t_step>agent.exploring_steps:
            print("saving latest model weights...")
            agent.save()
        # update eps
        if agent.t_step>agent.exploring_steps:
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
        frames.append(env.render(state, titleText='reward:{:.5f}'.format(reward)))
  
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
    if args.load is not None:
        agent.load(args.load)
    # train agent
    if args.train:
        train(args)
    # test agent
    test(args)
