from environment.baseEnvironment import BaseEnvironment
from agent.agent import Agent
from options.options import gather_options, print_options
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import os
import numpy as np
import torch
import wandb

# gather options
parser = gather_options()
config = parser.parse_args()
config.use_cuda = torch.cuda.is_available()
print_options(config, parser)

def train(parser):
    # initialize epsilon
    eps = parser.eps_start 
    # tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(agent.qnetwork_target, agent.loss, log="all", log_freq=parser.log_freq)
    # loop through episodes
    for episode in tqdm(range(1, parser.n_episodes+1)):
        # choose a random volume and a random starting plane
        env.reset()
        state = env.state
        rewards, TDerrors = 0, 0
        for _ in range(parser.n_steps_per_episode):
            # get action from current state
            actions = agent.act(state, eps)
            # observe next state and reward 
            next_state, reward = env.step(actions) 
            # update Q network using Q learning algo, return the TD error
            TDerror = agent.step(state, actions, reward, next_state)
            state = next_state
            # store logs
            rewards+=reward
            TDerrors+=TDerror

        # send logs to weights and biases
        if episode % parser.log_freq == 0:
            wandb.log({"mean_TD_error": TDerrors,
                       "total_reward_collected": rewards,
                       "epsilon": eps}, step=agent.t_step)
                
        # save agent locally
        if episode % config.save_every == 0 and agent.t_step>agent.exploring_steps:
            print("saving latest model weights...")
            agent.save()

        # save a gif of the agent exploiting its policy
        if episode % config.save_trajectory_every == 0:
            test(config, "episode%d.gif"%episode)

        # update eps
        if agent.t_step>agent.exploring_steps:
            eps = max(eps*parser.eps_decay, parser.eps_end)
    
    # at the very end save model as onnx for visualization and easy sharing to other frameworks
    dummy_input = torch.cat([env.sample(env.state).unsqueeze(0).unsqueeze(0)/255 for _ in range(10)], dim=0)
    torch.onnx.export(agent.qnetwork_target, dummy_input, "qnetwork.onnx")
    wandb.save("qnetwork.onnx")



def test(parser, fname=None):
    env.reset()
    state = env.state
    frames=[]
    total_reward=0
    for i in tqdm(range(parser.n_steps_per_episode)):
        actions = agent.act(state)
        state, reward = env.step(actions)
        frames.append(env.render(state, titleText='reward:{:.5f}'.format(reward)))
        total_reward+=reward
  
    # save all frames as a GIF
    if not os.path.exists(os.path.join(config.results_dir, config.name)):
        os.makedirs(os.path.join(config.results_dir, config.name))

    clip = ImageSequenceClip(frames, fps=10)
    if fname is None:
        fname = "navigation_samp%d.gif"%env.VolumeID
    clip.write_gif(os.path.join(config.results_dir, config.name, fname), fps=10)

    # save also on wandb
    wandb.log({"total_reward_collected_test": total_reward}, step=agent.t_step)
    wandb.save(os.path.join(config.results_dir, config.name, fname))




if __name__=="__main__":
    if config.wandb in ["online", "offline"]:
        wandb.login()


    # tell wandb to get started
    with wandb.init(project="AutomaticUSnavigation", name=config.name, config=config, mode=config.wandb):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        # instanciate environment
        env = BaseEnvironment(config)
        # instanciate agent
        agent = Agent(env, config)
        if config.load is not None:
            agent.load(config.load)
        # train agent
        if config.train:
            train(config)
        # test agent
        test(config)
