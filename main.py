from environment.baseEnvironment import BaseEnvironment
from agent.agent import Agent
from options.options import gather_options, print_options
from timer.timer import Timer
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import os
import torch
import wandb


# gather options
parser = gather_options()
config = parser.parse_args()
config.use_cuda = torch.cuda.is_available()
print_options(config, parser)

# decay factor for epsilon based on our eps_start, eps_end and stop_eps_decay factor.
# we also need to consider that for the first --esploring steps we are not decaying epsilon
# hence our decay factor should get to eps_end when we want even if we did't update eps during
# these exploring steps
EPS_DECAY_FACTOR = (config.eps_end/config.eps_start)**(1/int(config.stop_eps_decay*config.n_episodes - config.exploring_steps/config.n_steps_per_episode))

def train(config):
    # initialize epsilon
    eps = config.eps_start 
    # tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(agent.qnetwork_target, agent.loss, log="all", log_freq=config.log_freq)
    # loop through episodes
    for episode in tqdm(range(1, config.n_episodes+1)):
        # choose a random volume and a random starting plane
        env.reset()
        state = env.state
        # initiate episode logs
        logs = {key: 0 for key in env.logged_rewards}
        logs["TDerror"] = 0
        logs["epsilon"] = eps
        # start episode
        with Timer("episode", config.timer):
            for _ in range(config.n_steps_per_episode):
                # get action from current state
                actions = agent.act(state, eps)
                # observe next state and reward 
                next_state, reward = env.step(actions) 
                # update Q network using Q learning algo, return the TD error
                TDerror = agent.step(state, actions, sum(reward.values()), next_state)
                state = next_state
                # store logs
                for key, r in reward.items():
                    if key in logs:
                        logs[key]+=r
                logs["TDerror"]+=TDerror

        # send logs to weights and biases
        if episode % config.log_freq == 0:
            with Timer("wandb.log", config.timer):
                wandb.log(logs, step=agent.t_step, commit=True)
                
        # save agent locally
        if episode % config.save_every == 0 and agent.t_step>agent.exploring_steps:
            print("saving latest model weights...")
            agent.save()
            agent.save("episode%d.pth"%episode)
            # tests the agent greedily for logs
            test(config, "episode%d.gif"%episode)
            
        # update eps
        if agent.t_step>agent.exploring_steps:
            with Timer("decay eps", config.timer):
                eps = max(eps*EPS_DECAY_FACTOR, config.eps_end)
    
    # at the very end save model as onnx for visualization and easy sharing to other frameworks
    dummy_input = torch.cat([env.sample(env.state).unsqueeze(0).unsqueeze(0)/255 for _ in range(10)], dim=0)
    torch.onnx.export(agent.qnetwork_target, dummy_input, os.path.join(config.checkpoints_dir, "qnetwork.onnx"))
    wandb.save("qnetwork.onnx")


def test(config, fname=None):
    with Timer("test", config.timer):
        env.reset()
        state = env.state
        frames=[]
        total_reward=0
        for i in tqdm(range(config.n_steps_per_episode)):
            actions = agent.act(state)
            state, reward = env.step(actions)
            frames.append(env.render(state, titleText='reward:{:.5f}'.format(sum(reward.values()))))
            total_reward+=sum(reward.values())
    
        # save all frames as a GIF
        if not os.path.exists(os.path.join(config.results_dir, config.name)):
            os.makedirs(os.path.join(config.results_dir, config.name))

        clip = ImageSequenceClip(frames, fps=10)
        if fname is None:
            fname = "navigation_samp%d.gif"%env.VolumeID
        clip.write_gif(os.path.join(config.results_dir, config.name, fname), fps=10)

        # save also on wandb
        with Timer("wandab.log", config.timer):
            wandb.log({"total_reward_collected_test": total_reward}, step=agent.t_step, commit=True)
        with Timer("wandab.save", config.timer):
            wandb.save(os.path.join(config.results_dir, config.name, fname))




if __name__=="__main__":

    with Timer("init wandb", config.timer):
        if config.wandb in ["online", "offline"]:
            wandb.login()

    # tell wandb to get started
    with wandb.init(project="AutomaticUSnavigation", name=config.name, config=config, mode=config.wandb):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        # instanciate environment
        with Timer("init env", config.timer):
            env = BaseEnvironment(config)
        # instanciate agent
        with Timer("init agent", config.timer):
            agent = Agent(env, config)
        if config.load is not None:
            print("loading: {} ...".format(config.load))
            agent.load(config.load)
        # train agent
        if config.train:
            train(config)
        # test agent
        test(config)
    
    Timer(timer=config.timer).save(os.path.join(agent.savedir, "timer_logs.txt"))
