from agent.agent import Agent
from environment.xcatEnvironment import SingleVolumeEnvironment
from networks.Qnetworks import SimpleQNetwork
from options.options import gather_options, print_options
import torch, os
import numpy as np
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

def test(runs, steps, agent, env, model, fname="test"):
    """Test the greedy policy learned by an agent.
    Params:
    ==========
        runs (int): number of test runs to undergo
        steps (int): number of steps to test the agent for.
        env (environment/* instance): the environment the agent will interact with while testing.
        agent (agent/* instance): the agent class.
        model (PyTorch model): pytorch network that will be tested.
        fname (str): name of file to save (default = test)
    """
    slices = []
    # test runs and collect frames
    for run in tqdm(range(runs), desc="testing..."):
        # reset env to a random initial slice
        env.reset()
        slice = env.sample_plane(env.state)
        temp_slices = []
        for _ in range(1, steps+1):  
            # save frame
            temp_slices.append(slice[..., np.newaxis]*np.ones(3))
            # get action from current state
            actions = agent.act(slice, model)  
            # observe next state (automatically adds (state, action, reward, next_state) to env.buffer) 
            next_slice = env.step(actions)
            # set slice to next slice
            slice = next_slice
        slices.append(np.array(temp_slices)) #shape: stepsx256x256x3
    
    # concatenate frames and render GIF
    slices = np.concatenate(slices, axis=2) #shape: stepsx256x(256*runs)x3
    slices = [slice for slice in slices] # list of len = steps, each frame of shape 256x(256*runs)x3
    clip = ImageSequenceClip(slices, fps=10)
    clip.write_gif(os.path.join(agent.results_dir, fname+".gif"), fps=10)
    


if __name__ == "__main__":
    # 1. gather options
    parser = gather_options(phase="test")
    config = parser.parse_args()
    config.use_cuda = torch.cuda.is_available()
    config.device = torch.device("cuda" if config.use_cuda else "cpu")
    print_options(config, parser)

    # 2. instanciate environment(s)
    vol_ids = config.volume_ids.split(",")
    if len(vol_ids)>1:
        raise ValueError('only supporting single volume environments for now.')
        # envs = []
        # for vol_id in range(len(vol_ids)):
        #         envs.append(SingleVolumeEnvironment(config, vol_id=vol_id))
    else:
        env = SingleVolumeEnvironment(config)
    # 3. instanciate agent
    agent = Agent(config)
    # 4. instanciate Qnetwork
    qnetwork = SimpleQNetwork((1, config.load_size, config.load_size), config.action_size, config.n_agents, config.seed, config.n_blocks_Q,
                                config.downsampling_Q, config.n_features_Q, config.dropout_Q).to(config.device)
    if config.load is not None:
        print("loading: {} ...".format(config.load))
        qnetwork.load(os.path.join(config.checkpoints_dir, config.name, config.load+".pth"))
    # 5. launch test
    test(config.n_runs, config.n_steps, agent, env, qnetwork, fname="test")

