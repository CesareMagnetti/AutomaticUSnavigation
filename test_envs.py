from environment.xcatEnvironment import SingleVolumeEnvironment
from options.options import gather_options, print_options
import torch
import numpy as np

# gather options
parser = gather_options()
config = parser.parse_args()
config.use_cuda = torch.cuda.is_available()
print_options(config, parser)


# LAUNCH SOME TESTS ON THE ENVIRONMENT
if __name__ == "__main__":
    env = SingleVolumeEnvironment(config, vol_id=0)
    print("current volume used: {}".format(env.vol_id))
    trajectory = env.random_walk(config.exploring_steps, config.exploring_restarts, return_trajectory=True)
    #planes = env.sample_planes(trajectory, return_seg=True)
    #env.render_light(trajectory, fname="random_walk")
    batch = env.buffer.sample()
    states, actions, rewards, next_states = batch
    planes = env.sample_planes(states+next_states, process=True)

    # concatenate states, normalize and move to gpu
    states = torch.from_numpy(np.vstack(planes[:config.batch_size])).float().to(env.device)
    next_states = torch.from_numpy(np.vstack(planes[config.batch_size:])).float().to(env.device)
    print("states: ", states.shape, "next_states: ", next_states.shape)
    # convert rewards to tensor and move to gpu
    rewards = torch.tensor(rewards).repeat(3).float().unsqueeze(0).to(env.device)
    print("rewards: ", rewards.shape)
    # convert actions to tensor, actions are currently stored as a list of length batch_size, where each entry is
    # an np.vstack([action1, action2, action3]). We need to convert this to a list of:
    # [action1]*batch_size + [action2]*batch_size + [action3]*batch_size so thats it's coherent with Q and MaxQ
    print("actions from buffer: ", actions)
    actions = torch.from_numpy(np.hstack(actions)).flatten().long().to(env.device)
    print("actions rearranged: ", actions, actions.shape)
    actions = actions.view(1, -1)
    print("actions reshaped: ", actions, actions.shape)
