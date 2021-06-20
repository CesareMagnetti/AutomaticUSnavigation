from environment.xcatEnvironment import SingleVolumeEnvironment
from options.options import gather_options, print_options
import torch, os
import matplotlib.animation as animation
import matplotlib.pyplot as plt

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
    print(env.buffer.sample())
