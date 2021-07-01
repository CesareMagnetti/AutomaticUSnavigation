from environment.xcatEnvironment import SingleVolumeEnvironment
from utils import Visualizer
from options.options import gather_options, print_options
import torch, os
import numpy as np
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip


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
    
    visualizer = Visualizer()
    trajectory = env.random_walk(config.n_steps, return_trajectory=True)
    visualizer.render_states(trajectory)