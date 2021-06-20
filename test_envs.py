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
    fig, frames = env.random_walk(config.exploring_steps, config.exploring_restarts, visual=True)
    im_ani = animation.ArtistAnimation(fig, frames, interval=50, repeat_delay=3000, blit=True)
    if not os.path.exists(os.path.join(env.results_dir, "random_walk.gif")):
        os.makedirs(os.path.join(env.results_dir, "random_walk.gif"))
    #im_ani.save(os.path.join(env.results_dir, "random_walk.mp4"))
    plt.show()