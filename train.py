from environment.xcatEnvironment import SingleVolumeEnvironment
from agent.agent import SingleVolumeAgent
from options.options import gather_options, print_options
import torch

# gather options
parser = gather_options()
config = parser.parse_args()
config.use_cuda = torch.cuda.is_available()
print_options(config, parser)

if __name__=="__main__":
        # instanciate environment(s)
        vol_ids = config.volume_ids.split(",")
        if len(vol_ids)>1:
                env = []
                for vol_id in range(len(vol_ids)):
                        env.append(SingleVolumeEnvironment(config, vol_id=vol_id))
        else:
                if config.n_processes>1:
                        env = [SingleVolumeEnvironment(config)]*config.n_processes
                else:
                        env = SingleVolumeEnvironment(config)

        # instanciate agent
        agent = SingleVolumeAgent(config)
        # train agent
        agent.train(env)

