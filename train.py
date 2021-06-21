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
        # instanciate environment
        env = SingleVolumeEnvironment(config)
        # instanciate agent
        agent = SingleVolumeAgent(config)
        # train agent
        agent.train(env)
        # test agent when done training
        agent.test(config.n_steps_per_episode, env, "test_trajectory")
