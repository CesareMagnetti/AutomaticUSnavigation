from networks.Qnetworks import setup_networks
from options.options import gather_options, print_options
import torch.multiprocessing as mp
from utils import train

if __name__=="__main__":
        # 1. gather options
        parser = gather_options(phase="train")
        config = parser.parse_args()
        config.use_cuda = torch.cuda.is_available()
        config.device = torch.device("cuda" if config.use_cuda else "cpu")
        print_options(config, parser)
        # 2. instanciate Qnetworks
        qnetwork_local, qnetwork_target = setup_networks(config)
        qnetwork_local.share_memory()  # gradients are allocated lazily, so they are not shared here, necessary to train on multiple processes
        # 3. launch training
        # MULTI-PROCESS TRAINING
        if config.n_processes>1:
                raise NotImplementedError("MULTI-PROCESSING DOES NOT CURRENTLY WORK.")
                mp.set_start_method('spawn')
                processes = []
                for rank in range(config.n_processes):
                        p = mp.Process(target=train, args=(config, qnetwork_local, qnetwork_target, rank))
                        p.start()
                        processes.append(p)
                for p in processes:
                        p.join()
        # SINGLE PROCESS TRAINING
        else:
                train(config, qnetwork_local, qnetwork_target, sweep=False)

