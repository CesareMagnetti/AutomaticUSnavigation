import numpy as np
from matplotlib import pyplot as plt
import os
import warnings
import json

class Log():
    def __init__(self, name):
        """
        handles logs of an input variable.

        Params
        ======
        name (str): name of the variable we are logging
        Attributes
        ======
            logs (dict[list]): each element in the dict will be one episode. each element in the list will be one time step.

        """
        self.name=name
        self.logs = {1: []}
        self.N = 1
    
    def push(self, item, episode=None):
        if episode is None:
            self.logs[self.N].append(item)
        else:
            self.logs[episode].append(item)

    def pop(self, episode=None):
        # REMOVES ITEM FROM LOGS
        if episode is None:
            return self.logs.pop(self.N)
        else:
            return self.logs.pop(episode)

    def get(self, episode=None):
        # DOES NOT REMOVE ITEM FROM LOGS
        if episode is None:
            return self.logs[self.N]
        else:
            return self.logs[episode]
  
    def mean(self, episodes=None, axis=0):
        if episodes:
            return np.mean([log for i, (_,log) in enumerate(self.logs.items()) if i in episodes], axis=axis)
        else:
            return np.mean([log for _,log in self.logs.items()], axis=0)
    
    def std(self, episodes=None, axis=0):
        if episodes:
            return np.mean([log for i, (_,log) in enumerate(self.logs.items()) if i in episodes], axis=axis)
        else:
            return np.mean([log for _,log in self.logs.items()], axis=axis)

    def cumulative_sum(self, episode=None, GAMMA=1):
        if not episode:
            episode = self.N
        logs = self.logs[episode].copy()
        return sum([l*GAMMA**i for i,l in enumerate(reversed(logs))])

    def step(self):
        self.N+=1

    def __len__(self):
        return self.N

class Logger():
    def __init__(self, savedir, *args):
        self.savedir = savedir
        self.logs = {log.name: log for log in args if isinstance(log, Log)}

    def step(self):
        for log in self.logs:
            self.logs[log].step()
            
    def push(self, log):
        self.logs[log.name] = log   

    def pop(self, item=None):
        # REMOVES ITEM FROM self.logs AND RETURNS IT
        return self.logs.pop(item)  

    def get(self, item):
        # DOES NOT REMOVE ITEM FROM self.logs
        return self.log.get(item)  

    def save_logs_to_txt(self, fname):
        if not ".txt" in fname:
            fname+=".txt"
        if not os.path.exists(os.path.join(self.savedir, "logs")):
            os.makedirs(os.path.join(self.savedir, "logs"))
        with open(os.path.join(self.savedir, "logs", fname), 'w') as file:
            file.write(json.dumps(self.logs))
    
    def load_logs_from_txt(self, fname):
        with open(os.path.join(self.savedir, "logs", fname)) as file:
            self.logs = json.loads(file) 

    def visuals(self, names=None, episode=None, show=False, save=False):

        if show or save:
            if names is None:
                names = [log for log in self.logs]

            # get size of grid for subplots
            N = len(names)
            nrows = int(np.sqrt(N))
            ncols = int(N/nrows)
            if not N%nrows == 0:
                nrows+=1
            
            fig, axs = plt.subplots(nrows, ncols)
            fig.suptitle("visulas at episode: {}".format(episode if episode else "LAST"))
            fname = "episode{}".format(episode if episode else "LAST")

            for name, ax in zip(names, axs.ravel()):
                try:
                    ax.plot(range(len(self.logs[name].get(episode))), self.logs[name].get(episode))
                    ax.set_xlabel("time-step")
                except:
                    ax.plot(range(len(self.logs[name].get(episode))), self.logs[name])
                    ax.set_xlabel("episodes")
                ax.set_title(name)
                

            plt.tight_layout()

            if show:
                plt.show()
            elif save:
                if not os.path.exists(os.path.join(self.savedir, "logs", "visuals")):
                    os.makedirs(os.path.join(self.savedir, "logs", "visuals"))
                plt.savefig(os.path.join(self.savedir, "logs", "visuals", fname))
        
        else:
            warnings.warn("called Visualize.visuals() but did not pass in neither ``save`` or ``show`` flags. Nothing will be executed.")