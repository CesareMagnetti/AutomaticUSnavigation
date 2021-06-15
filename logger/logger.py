import numpy as np
from matplotlib import pyplot as plt
import os
import warnings

class Log():
    def __init__(self, name, max_i, max_j=None):
        """
        handles logs of an input variable.

        Params
        ======
        name (str): name of the variable we are logging
        max_i (int): maximum number of rows in the log array (i.e. number of episodes)
        max_j (int): maxim number of columns in the log array (i.e. number of steps per episode) (default=None)
        Attributes
        ======
            logs (ndarray): each row corresponds to one episode. each column to one time step.

        """
        self.name, self.max_i, self.max_j = name, max_i, max_j
        shape = (max_i, max_j) if max_j else (max_i,1)
        self.logs = np.zeros(shape, dtype=float)
        self.i, self.j = 0, 0
    
    def push(self, item):
        self.logs[self.i, self.j] = item
        self.j+=1

    def __getitem__(self, i, j=None):
        if j:
            try: return self.logs[i, j].item()
            except: return self.logs[i, j]
        else:
            try: return self.logs[i].item()
            except: return self.logs[i]

    def current(self):
        try: return self.logs[self.i].item()
        except: return self.logs[self.i]

    def mean(self, episodes=None, axis=0):
        if not episodes: logs = self.logs.copy()
        else: logs = self.logs[episodes, ...].copy()
        try: return np.mean(logs, axis=axis).item()
        except: return np.mean(logs, axis=axis)
    
    def std(self, episodes=None, axis=0):
        if not episodes: logs = self.logs.copy()
        else: logs = self.logs[episodes, ...].copy()
        try: return np.std(logs, axis=axis).item()
        except: return np.std(logs, axis=axis)

    def cumulative_sum(self, episodes=None, GAMMA=1, REVERSED=True):
        logs = self.logs[episodes, ...].copy() if episodes else self.logs.copy()
        # build discount array
        discount = [GAMMA**i for i in range(self.max_j)]
        if REVERSED: discount = discount[::-1]
        discount = np.array(discount)
        # discount the logs along last dimension and sum
        ret = np.cumsum(logs*discount, axis=-1)
        # this will return a scalar if single element array
        try: return ret.item() 
        except: return ret

    def step(self):
        # reset indeces for next episode
        self.i, self.j = self.i+1, 0

    def __len__(self):
        return self.i

class Logger():
    def __init__(self, savedir, *args):
        self.savedir = savedir
        self.logs = {log.name: log for log in args if isinstance(log, Log)}

    def step(self):
        for log in self.logs:
            self.logs[log].step()
            
    def push(self, **kwargs):
        for key, val in kwargs.items():
            self.logs[key].push(val)    

    def save_current_logs_to_txt(self, fname=None):
        if fname is None:
            fname = "last_logs.txt"

        if not os.path.exists(os.path.join(self.savedir, "logs")): 
            os.makedirs(os.path.join(self.savedir, "logs"))
        # gather logs up to current episode
        logs = {}
        for key, log in self.logs.items():
            logs[key] = log.logs[:log.i, ...]

        np.savez(os.path.join(self.savedir, "logs", fname), **logs)

    def current_visuals(self, with_total_reward=True, show=False, save=False, title=None):

        if not (show or save):
            warnings.warn("called Visualize.visuals() but did not pass in neither ``save`` or ``show`` flags. Nothing will be executed.")
        else:

            # get current episode
            episode = next(iter(self.logs.values())).i
            # get a copy of current results
            logs = self.logs.copy()
            # calculate total collected rewards in previous episodes if needed
            if "rewards" in logs and with_total_reward:
                logs["rewards_total"] = np.sum(logs["rewards"][:episode], axis=-1)

            # plot visuals
            fig, axs = plt.subplots(len(logs), 1)
            fig.suptitle("visuals at episode:%d"%(episode+1))
            for name, ax in zip(logs, axs.ravel()):
                if not "total" in name:
                    ax.plot(logs[name].current())
                    ax.set_xlabel("time-step")
                else:
                    ax.plot(logs[name])
                    ax.set_xlabel("episodes")                    
                ax.set_title(name)
            plt.tight_layout()

            if show:
                plt.show()
            elif save:
                if not os.path.exists(os.path.join(self.savedir, "logs", "visuals")):
                    os.makedirs(os.path.join(self.savedir, "logs", "visuals"))
                if title is None:
                    title="last_visuals.png"
                plt.savefig(os.path.join(self.savedir, "logs", "visuals", title))