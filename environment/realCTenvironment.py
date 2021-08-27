from environment.baseEnvironment import BaseEnvironment
from rewards.rewards import *
import numpy as np
import SimpleITK as sitk
import os

class realCTtestEnvironment(BaseEnvironment):
    def __init__(self, config, vol_id=0):
        """
        Initialize an Environment object, the environment will focus on a single real CT volume.
    
        Params that we will use:
        ======
            config (argparse object): contains all options. see ./options/options.py for more details
            vol_id (int): specifies which volume we want to load. Specifically ``config.volume_ids`` may contain
                          all queried volumes by the main.py script. (i.e. parser.volume_ids = 'samp0,samp1,samp2,samp3,samp4,samp5,samp6,samp7')
                          we will load the volume at position ``vol_id`` (default=0)               
        """

        # initialize the base environment
        BaseEnvironment.__init__(self, config)
        # identify the volume used by this environment instance
        self.vol_id = vol_id

        # load queried CT volume
        itkVolume = sitk.ReadImage(os.path.join(self.dataroot, self.vol_id+".nii.gz"))
        Volume = sitk.GetArrayFromImage(itkVolume) 
        self.Volume = Volume.astype(np.uint8) 
        self.sx, self.sy, self.sz = self.Volume.shape 

        # monitor oscillations for termination
        if config.termination == "oscillate":
            self.oscillates = Oscillate(history_length=config.termination_history_len, stop_freq=config.termination_oscillation_freq) 

        # get starting configuration and reset environment for a new episode
        self.reset()

    def sample_plane(self, state, return_seg=False, oob_black=True, preprocess=False, **kwargs):
        """ function to sample a plane from 3 3D points (state)
        Params:
        ==========
            state (np.ndarray of shape (3,3)): v-stacked 3D points that will define a particular plane in the CT volume.
            return_seg (bool): flag if we wish to return the corresponding segmentation map. (default=False)
            oob_black (bool): flack if we wish to mask out of volume pixels to black. (default=True)
            preprocess (bool): if to preprocess the plane before returning it (unsqueeze to BxCxHxW, normalize) 

            returns -> 
                plane (torch.tensor of shape (1, 1, self.sy, self.sx)): corresponding plane sampled from the CT volume
        """
        out = {}
        # 1. extract plane specs
        XYZ, P, S = self.get_plane_from_points(state, (self.sx, self.sy, self.sz))
        # 2. sample plane from the current volume
        X,Y,Z = XYZ
        plane = self.Volume[X,Y,Z]
        # mask out of boundary pixels to black
        if oob_black == True:
            plane[P < 0] = 0
            plane[P > S] = 0
        # if needed randomize intensities
        if self.config.randomize_intensities:
            plane = self.random_intensity_scaling(plane)
        # normalize and unsqueeze array if needed.
        if preprocess:
            plane = plane[np.newaxis, np.newaxis, ...]/255
        # add to output
        out["plane"] = plane
        return out
    
    def step(self, action, preprocess=False):
        """Perform an input action (discrete action), observe the next state and reward.
        Params:
        ==========
            action (int): discrete action (see baseEnvironment.mapActionToIncrement())
            preprocess (bool): if to preprocess the plane before returning it (unsqueeze to BxCxHxW, normalize)
        """
        # get the increment corresponding to this action
        increment = np.vstack([self.mapActionToIncrement(act) for act in action])
        # step into the next state
        state = self.state
        next_state = state + increment
        # observe the next plane and get the reward from segmentation map
        sample = self.sample_plane(state=next_state, return_seg=True, preprocess=preprocess)
        # update the current state
        self.state = next_state
        # episode termination, when done is True, the agent will consider only immediate rewards (no bootstrapping)
        if self.config.termination == "oscillate":
            # check if oscillating for episode termination
            done = self.oscillates(tuple(self.get_plane_coefs(*next_state)))
        elif self.config.termination == "learned":
            # check if all agents chose to stop (increment will be all zero)
            done = not increment.any()
        else:
            raise ValueError('unknown termination method: {}'.format(self.config.termination))
        
        # return transition and the sample
        return (state, action, next_state, done), sample
    
    def reset(self):
        if self.config.easy_objective:
            # shuffle rows of the goal state
            np.random.shuffle(self.goal_state)
            # add a random increment of +/- 10 pixels to this goal state
            noise = np.random.randint(low=-20, high=20, size=(3,3))
            self.state = self.goal_state + noise
        else:
            # sample a random plane (defined by 3 points) to start the episode from
            pointA = np.array([np.random.uniform(low=0., high=1)*self.sx-1,
                              np.random.uniform(low=0., high=1)*self.sy-1,
                              np.random.uniform(low=0., high=1.)*self.sz-1])

            pointB = np.array([np.random.uniform(low=0., high=1.)*self.sx-1,
                              np.random.uniform(low=0., high=1)*self.sy-1,
                              np.random.uniform(low=0., high=1.)*self.sz-1])

            pointC = np.array([np.random.uniform(low=0., high=1.)*self.sx-1,
                              np.random.uniform(low=0., high=1)*self.sy-1,
                              np.random.uniform(low=0., high=1.)*self.sz-1])             
            # stack points to define the state
            self.state = np.vstack([pointA, pointB, pointC]).astype(np.int)

        # reset the oscillation monitoring
        if self.config.termination == "oscillate":
            self.oscillates.history.clear()