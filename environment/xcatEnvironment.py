from environment.baseEnvironment import BaseEnvironment
import numpy as np
import SimpleITK as sitk
import os

def get_traingle_area(a, b, c) :
    return 0.5 * np.linalg.norm( np.cross( b-a, c-a ) )

class SingleVolumeEnvironment(BaseEnvironment):
    def __init__(self, config, vol_id=0):
        """
        Initialize an Environment object, the environment will focuse on a single XCAT/CT volume.
    
        Params that we will use:
        ======
            config (argparse object): contains all options. see ./options/options.py for more details
            vol_id (int): specifies which volume we want to load. Specifically ``config.volume_ids`` should contain
                          all queried volumes by the main.py script. (i.e. parser.volume_ids = 'samp0,samp1,samp2,samp3,samp4,samp5,samp6,samp7')
                          we will load the volume at position ``vol_id`` (default=0)               
        """

        # initialize the base environment
        BaseEnvironment.__init__(self, config)

        # identify the volume used by this environment instance
        volume_ids = config.volume_ids.split(',')
        self.vol_id = volume_ids[vol_id]
        
        # load queried CT volume
        itkVolume = sitk.ReadImage(os.path.join(self.dataroot, self.vol_id+"_CT_1.nii.gz"))
        Volume = sitk.GetArrayFromImage(itkVolume) 
        # preprocess volume
        if not config.no_preprocess:
            Volume = Volume/Volume.max()*255
            Volume = intensity_scaling(Volume, pmin=config.pmin, pmax=config.pmax, nmin=config.nmin, nmax=config.nmax)
        self.Volume = Volume.astype(np.uint8)  

        # load queried CT segmentation
        itkSegmentation = sitk.ReadImage(os.path.join(self.dataroot, self.vol_id+"_SEG_1.nii.gz"))
        Segmentation = sitk.GetArrayFromImage(itkSegmentation)
        self.Segmentation = Segmentation

        # setup the reward function arguments:
        # 1. we reward based on a particular anatomical structure being present in the slice sampled by the current state
        #    to this end we have segmentations of the volume and reward_id corresponds to the value of the anatomical tissue
        #    of interest in the segmentation (i.e. the ID of the left ventricle is 2885)
        self.rewardID = config.reward_id
        # 2. we give a small penalty for each step in which the above ID is not present in the sampled slice. As soon as even
        #    one pixel of the structure of interest enters the sampled slice, we stop the penalty. Like this the agent is incetivized
        #    to move towards the region of interest quickly.
        self.penalty_per_step = config.penalty_per_step
        # 3. In order to avoid the agents to cluster together at an edge, we give a reward to maximize the area of the 2D triangle spanned
        #    by the 3 agents. This should incentivize the agents to sample more meaningful planes as if they clustered at an edge, the resulting
        #    image would be meaningless and mostly black. Arguably this should also embed the agent with some prior information about the fact 
        #    that they are sampling a plane and should work together (with a shared objective) rather than indepentently.
        self.area_penalty_weight = config.area_penalty_weight
        # based on the above values we select which rewards we are logging
        self.logged_rewards = ["rewardAnatomy"]
        if self.penalty_per_step>0:
            self.logged_rewards+=["rewardStep"]
        if self.area_penalty_weight>0:
            self.logged_rewards+=["rewardArea"]
        
        # get starting configuration
        self.sx, self.sy, self.sz = self.Volume.shape
        self.reset()

    def sample_plane(self, state, return_seg=False, oob_black=True):
        """ function to sample a plane from 3 3D points (state)
        Params:
        ==========
            state (np.ndarray of shape (3,3)): v-stacked 3D points that will define a particular plane in the CT volume.
            return_seg (bool): flag if we wish to return the corresponding segmentation map. (default=False)
            oob_black (bool): flack if we wish to mask out of volume pixels to black. (default=True)

            returns -> plane (torch.tensor of shape (1, 1, self.sy, self.sx)): corresponding plane sampled from the CT volume (normalized and unsqueezed to 4D)
                       seg (optional, np.ndarray of shape (self.sy, self.sx)): segmentation map of the sampled plane
        """
        # get plane coefs
        a,b,c,d = self.get_plane_coefs(*state)
        # extract corresponding slice
        main_ax = np.argmax([abs(a), abs(b), abs(c)])
        if main_ax == 0:
            Y, Z = np.meshgrid(np.arange(self.sy), np.arange(self.sz), indexing='ij')
            X = (d - b * Y - c * Z) / a

            X = X.round().astype(np.int)
            P = X.copy()
            S = self.sx-1

            X[X <= 0] = 0
            X[X >= self.sx] = self.sx-1

        elif main_ax==1:
            X, Z = np.meshgrid(np.arange(self.sx), np.arange(self.sz), indexing='ij')
            Y = (d - a * X - c * Z) / b

            Y = Y.round().astype(np.int)
            P = Y.copy()
            S = self.sy-1

            Y[Y <= 0] = 0
            Y[Y >= self.sy] = self.sy-1
        
        elif main_ax==2:
            X, Y = np.meshgrid(np.arange(self.sx), np.arange(self.sy), indexing='ij')
            Z = (d - a * X - b * Y) / c

            Z = Z.round().astype(np.int)
            P = Z.copy()
            S = self.sz-1

            Z[Z <= 0] = 0
            Z[Z >= self.sz] = self.sz-1
        
        # sample plane from the current volume
        plane = self.Volume[X, Y, Z]

        if oob_black == True:
            plane[P < 0] = 0
            plane[P > S] = 0
        
        if return_seg:
            return plane, self.Segmentation[X, Y, Z]
        else:
            return plane

    def get_reward(self, seg):
        """Calculates the corresponding reward of stepping into a state given its segmentation map.
        Params:
        ==========
            seg (np.ndarray of shape (self.sy, self.sx)): segmentation map from which to extract the reward
        """
        rewards = {}
        # sample the according reward (i.e. count of pixels of a particular anatomical structure)
        # by default the left ventricle ID in the segmentation is 2885. We will count the number
        # of pixels in the the queried anatomical structure as a fit function.The agent will have to
        # find a view that maximizes the amount of context pixels present in the image.
        rewardAnatomy = (seg==self.rewardID).sum().item()
        rewardAnatomy/=np.prod(seg.shape) # normalize by all pixels count to set in (0, 1) range
        rewards["rewardAnatomy"] = rewardAnatomy

        # give a penalty for each step that does not contain any anatomical structure of interest.
        # this should incentivize moving towards planes of interest quickly to not receive negative rewards.
        if "rewardStep" in self.logged_rewards:
            if rewardAnatomy > 0:
                rewardStep = 0
            else:
                rewardStep = -self.penalty_per_step 
            rewards["rewardStep"] = rewardStep

        # incentivize the agent to stay in a relevant plane (not at the edges of the volume) by
        # maximizing the area of the triangle spanned by the three points
        if "rewardArea" in self.logged_rewards:
            area = get_traingle_area(*self.state)
            # normalize this area by the area of a 2D slice (like above)
            area/=np.prod(seg.shape)
            rewards["rewardArea"] = self.area_penalty_weight*area 

        return rewards

    def step(self, action, buffer=None):
        """Perform an input action (discrete action), observe the next state and reward.
        Automatically stores the tuple (state, action, reward, next_state) to the replay buffer.
        Params:
        ==========
            action (int): discrete action (see baseEnvironment.mapActionToIncrement())
            buffer(buffer/* instance): ReplayBuffer class to store memory.
        """
        # get the increment corresponding to this action
        increment = np.vstack([self.mapActionToIncrement(act) for act in action])
        # step into the next state
        state = self.state
        next_state = state + increment
        # observe the next plane and get the reward from segmentation map
        next_slice, segmentation = self.sample_plane(state=next_state, return_seg=True)
        rewards = self.get_reward(segmentation)
        # log these rewards to the current episode count
        for r in rewards:
            self.logs[r]+=rewards[r]
        # add transition to replay buffer
        if buffer is not None:
            buffer.add(state, action, sum(rewards.values()), next_state)
        # update the current state
        self.state = next_state
        return next_slice
    
    def reset(self):
        # sample a random plane (defined by 3 points) to start the episode from
        if self.config.easy_objective:
            # these planes correspond more or less to a 4-chamber view
            pointA = np.array([np.random.uniform(low=0.85, high=1)*self.sx,
                              0,
                              np.random.uniform(low=0.7, high=0.92)*self.sz], type=np.int)

            pointB = np.array([np.random.uniform(low=0.3, high=0.43)*self.sx,
                              self.sy,
                              0], type=np.int)

            pointC = np.array([np.random.uniform(low=0.3, high=0.43)*self.sx,
                              self.sy,
                              self.sz], type=np.int) 
        else:
            pointA = np.array([np.random.uniform(low=0., high=1)*self.sx),
                              0,
                              np.random.uniform(low=0., high=1.)*self.sz)], type=np.int)

            pointB = np.array([np.random.uniform(low=0., high=1.)*self.sx),
                              self.sy,
                              np.random.uniform(low=0., high=1.)*self.sz], type=np.int)

            pointC = np.array([np.random.uniform(low=0., high=1.)*self.sx,
                              self.sy,
                              np.random.uniform(low=0., high=1.)*self.sz], type=np.int)             
        # stack points to define the state
        self.state = np.vstack([pointA, pointB, pointC])

        # reset the logged rewards for this episode
        self.logs = {r: 0 for r in self.logged_rewards}

# ==== HELPER FUNCTIONS ====
def intensity_scaling(ndarr, pmin=None, pmax=None, nmin=None, nmax=None):
    pmin = pmin if pmin != None else ndarr.min()
    pmax = pmax if pmax != None else ndarr.max()
    nmin = nmin if nmin != None else pmin
    nmax = nmax if nmax != None else pmax
    
    ndarr[ndarr<pmin] = pmin
    ndarr[ndarr>pmax] = pmax
    ndarr = (ndarr-pmin)/(pmax-pmin)
    ndarr = ndarr*(nmax-nmin)+nmin
    return ndarr 