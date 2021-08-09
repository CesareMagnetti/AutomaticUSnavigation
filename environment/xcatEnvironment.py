from environment.baseEnvironment import BaseEnvironment
from rewards.rewards import *
import numpy as np
import SimpleITK as sitk
import torch, os
import scipy.ndimage as ndimage

class SingleVolumeEnvironment(BaseEnvironment):
    def __init__(self, config, vol_id=0):
        """
        Initialize an Environment object, the environment will focus on a single XCAT/CT volume.
    
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
        volume_ids = config.volume_ids.split(',')
        self.vol_id = volume_ids[vol_id]
        
        # load queried CT volume
        itkVolume = sitk.ReadImage(os.path.join(self.dataroot, self.vol_id+"_1_CT.nii.gz"))
        Volume = sitk.GetArrayFromImage(itkVolume) 
        # preprocess volume
        if not config.no_preprocess:
            Volume = Volume/Volume.max()*255
            Volume = intensity_scaling(Volume, pmin=config.pmin, pmax=config.pmax, nmin=config.nmin, nmax=config.nmax)
        self.Volume = Volume.astype(np.uint8) 
        self.sx, self.sy, self.sz = self.Volume.shape 

        # load queried CT segmentation
        itkSegmentation = sitk.ReadImage(os.path.join(self.dataroot, self.vol_id+"_1_SEG.nii.gz"))
        Segmentation = sitk.GetArrayFromImage(itkSegmentation)
        self.Segmentation = Segmentation

        # get an approximated location for the 4-chamber slice using the centroids
        LVcentroid = get_centroid(Segmentation, 2885)
        RVcentroid = get_centroid(Segmentation, 2897)
        LAcentroid = get_centroid(Segmentation, 2893)
        self.goal_state = np.vstack([LVcentroid, RVcentroid, LAcentroid])
        # get the corresponding plane coefficients
        self.goal_plane = self.get_plane_coefs(LVcentroid,RVcentroid,LAcentroid)

        # setup the reward function handling:
        self.logged_rewards = []
        self.rewards = {}
        if config.mainReward == "planeDistanceReward":
            self.logged_rewards.append("planeDistanceReward")
            self.rewards["planeDistanceReward"] = PlaneDistanceReward(self.goal_plane)
        elif config.mainReward == "anatomyReward":
            self.logged_rewards.append("anatomyReward")
            self.rewards["anatomyReward"] = AnatomyReward(config.anatomyRewardIDs, incremental=config.incrementalAnatomyReward)
        elif config.mainReward == "both":
            self.logged_rewards.append("planeDistanceReward")
            self.rewards["planeDistanceReward"] = PlaneDistanceReward(self.goal_plane)
            self.logged_rewards.append("anatomyReward")
            self.rewards["anatomyReward"] = AnatomyReward(config.anatomyRewardIDs, incremental=config.incrementalAnatomyReward)
        else:
            raise ValueError('unknown ``mainReward`` parameter. options: <planeDistanceReward,anatomyReward,both> got: {}'.format(config.mainReward))
        # # stepping reward not that effective when considering incremental rewards 
        # #(the agent would already receive a penalty for stepping if it worsens the view) 
        # if abs(config.steppingReward) > 0:
        #     self.logged_rewards.append("steppingReward")
        #     self.rewards["steppingReward"] = SteppingReward(config.steppingReward)
        if abs(config.areaRewardWeight) > 0:
            self.logged_rewards.append("areaReward")
            self.rewards["areaReward"] = AreaReward(config.areaRewardWeight, self.sx*self.sy)
        if abs(config.oobReward) > 0:
            for i in range(config.n_agents):
                self.logged_rewards.append("oobReward_%d"%(i+1))
            self.rewards["oobReward"] = OutOfBoundaryReward(config.oobReward, self.sx, self.sy, self.sz)
        if abs(config.stopReward) > 0:
            assert "anatomyReward" in self.rewards, "stopReward only implemented when using anatomyReward."
            assert config.termination == "learned", "stopReward is only meaningful when learning how to stop (action_size = 7, termination = learned)"
            self.logged_rewards.append("stopReward")
            self.rewards["stopReward"] = StopReward(config.stopReward,
                                                    goal_reward = self.rewards["anatomyReward"].get_anatomy_reward(self.sample_plane(self.goal_state)["seg"]))
        if config.penalize_oob_pixels:
            self.logged_rewards.append("oobPixelsReward")
            self.rewards["oobPixelsReward"] = AnatomyReward("0", is_penalty=True)           

        # monitor oscillations for termination
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
            returns -> plane (torch.tensor of shape (1, 1, self.sy, self.sx)): corresponding plane sampled from the CT volume
                       seg (optional, np.ndarray of shape (self.sy, self.sx)): segmentation map of the sampled plane
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
        # normalize and unsqueeze array if needed.
        if preprocess:
            plane = plane[np.newaxis, np.newaxis, ...]/255
        # add to output
        out["plane"] = plane
        # 3. sample the segmentation if needed
        if return_seg:
            seg = self.Segmentation[X,Y,Z]
            if oob_black == True:
                seg[P < 0] = 0
                seg[P > S] = 0
            out["seg"] = seg
        return out
    
    def get_reward(self, seg, state=None, increment=None):
        """Calculates the corresponding reward of stepping into a state given its segmentation map.
        Params:
        ==========
            seg (np.ndarray of shape (self.sy, self.sx)): segmentation map from which to extract the reward.
            state (np.ndarray): 3 stacked 3D arrays representing the coordinates of the agent.
            increment (np.ndarray): 3 stacked 3D arrays containing the increment action chosen by each agent.
                                    we use this to penalize the agents if they choose to stop on a bad frame.
        Returns -> rewards (dict): the corresponding rewards collected by the agents.
        """
        shared_rewards = {}
        single_rewards = {}
        for key, func in self.rewards.items():
            if key == "planeDistanceReward":
                # this will contain a single reward for all agents
                shared_rewards["planeDistanceReward"] = func(self.get_plane_coefs(*state))
            elif key == "anatomyReward":
                # this will contain a single reward for all agents
                shared_rewards["anatomyReward"] = func(seg) 
            # # stepping reward not that effective when considering incremental rewards 
            # #(the agent would already receive a penalty for stepping if it worsens the view)             
            # elif key == "steppingReward":
            #     # this will contain a single reward for all agents
            #     shared_rewards["steppingReward"] = func(not shared_rewards["anatomyReward"]>0)
            elif key == "areaReward":
                # this will contain a single reward for all agents
                shared_rewards["areaReward"] = func(state)
            elif key == "stopReward":
                # this will contain a single reward for all agents, only really applicable when we are using anatomyReward
                shared_rewards["stopReward"] = func(increment, self.rewards["anatomyReward"].get_anatomy_reward(seg))
            elif key == "oobPixelsReward":
                # this will contain a single reward for all agents
                shared_rewards["oobPixelsReward"] = func(seg)
            elif key == "oobReward":
                # this will contain a reward for each agent
                for i, point in enumerate(state):
                    single_rewards["oobReward_%d"%(i+1)] = func(point) 

        # extract total rewards of each agent
        total_rewards = [sum(shared_rewards.values())]*self.config.n_agents
        if "oobReward" in self.rewards:
            for i in range(self.config.n_agents):
                total_rewards[i]+=single_rewards["oobReward_%d"%(i+1)]
        total_rewards = np.array(total_rewards)
    
        # log these rewards to the current episode count
        shared_rewards.update(single_rewards)
        for r in self.logged_rewards:
            self.current_logs[r] = shared_rewards[r]
            self.logs[r]+=shared_rewards[r]   
        return total_rewards[..., np.newaxis].astype(np.float)

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
        rewards = self.get_reward(sample["seg"], next_state, increment)
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
        return (state, action, rewards, next_state, done), sample
    
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
        # reset the logged rewards for this episode
        self.logs = {r: 0 for r in self.logged_rewards}
        self.current_logs = {r: 0 for r in self.logged_rewards}
        # set the previous plane attribute in the planeDistanceReward if present
        if "planeDistanceReward" in self.rewards:
            self.rewards["planeDistanceReward"].previous_plane = self.get_plane_coefs(*self.state)
        # set the previous anatomyReward if we are using incremental anatomy reward
        if "anatomyReward" in self.rewards and self.config.incrementalAnatomyReward:
            #sample = self.sample_plane(self.state, return_seg=True)
            #self.rewards["anatomyReward"].previous_reward = self.rewards["anatomyReward"].get_anatomy_reward(sample["seg"])
            self.rewards["anatomyReward"].previous_reward = 0
        # reset the oscillation monitoring
        self.oscillates.history.clear()

# ======== LOCATION AWARE ENV ==========
class LocationAwareSingleVolumeEnvironment(SingleVolumeEnvironment):
    """
    Environment object that inherits most of SingleVolumeEnvironment defined above. In addition it will compute binary maps of the agents location in the 3D volume
    and concatenate them to the slice along the channel dimension before passing them to the Qnetwork. We believe this should embed the network with some information
    about the location of each agent, coordinating the action of each agent head with respect to the position of the other 2 agents. However we acknowledge that this
    will yield to a heavily sparse input space and may not be the optimal solution.
    """
    def __init__(self, config, vol_id=0):
        """
        Initialize an Environment object, the environment will focus on a single XCAT/CT volume.
    
        Params that we will use:
        ======
            config (argparse object): contains all options. see ./options/options.py for more details
            vol_id (int): specifies which volume we want to load. Specifically ``config.volume_ids`` may contain
                          all queried volumes by the main.py script. (i.e. parser.volume_ids = 'samp0,samp1,samp2,samp3,samp4,samp5,samp6,samp7')
                          we will load the volume at position ``vol_id`` (default=0)               
        """
        # initialize environment
        SingleVolumeEnvironment.__init__(self, config, vol_id)
        # generate cube for agent position retrieval
        self.agents_cube = np.zeros_like(self.Volume)

    
    def sample_agents_position(self, state, X, Y, Z):
        
        """ function to get the agents postion on the sampled plane
        Params:
        ==========
            state (np.ndarray of shape (3,3)): v-stacked 3D points that will define a particular plane in the CT volume.
            X (np.ndarray ): X coordinates to sample from the volume (output of ``get_plane_from_points()``)
            Y (np.ndarray ): Y coordinates to sample from the volume (output of ``get_plane_from_points()``)
            Z (np.ndarray ): Z coordinates to sample from the volume (output of ``get_plane_from_points()``)

            returns -> plane (np.array of shape (3, self.sx, self.sy)): 3 planes each containing one white dot representing 
                        the position of the corresponding agent
        """
        # Retrieve agents positions and empty volume
        A, B, C = state
      
        #Identify pixels where the agents are with unique values
        if is_in_volume(self.Volume, A):
            self.agents_cube[A[0], A[1], A[2]] = 1
        if is_in_volume(self.Volume, B):
            self.agents_cube[B[0], B[1], B[2]] = 2
        if is_in_volume(self.Volume, C):
            self.agents_cube[C[0], C[1], C[2]] = 3

        # Sample plane defined by the sample_plane function in the empty volume
        plane = self.agents_cube[X,Y,Z]
        # extract the 2D location of the agents
        loc1, loc2, loc3 = np.where(plane == 1), np.where(plane == 2), np.where(plane == 3)
        # make each map a smooth (distances from the white dot)
        maps = []
        for loc in [loc1,loc2,loc3]:
            rows, cols = np.meshgrid(np.arange(self.sx), np.arange(self.sy))
            try:
                rows-=loc[0]
                cols-=loc[1]
                arr = np.stack([abs(rows),abs(cols)])
                maps.append(arr.max(0))
            except:
                maps.append(np.zeros_like(plane))
        # Separate each agent into it's own channel, set them as white (255 since we use uint8)
        #plane = np.stack((plane == 1, plane == 2, plane == 3)).astype(np.uint8)*255
        plane = np.stack(maps)
        # reset the modified pixels to black
        if is_in_volume(self.Volume, A):
            self.agents_cube[A[0], A[1], A[2]] = 0
        if is_in_volume(self.Volume, B):
            self.agents_cube[B[0], B[1], B[2]] = 0
        if is_in_volume(self.Volume, C):
            self.agents_cube[C[0], C[1], C[2]] = 0
            
        return plane
    
    def sample_plane(self, state, return_seg=False, oob_black=True, preprocess=False):
        """ function to sample a plane from 3 3D points (state)
        Params:
        ==========
            state (np.ndarray of shape (3,3)): v-stacked 3D points that will define a particular plane in the CT volume.
            return_seg (bool): flag if we wish to return the corresponding segmentation map. (default=False)
            oob_black (bool): flack if we wish to mask out of volume pixels to black. (default=True)
            preprocess (bool): if to preprocess the plane before returning it (unsqueeze to BxCxHxW, normalize and/or add positional binary maps) 
            returns -> plane (torch.tensor of shape (1, 1, self.sy, self.sx)): corresponding plane sampled from the CT volume
                       seg (optional, np.ndarray of shape (self.sy, self.sx)): segmentation map of the sampled plane
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
        # concatenate binary location maps along channel diension
        pos = self.sample_agents_position(state, X, Y, Z)
        plane = np.concatenate((plane[np.newaxis, ...], pos), axis=0)
        # normalize and unsqueeze array if needed.  if necessary.
        if preprocess:
            plane = plane[np.newaxis, ...]/255
        # add to output
        out["plane"] = plane
        # 3. sample the segmentation if needed
        if return_seg:
            seg = self.Segmentation[X,Y,Z]
            if oob_black == True:
                seg[P < 0] = 0
                seg[P > S] = 0
            out["seg"] = seg
        return out

# ==== HELPER FUNCTIONS ====
def get_centroid(data, seg_id, norm=False):
    volume = data == seg_id # Get binary map for the specified seg id
    values = np.array(ndimage.measurements.center_of_mass(volume), dtype=np.int)
    if norm:
        values = values / data.shape
    return values

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

def draw_sphere(arr, point, r=3, color = 255):
    i,j,k = np.indices(arr.shape)
    dist = np.sqrt( (point[0]-i)**2 + (point[1]-j)**2 + (point[2]-k)**2)
    arr[dist < r] = color
    return arr

def is_in_volume(volume, point):
    sx, sy, sz = volume.shape
    return  (point[0] >= 0 and point[0] < sx) and (point[1] >= 0 and point[1] < sy) and (point[2] >= 0 and point[2] < sz)

# Trick function to enable one-line conditions
def doNothing():
    return None