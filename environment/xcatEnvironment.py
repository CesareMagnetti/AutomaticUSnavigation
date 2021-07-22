from environment.baseEnvironment import BaseEnvironment
from rewards.rewards import *
import numpy as np
import SimpleITK as sitk
import os

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

        # setup the reward function handling:
        self.logged_rewards = ["anatomyReward"]
        self.rewards = {"anatomyReward": AnatomyReward(config.anatomyRewardIDs)}
        if abs(config.steppingReward) > 0:
            self.logged_rewards.append("steppingReward")
            self.rewards["steppingReward"] = SteppingReward(config.steppingReward)
        if abs(config.areaRewardWeight) > 0:
            self.logged_rewards.append("areaReward")
            self.rewards["areaReward"] = AreaReward(config.areaRewardWeight, self.sx*self.sy)
        if abs(config.areaRewardWeight) > 0:
            for i in range(config.n_agents):
                self.logged_rewards.append("oobReward_%d"%(i+1))
            self.rewards["oobReward"] = OutOfBoundaryReward(config.oobReward, self.sx, self.sy, self.sz)
        if abs(config.stopReward) > 0:
            self.logged_rewards.append("stopReward")
            self.rewards["stopReward"] = StopReward(config.stopReward)
                
        # get starting configuration
        self.reset()

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
        # 3. sample the segmentation if needed
        if return_seg:
            return plane, self.Segmentation[X,Y,Z]
        else:
            return plane
    
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
            if key == "anatomyReward":
                # this will contain a single reward for all agents
                shared_rewards["anatomyReward"] = func(seg)               
            elif key == "steppingReward":
                # this will contain a single reward for all agents
                shared_rewards["steppingReward"] = func(not shared_rewards["anatomyReward"]>0)
            elif key == "areaReward":
                # this will contain a single reward for all agents
                shared_rewards["areaReward"] = func(state)
            elif key == "stopReward":
                # this will contain a single reward for all agents
                shared_rewards["stopReward"] = func(increment, not shared_rewards["anatomyReward"]>0)
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
        """
        # get the increment corresponding to this action
        increment = np.vstack([self.mapActionToIncrement(act) for act in action])
        # step into the next state
        state = self.state
        next_state = state + increment
        # observe the next plane and get the reward from segmentation map
        next_slice, segmentation = self.sample_plane(state=next_state, return_seg=True, preprocess=preprocess)
        rewards = self.get_reward(segmentation, next_state, increment)
        # update the current state
        self.state = next_state
        # return transition and the next_slice which has already been sampled
        return (state, action, rewards, next_state), next_slice
    
    def reset(self):
        # sample a random plane (defined by 3 points) to start the episode from
        if self.config.easy_objective:
            # these planes correspond more or less to a 4-chamber view
            pointA = np.array([np.random.uniform(low=0.85, high=1)*self.sx-1,
                              0,
                              np.random.uniform(low=0.7, high=0.92)*self.sz-1])

            pointB = np.array([np.random.uniform(low=0.3, high=0.43)*self.sx-1,
                              self.sy-1,
                              0])

            pointC = np.array([np.random.uniform(low=0.3, high=0.43)*self.sx-1,
                              self.sy-1,
                              self.sz-1]) 
        else:
            pointA = np.array([np.random.uniform(low=0., high=1)*self.sx-1,
                              0,
                              np.random.uniform(low=0., high=1.)*self.sz-1])

            pointB = np.array([np.random.uniform(low=0., high=1.)*self.sx-1,
                              self.sy-1,
                              np.random.uniform(low=0., high=1.)*self.sz-1])

            pointC = np.array([np.random.uniform(low=0., high=1.)*self.sx-1,
                              self.sy-1,
                              np.random.uniform(low=0., high=1.)*self.sz-1])             
        # stack points to define the state
        self.state = np.vstack([pointA, pointB, pointC]).astype(np.int)
        # reset the logged rewards for this episode
        self.logs = {r: 0 for r in self.logged_rewards}
        self.current_logs = {r: 0 for r in self.logged_rewards}

# ======= TEST NEW PLANE SAMPLING =======
class ImageTransformation():
    def __init__(self) -> None:
        self.tl = [] # transformation list

    def add(self, name, value):
        assert name in ['flip', 'rot']
        if len(self.tl) > 0 and self.tl[-1][0] == name:
            if name == 'flip':
                if value in self.tl[-1][1]:
                    self.tl[-1][1].remove(value)
                    if len(self.tl[-1][1]) == 0:
                        del self.tl[-1]
                else:
                    self.tl[-1][1].add(value)
            elif name == 'rot':
                self.tl[-1][1] = (self.tl[-1][1] + value)%4
                if self.tl[-1][1] == 0:
                    del self.tl[-1]
        else:
            if name == 'flip':
                self.tl.append([name,{value}])
            elif name == 'rot':
                self.tl.append([name,value])
     
    def __call__(self, img):
        for t in self.tl:
            if t[0] == 'flip':
                if 0 in t[1]:
                    img = np.flip(img, axis=0)
                if 1 in t[1]:
                    img = np.flip(img, axis=1)
            elif t[0] == 'rot':
                img = np.rot90(img, k=t[1])
        return img
    
    def __str__(self) -> str:
        string = []
        for i, t in enumerate(self.tl):
                string.append(f'{i}: {t[0]} {t[1]}')
        return str(string)

class TestNewSamplingEnvironment(SingleVolumeEnvironment):
    """Inherits everything from the standard environment class but overrides the plane sampling mehtod.
    """
    def __init__(self, config, vol_id=0):
        # initialize environment
        SingleVolumeEnvironment.__init__(self, config)

        # new plane sampling handles
        self.prev_ax = None
        self.trans = ImageTransformation()

    # overwrite sample plane function
    def sample_plane(self, state, return_seg=False, oob_black=True, preprocess=False):
        # get plane coefs
        a,b,c,d = self.get_plane_coefs(*state)
        # new plane sampling from Hadrien's repo
        main_ax = np.argmax([abs(a), abs(b), abs(c)])
        sa, sb, sc = np.roll([self.sx, self.sy, self.sz],shift=(2-main_ax))
        na, nb, nc = np.roll([a, b, c],   shift=(2-main_ax))        
        A,B = np.meshgrid(np.arange(sa), np.arange(sb), indexing='ij')
        C = (d - na * A - nb * B) / nc

        C = C.round().astype(np.int)
        P = C.copy()
        S = sc-1

        C[C <= 0]  = 0
        C[C >= sc] = sc-1

        # Solves weird issue with np.roll on list of arrays
        ABC = np.stack((A, B, C))
        idxX, idxY, idxZ = np.roll([0, 1, 2],shift=(main_ax-2))
        X, Y, Z = ABC[idxX], ABC[idxY], ABC[idxZ]

        plane = self.Volume[X, Y, Z]

        if oob_black == True:
            plane[P < 0] = 0
            plane[P > S] = 0
        
        # Plane adjustements:
        if main_ax == 2:
            plane = np.rot90(plane, k=1)

        if self.prev_ax == 2 and main_ax==1:
            self.trans.add('flip', 0)
        elif self.prev_ax == 0 and main_ax==2:
            self.trans.add('flip', 0)
        elif self.prev_ax == 2 and main_ax==0:
            self.trans.add('flip', 0)
            self.trans.add('flip', 1)
        elif self.prev_ax == 0 and main_ax==1:
            if ['flip', {0}] in self.trans.tl:
                self.trans.add('rot', -1)
            else:
                self.trans.add('rot', 1)
        elif self.prev_ax == 1 and main_ax==0:
            self.trans.add('rot', -1)
            self.trans.add('flip', 0)
        
        self.prev_ax = main_ax
        plane = self.trans(plane)

        # normalize and unsqueeze array if needed.
        if preprocess:
            plane = plane[np.newaxis, np.newaxis, ...]/255
        
        if return_seg:
            seg = self.Segmentation[X,Y,Z]
            # Plane adjustements:
            if main_ax == 2:
                seg = np.rot90(seg, k=1)
            seg = self.trans(seg)
            return plane, seg
        else:
            return plane
            

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
        SingleVolumeEnvironment.__init__(self, config)
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
      
        # Identify pixels where the agents are with unique values
        if is_in_volume(self.Volume, A):
            self.agents_cube[A[0], A[1], A[2]] = 1
        if is_in_volume(self.Volume, B):
            self.agents_cube[B[0], B[1], B[2]] = 2
        if is_in_volume(self.Volume, C):
            self.agents_cube[C[0], C[1], C[2]] = 3

        # Sample plane defined by the sample_plane function in the empty volume
        plane = self.agents_cube[X,Y,Z]
        # Separate each agent into it's own channel
        plane = np.stack((plane == 1, plane == 2, plane == 3)).astype(np.uint8)

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
        # 3. sample the segmentation if needed
        if return_seg:
            return plane, self.Segmentation[X,Y,Z]
        else:
            return plane

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