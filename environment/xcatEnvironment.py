from environment.baseEnvironment import BaseEnvironment
from rewards.rewards import *
import numpy as np
import SimpleITK as sitk
from skimage.draw import polygon, circle
import torch, os, functools
import torch.nn as nn

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
        if config.penalize_oob_pixels:
            self.logged_rewards.append("oobPixelsReward")
            self.rewards["oobPixelsReward"] = AnatomyReward("0", is_penalty=True)           
                
        # get starting configuration
        self.reset()

    def sample_plane(self, state, return_seg=False, oob_black=True, preprocess=False, **kwargs):
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
            seg = self.Segmentation[X,Y,Z]
            if oob_black == True:
                seg[P < 0] = 0
                seg[P > S] = 0
            return plane, seg
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

    def step(self, action, preprocess=False, return_seg=False):
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
        if return_seg:
            return (state, action, rewards, next_state), next_slice, segmentation
        else:
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

        if main_ax == 0:
            X,Y,Z = C,A,B
        elif main_ax == 1:
            X,Y,Z = B,C,A
        else:
            X,Y,Z = A,B,C

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
    
    def sample_agents_position(self, state, X, Y, Z, radius=10):
        
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

class CT2USSingleVolumeEnvironment(SingleVolumeEnvironment):
    def __init__(self, config):
        SingleVolumeEnvironment.__init__(self, config)
        # load the queried CT2US model
        self.CT2USmodel = get_model(config.ct2us_model_name).to(config.device)
    
    # we need to rewrite the sample_plane and step functions, everything else should work fine
    def sample_plane(self, state, return_seg=False, oob_black=True, preprocess=False, return_ct = False):
        """ function to sample a plane from 3 3D points (state) and convert it to US
        Params:
        ==========
            state (np.ndarray of shape (3,3)): v-stacked 3D points that will define a particular plane in the CT volume.
            return_seg (bool): flag if we wish to return the corresponding segmentation map. (default=False)
            oob_black (bool): flack if we wish to mask out of volume pixels to black. (default=True)
            preprocess (bool): if to preprocess the plane before returning it (unsqueeze to BxCxHxW, normalize and/or add positional binary maps) 
            returns -> plane (torch.tensor of shape (1, 1, self.sy, self.sx)): corresponding plane sampled from the CT volume transformed to US
                       seg (optional, np.ndarray of shape (self.sy, self.sx)): segmentation map of the sampled plane
        """
        # sample the CT plane and the segmentation map calling the parent sample_plane method
        planeCT, seg = super().sample_plane(state=state, return_seg=True, oob_black=oob_black, preprocess=False)

        # preprocess the CT slice before sending to transformation network
        planeCT = mask_array(planeCT)
        planeCT = torch.tensor(planeCT/255).unsqueeze(0).unsqueeze(0).float().to(self.config.device)

        # transform the image to US (do not store gradients)
        with torch.no_grad():
            planeUS = self.CT2USmodel(planeCT).detach().cpu().numpy()
        
        # image is already unsqueezed and normalized, if we did not want to preprocess the image then squeeze 
        # and multiply by 255 to undo preprocessing
        if not preprocess:
            planeUS = planeUS.squeeze()*255
        
        if not return_seg and not return_ct:
            return planeUS
        elif return_seg and not return_ct:
            return planeUS, seg
        elif not return_seg and return_ct:
            return planeUS, planeCT.detach().cpu().numpy()
        else:
            return planeUS, seg, planeCT.detach().cpu().numpy()
    
    def step(self, action, preprocess=False, return_seg=False, return_ct=False):
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
        next_slice, segmentation, next_sliceCT = self.sample_plane(state=next_state, return_seg=True, return_ct=True, preprocess=preprocess)
        rewards = self.get_reward(segmentation, next_state, increment)
        # update the current state
        self.state = next_state
        # return transition and the next_slice which has already been sampled        
        if not return_seg and not return_ct:
            return (state, action, rewards, next_state), next_slice
        elif return_seg and not return_ct:
            return (state, action, rewards, next_state), next_slice, segmentation
        elif not return_seg and return_ct:
            return (state, action, rewards, next_state), next_slice, next_sliceCT
        else:
            return (state, action, rewards, next_state), next_slice, segmentation, next_sliceCT


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

# draw a mask on US image
def mask_array(arr):
    assert arr.ndim == 2
    sx, sy = arr.shape # H, W
    mask = np.ones_like(arr)*1
    # Bottom circular shape
    rr, cc = circle(0, int(sy//2), sy-1)
    rr[rr >= sy] = sy-1
    cc[cc >= sy] = sy-1
    rr[rr < 0] = 0
    cc[cc < 0] = 0
    mask[rr, cc] = 0
    # Left triangle
    r = (0, 0, sx*5/9)
    c = (0, sy//2, 0)
    rr, cc = polygon(r, c)
    mask[rr, cc] = 1
    # Right triangle
    r = (0,      0, sx*5/9)
    c = (sy//2, sy-1, sy-1)
    rr, cc = polygon(r, c)
    mask[rr, cc] = 1
    mask = (1-mask).astype(np.bool)
    return mask*arr

# Trick function to enable one-line conditions
def doNothing():
    return None

# ===== THE FOLLOWING FUNCTIONS HANDLE THE CYCLEGAN NETWORK INSTANCIATION AND WEIGHTS LOADING ====

def get_model(name, use_cuda=False):
    # instanciate cyclegan architecture used in CT2UStransfer (this is also the default architecture recommended by the authors)
    model = ResnetGeneratorNoTrans(1, 1, 64, norm_layer=nn.InstanceNorm2d, use_dropout=True, n_blocks=9)
    state_dict = torch.load(os.path.join(os.getcwd(), "environment", "CT2USmodels", "%s.pth"%name), map_location='cpu')
    model.load_state_dict(state_dict)
    return model

# THE FOLLOWING CODE WAS ADAPTED FROM THE CYCLEGAN-PIX2PIX REPO: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

class ResnetGeneratorNoTrans(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGeneratorNoTrans, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            # model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
            #                              kernel_size=3, stride=2,
            #                              padding=1, output_padding=1,
            #                              bias=use_bias),
            #           norm_layer(int(ngf * mult / 2)),
            #           nn.ReLU(True)]
            model += [  nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out