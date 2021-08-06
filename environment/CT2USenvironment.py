
from environment.xcatEnvironment import *
import torch, os, functools
import torch.nn as nn
from skimage.draw import polygon, circle

# ===== CT2US ENVIRONMENT =====
class CT2USSingleVolumeEnvironment(SingleVolumeEnvironment):
    def __init__(self, config, vol_id=0):
        SingleVolumeEnvironment.__init__(self, config, vol_id)
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
        # call environemnt sample_plane function, do not preprocess as we will have to do it ourselves anyways
        sample = super().sample_plane(state=state, return_seg=return_seg, oob_black=oob_black, preprocess=False)
        # we need to preprocess the CT slice before 
        sample["planeCT"] = preprocessCT(sample["plane"]).float().to(self.config.device)
        # transform the image to US (do not store gradients)
        with torch.no_grad():
            sample["plane"] = self.CT2USmodel(sample["planeCT"]).detach().cpu().numpy()
        # remove planeCT from GPU and convert to numpy
        sample["planeCT"] = sample["planeCT"].detach().cpu().numpy() 
        # both planes are already unsqueezed and normalized, if we did not want to preprocess the image then squeeze and multiply by 255 to undo preprocessing
        if not preprocess:
            sample["plane"] = sample["plane"].squeeze()*255
            sample["planeCT"] = sample["planeCT"].squeeze()*255
        return sample

class LocationAwareCT2USSingleVolumeEnvironment(LocationAwareSingleVolumeEnvironment):
    def __init__(self, config, vol_id=0):
        SingleVolumeEnvironment.__init__(self, config, vol_id)
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
        # call environemnt sample_plane function, do not preprocess as we will have to do it ourselves anyways
        sample = super().sample_plane(state=state, return_seg=return_seg, oob_black=oob_black, preprocess=False)
        # we need to preprocess the CT slice before 
        sample["planeCT"] = preprocessCT(sample["plane"][0, ...]).float().to(self.config.device)
        # transform the image to US (do not store gradients)
        with torch.no_grad():
            plane = self.CT2USmodel(sample["planeCT"]).detach().cpu().numpy()
            sample["plane"] = np.stack([plane[0, ...], sample["plane"][1:, ...]/255])
        # remove planeCT from GPU and convert to numpy
        sample["planeCT"] = sample["planeCT"].detach().cpu().numpy() 
        # both planes are already unsqueezed and normalized, if we did not want to preprocess the image then squeeze and multiply by 255 to undo preprocessing
        if not preprocess:
            sample["plane"] = sample["plane"].squeeze()*255
            sample["planeCT"] = sample["planeCT"].squeeze()*255
        return sample

# ===== PREPROCESSING FOR US IMAGES ======
def preprocessCT(CT):
    # normalize and mask
    CT = mask_array(CT/255)
    # convert to tensor and unsqueeze
    CT = torch.from_numpy(CT).unsqueeze(0).unsqueeze(0)
    # apply salt-pepper noise
    noise = torch.rand(*CT.shape)
    proba = 0.1
    disrupt = 1.5
    CT[noise < proba] = CT[noise < proba]/disrupt
    CT[noise > 1-proba] = CT[noise > 1-proba]*disrupt
    return CT
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

# ===== THE FOLLOWING FUNCTIONS HANDLE THE CYCLEGAN NETWORK INSTANCIATION AND WEIGHTS LOADING ====

def get_model(name, use_cuda=False):
    # instanciate cyclegan architecture used in CT2UStransfer (this is also the default architecture recommended by the authors)
    model = ResnetGeneratorNoTrans(1, 1, 64, norm_layer=nn.InstanceNorm2d, use_dropout=True, n_blocks=9)
    state_dict = torch.load(os.path.join(os.getcwd(), "environment", "CT2USmodels", "%s.pth"%name), map_location='cpu')
    print("loading: {} ...".format(os.path.join(os.getcwd(), "environment", "CT2USmodels", "%s.pth"%name)))
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