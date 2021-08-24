
from environment.xcatEnvironment import *
import torch, os, functools
import torch.nn as nn
from skimage.draw import polygon, circle
import torch.nn.functional as F

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
            plane = self.CT2USmodel(sample["planeCT"]).clamp(-1.0, 1.0).cpu().numpy()
            # the cyclegan uses tanh, hence output is in (-1,1), need to bring it to (0,1)
            # sample["planeUS"] = (plane +1)/2 # now its in (0,1)
            sample["plane"] = (plane +1)/2 # now its in (0,1)
            sample["plane"] = mask_array(sample["plane"]) # apply US mask
        # remove planeCT from GPU and convert to numpy
        sample["planeCT"] = sample["planeCT"].detach().cpu().numpy() 
        # both planes are already unsqueezed and normalized, if we did not want to preprocess the image then squeeze and multiply by 255 to undo preprocessing
        # sample["plane"] = sample["plane"][np.newaxis, np.newaxis, ...]/255
        if not preprocess:
            # sample["planeUS"] = sample["planeUS"].squeeze()*255
            sample["plane"] = sample["plane"].squeeze()*255
            sample["planeCT"] = sample["planeCT"].squeeze()*255
        return sample

class LocationAwareCT2USSingleVolumeEnvironment(LocationAwareSingleVolumeEnvironment):
    def __init__(self, config, vol_id=0):
        LocationAwareSingleVolumeEnvironment.__init__(self, config, vol_id)
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
            plane = self.CT2USmodel(sample["planeCT"]).clamp(-1.0, 1.0).cpu().numpy()
            # the cyclegan uses tanh, hence output is in (-1,1), need to bring it to (0,1)
            plane = (plane +1)/2 # now its in (0,1)
            # concatenate new plane with the positional embeddings
            #pos = [p/255 for p in np.split(sample["plane"], sample["plane"].shape[0], axis=0)]
            pos = sample["plane"][1:, ...]/255
            sample["plane"] = np.concatenate([plane, + pos[np.newaxis, ...]], axis=1)
        # remove planeCT from GPU and convert to numpy
        sample["planeCT"] = sample["planeCT"].detach().cpu().numpy() 
        # both planes are already unsqueezed and normalized, if we did not want to preprocess the image then squeeze and multiply by 255 to undo preprocessing
        if not preprocess:
            sample["plane"] = sample["plane"].squeeze()*255
            sample["planeCT"] = sample["planeCT"].squeeze()*255
        return sample

# ===== PREPROCESSING FOR US IMAGES ======
def preprocessCT(CT):
    # # normalize and mask
    # CT = mask_array(CT/255)
    # # convert to tensor and unsqueeze
    # CT = torch.from_numpy(CT).unsqueeze(0).unsqueeze(0)
    # # apply salt-pepper noise
    # noise = torch.rand(*CT.shape)
    # proba = 0.1
    # disrupt = 1.5
    # CT[noise < proba] = CT[noise < proba]/disrupt
    # CT[noise > 1-proba] = CT[noise > 1-proba]*disrupt

    # normalize and invert intensities
    CT = (255 - CT)/255
    # convert to tensor and unsqueeze
    CT = torch.from_numpy(CT).unsqueeze(0).unsqueeze(0)
    return CT

# draw a mask on US image
def mask_array(arr):
    #assert arr.ndim == 2 # on 1 channel images broadcasting will work
    _,_,sx, sy = arr.shape # H, W
    mask = np.ones((sx, sy))
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
    model = ResnetGenerator(input_nc=1, output_nc=1, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True, n_blocks=9,
                            padding_type='reflect', no_antialias=True, no_antialias_up=False, opt=None)
    state_dict = torch.load(os.path.join(os.getcwd(), "environment", "CT2USmodels", "%s.pth"%name), map_location='cpu')
    print("loading: {} ...".format(os.path.join(os.getcwd(), "environment", "CT2USmodels", "%s.pth"%name)))
    model.load_state_dict(state_dict)
    return model

# THE FOLLOWING CODE WAS ADAPTED FROM THE CYCLEGAN-PIX2PIX REPO: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
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
        super(ResnetGenerator, self).__init__()
        self.opt = opt
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
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake

# class ResnetGenerator(nn.Module):
#     """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

#     We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
#     """

#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect', transposeConv = False):
#         """Construct a Resnet-based generator

#         Parameters:
#             input_nc (int)      -- the number of channels in input images
#             output_nc (int)     -- the number of channels in output images
#             ngf (int)           -- the number of filters in the last conv layer
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers
#             n_blocks (int)      -- the number of ResNet blocks
#             padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
#             transposeConv (bool) -- if using transposeConvolutions or nearest neighbour upsampling + standard convolution
#         """
#         assert(n_blocks >= 0)
#         super(ResnetGenerator, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d

#         model = [nn.ReflectionPad2d(3),
#                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
#                  norm_layer(ngf),
#                  nn.ReLU(True)]

#         n_downsampling = 2
#         for i in range(n_downsampling):  # add downsampling layers
#             mult = 2 ** i
#             model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                       norm_layer(ngf * mult * 2),
#                       nn.ReLU(True)]

#         mult = 2 ** n_downsampling
#         for i in range(n_blocks):       # add ResNet blocks

#             model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

#         for i in range(n_downsampling):  # add upsampling layers
#             mult = 2 ** (n_downsampling - i)
#             if transposeConv:
#                 model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                             kernel_size=3, stride=2,
#                                             padding=1, output_padding=1,
#                                             bias=use_bias),
#                         norm_layer(int(ngf * mult / 2)),
#                         nn.ReLU(True)]
#             else:
#                 model += [  nn.Upsample(scale_factor=2, mode='nearest'),
#                             nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
#                             norm_layer(int(ngf * mult / 2)),
#                             nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(3)]
#         model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
#         model += [nn.Tanh()]

#         self.model = nn.Sequential(*model)

#     def forward(self, input):
#         """Standard forward"""
#         return self.model(input)

def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt

class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample2(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)

class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

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