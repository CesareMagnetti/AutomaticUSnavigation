import torch
import torch.nn as nn
import numpy as np
import functools
import os

# ==== THE FOLLOWING FUNCTION HANDLE 2D PLANE SAMPLING OF A 3D VOLUME ====
def get_plane_coefs(p1, p2, p3):
    """ Gets the coefficients of a 3D plane given the coordinates of 3 3D points
    """
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp
    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)
    return a, b, c, d

def get_plane_from_points(points, shape):
        """ function to sample a plane from 3 3D points (state)
        Params:
        ==========
            points (np.ndarray of shape (3,3)): v-stacked 3D points that will define a particular plane in the volume.
            shape (np.ndarry): shape of the 3D volume to sample plane from

            returns -> (X,Y,Z) ndarrays that will index Volume in order to extract a specific plane.
        """
        # get plane coefs
        a,b,c,d = get_plane_coefs(*points)
        # get volume shape
        sx, sy, sz = shape
        # extract corresponding slice
        main_ax = np.argmax([abs(a), abs(b), abs(c)])
        if main_ax == 0:
            Y, Z = np.meshgrid(np.arange(sy), np.arange(sz), indexing='ij')
            X = (d - b * Y - c * Z) / a

            X = X.round().astype(np.int)
            P = X.copy()
            S = sx-1

            X[X <= 0] = 0
            X[X >= sx] = sx-1

        elif main_ax==1:
            X, Z = np.meshgrid(np.arange(sx), np.arange(sz), indexing='ij')
            Y = (d - a * X - c * Z) / b

            Y = Y.round().astype(np.int)
            P = Y.copy()
            S = sy-1

            Y[Y <= 0] = 0
            Y[Y >= sy] = sy-1
        
        elif main_ax==2:
            X, Y = np.meshgrid(np.arange(sx), np.arange(sy), indexing='ij')
            Z = (d - a * X - b * Y) / c

            Z = Z.round().astype(np.int)
            P = Z.copy()
            S = sz-1

            Z[Z <= 0] = 0
            Z[Z >= sz] = sz-1
        
        return (X,Y,Z), P, S

def convertCoordinates3Dto2D(p1, p2, p3, origin = None):
        """ defines the 2D plane, the basis vectors of the 2D coord systems and the origin of the new coord system.
        """
        # These two vectors are in the plane
        v1 = p3 - p1
        v2 = p2 - p1
        # the cross product is a vector normal to the plane
        n = np.cross(v1, v2)
        # define first basis vector
        ex = np.cross(n, ex)/np.linalg.norm(np.cross(v1, n))
        # define the second basis vector
        ey = np.cross(n, ex)/np.linalg.norm(np.cross(n, ex))
        # we define the origin
        if origin is None:
            origin = np.array([0., 0., 0.])
        # convert the three points
        _out = []
        for point in [p1, p2, p3]:
            x = np.dot(point - origin, ex)
            y = np.dot(point - origin, ey)
            _out.append(np.array([x, y]))
        # return new 2D state representation
        return np.vstack(_out)
        
# ===== THE FOLLOWING FUNCTIONS HANDLE THE CYCLEGAN NETWORK INSTANCIATION AND WEIGHTS LOADING ====

def get_model(name, use_cuda=False):
    # instanciate cyclegan architecture used in CT2UStransfer (this is also the default architecture recommended by the authors)
    model = JohnsonResnetGenerator(1, 1, 64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9)
    state_dict = torch.load(os.path.join(os.getcwd(), "environment", "models", "%s.pth"%name), map_location='cpu')
    model.load_state_dict(state_dict)
    return model

# THE FOLLOWING MODELS WHERE TAKEN FROM THE CYCLEGAN-PIX2PIX REPO: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

class JohnsonResnetGenerator(nn.Module):
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
        super(JohnsonResnetGenerator, self).__init__()
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
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
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

