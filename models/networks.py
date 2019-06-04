import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################


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
    # print(norm_type)
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    elif norm_type == 'spectral':
        norm_layer = functools.partial(nn.utils.spectral_norm)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    # print(norm_layer)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], nz=0):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    # norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm=norm, use_dropout=use_dropout, n_blocks=9, nz=nz)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm=norm, use_dropout=use_dropout, n_blocks=6, nz=nz)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm=norm, use_dropout=use_dropout, nz=nz)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm=norm, use_dropout=use_dropout, nz=nz)
    # elif netG == 'atte_noise_256'
    #     net =
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], nz_D=4):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    # norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm=norm, nz_D=nz_D)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm=norm, nz_D=nz_D)
    # elif netD == 'pixel':     # classify if each pixel is real or fake
    #     net = PixelDiscriminator(input_nc, ndf, norm=norm)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm='instance', use_dropout=False, n_blocks=6, padding_type='reflect', nz=4):
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

        self.nz = nz

        norm_layer = get_norm_layer(norm_type=norm)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc + self.nz, ngf, kernel_size=7, padding=0, bias=use_bias),
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
            # nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
            #                              kernel_size=3, stride=2,
            #                              padding=1, output_padding=1,
            #                              bias=use_bias)
            model += [nn.Upsample(scale_factor=2, mode='bilinear'),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z):
        """Standard forward"""
        z = z[:, 0:self.nz]

        z = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), input.size(2), input.size(3))
        input_and_noise = torch.cat([input, z], 1)
        return self.model(input_and_noise)


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


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm='batch', use_dropout=False, nz=4):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm=norm, nz=nz,
                                             innermost=True, skip=False)  # add the innermost layer
        for i in range(num_downs - 5): # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, nz=nz,
                                                 norm=norm, use_dropout=use_dropout, skip=False, downsize=True)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm=norm, skip=False, nz=nz,)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm=norm, nz=nz, skip=False)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm=norm, nz=nz, skip=False)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, nz=nz,
                                             norm=norm, skip=False)  # add the outermost layer

        for name, param in self.named_parameters():
            print(name)

    def forward(self, input, z):
        """Standard forward"""
        return self.model(input, z)



class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm='batch', use_dropout=False, skip=False, nz=0, downsize=True):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        self.skip = skip

        norm_layer = get_norm_layer(norm_type=norm)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        # downconv = nn.Conv2d(input_nc + self.nz, inner_nc, kernel_size=4,
        #                      stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)
        # downnorm = norm_layer(inner_nc + self.nz)
        uprelu = nn.ReLU()
        # upnorm = norm_layer(outer_nc)

        if outermost:
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)
            down = [downconv]

            up = nn.Upsample(scale_factor=2, mode='bilinear')
            uppad = nn.ReflectionPad2d(1)
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=0)
            up = [uprelu, up, uppad, upconv, nn.Tanh()]
        elif innermost:
            self.gamma = nn.Parameter(torch.tensor(1e-9))

            downconv = nn.Conv2d(input_nc + self.nz, inner_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]

            up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
            uppad = nn.ReflectionPad2d(1)
            upconv = nn.Conv2d(inner_nc + self.nz, outer_nc, kernel_size=3, stride=1, padding=0)
            if norm == 'spectral':
                # print('this is the innerest')
                upnorm = norm_layer(nn.Conv2d(inner_nc + self.nz, outer_nc, kernel_size=3, stride=1, padding=0))
                up = [uprelu, up_sample, uppad, upnorm]
                # print(upnorm)
            else:
                # print('Don\'t use spectral normalization' )
                upnorm = norm_layer(outer_nc)
                up = [uprelu, up_sample, uppad, upconv, upnorm]

            # model = down + up
        else:
            # print('this is the middle')
            # print(norm)

            self.gamma = nn.Parameter(torch.tensor(1e-9))

            if skip:
                self.f = nn.Conv2d(in_channels=input_nc, out_channels=input_nc // 8, kernel_size=1)
                self.g = nn.Conv2d(in_channels=input_nc, out_channels=input_nc // 8, kernel_size=1)
                self.h = nn.Conv2d(in_channels=input_nc, out_channels=input_nc, kernel_size=1)
                self.softmax = nn.Softmax(dim=2)


            if downsize:
                if norm == 'spectral':
                    # print('this use spectral normaltization')
                    downnorm = norm_layer(nn.Conv2d(input_nc + self.nz, inner_nc, kernel_size=3, stride=2, padding=1, bias=use_bias))
                    down = [downrelu, downnorm]
                else:
                    # print('Don\'t use spectral normalization' )
                    downconv = nn.Conv2d(input_nc + self.nz, inner_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)
                    downnorm = norm_layer(inner_nc)
                    down = [downrelu, downconv, downnorm]


                up_sampling = nn.Upsample(scale_factor=2, mode='bilinear')
                uppad = nn.ReflectionPad2d(1)
                if norm == 'spectral':
                    # print('this use spectral normaltization')
                    upnorm = norm_layer(nn.Conv2d(inner_nc * 2 + self.nz, outer_nc, kernel_size=3, stride=1, padding=0))
                    up = [uprelu, up_sampling, uppad, upnorm]
                else:
                    # print('Don\'t use spectral normalization' )
                    upconv = nn.Conv2d(inner_nc * 2 + self.nz, outer_nc, kernel_size=3, stride=1, padding=0)
                    upnorm = norm_layer(outer_nc)
                    up = [uprelu, up_sampling, uppad, upconv, upnorm]
            else:
                if norm == 'spectral':
                    # print('this use spectral normaltization')
                    # print('this use spectral normaltization')
                    downnorm = norm_layer(nn.Conv2d(input_nc + self.nz, inner_nc, kernel_size=3, stride=1, padding=1, bias=use_bias))
                    down = [downrelu, downnorm]
                else:
                    # print('Don\'t use spectral normalization' )
                    downconv = nn.Conv2d(input_nc + self.nz, inner_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
                    downnorm = norm_layer(inner_nc)
                    down = [downrelu, downconv, downnorm]

                if norm == 'spectral':
                    upnorm = norm_layer(nn.Conv2d(inner_nc * 2 + self.nz, outer_nc, kernel_size=3, stride=1, padding=1))
                    up = [uprelu, upnorm]
                else:
                    # print('Don\'t use spectral normalization' )
                    upconv = nn.Conv2d(inner_nc * 2 + self.nz, outer_nc, kernel_size=3, stride=1, padding=1)
                    upnorm = norm_layer(outer_nc)
                    up = [uprelu, upconv, upnorm]


        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)
        self.submodule = submodule


    def forward(self, x, z):
        if self.outermost:
            # print('this is outermost')
            # print(x.shape)
            # print('This is a new beginning')

            x1 = self.down(x)
            # print('----------------------------------')
            # print(x1.shape)

            x2 = self.submodule(x1, z)
            output = self.up(x2)

            # print('after up')
            # print(z.shape)
            #
            # print('this is a end of one iteration')
            # print(output.shape)
            return output
        # if self.innermost:
        else:
            # print(self.nz)
            # step 1: compute self attention feature map


            # step 2: add the noise plane if the layer is the innerest

            # print('not outest')
            # print(z.shape)
            z_layer = z[:, 0:self.nz]
            z = z[:, self.nz:]
            # z_layer = z

            noise = z_layer.view(z_layer.size(0), z_layer.size(1), 1, 1).expand(z_layer.size(0), z_layer.size(1), x.size(2), x.size(3))
            x_and_noise = torch.cat([x, noise], 1)

            # print('x0 size is')
            # print(x_and_noise.shape)
            # print('this is down')

            x1 = self.down(x_and_noise)
            # print('x1 size is ')
            # print(x1.shape)

            # print('this is submodule')
            # print(z.shape)
            if self.submodule is not None:
                x2 = self.submodule(x1, z)
            else:
                x2 = x1

            # print('x2 size is')
            # print(x2.shape)
            #
            # print('this is up')

            z_layer = z[:, 0:self.nz]
            z = z[:, self.nz:]

            noise = z_layer.view(z_layer.size(0), z_layer.size(1), 1, 1).expand(z_layer.size(0), z_layer.size(1), x2.size(2), x2.size(3))
            # print(noise.shape)
            x_and_noise = torch.cat([x2, noise], 1)
            # print('This is up sampling')
            out = self.up(x_and_noise)

            if self.skip:
                batch_size, channels, height, width = x.size()
                # assert channels == self.in_channels
                f = self.f(x).view(batch_size, -1, height * width).permute(0, 2, 1)      # B * (H * W) * C//8
                g = self.g(x).view(batch_size, -1, height * width)                       # B * C//8 * (H * W)

                attention = torch.bmm(f, g)                                        # B * (H * W) * (H * W)
                attention = self.softmax(attention)

                h = self.h(x).view(batch_size, channels, -1)                       # B * C * (H * W)
                #
                self_attention_map = torch.bmm(h, attention).view(batch_size, channels, height, width) # B * C * H * W

                # output = out + self_attention_map # * self.gamma
                output = torch.cat([out, self_attention_map], 1)
            else:
                # output = out + x # * self.gamma
                # if add:
                output = torch.cat([out, x], 1)

            # print(self.gamma)


            return output


# class xml(nn.Module):
#     """Defines the Unet submodule with skip connection.
#         X -------------------identity----------------------
#         |-- downsampling -- |submodule| -- upsampling --|
#     """
#
#     def __init__(self, outer_nc, inner_nc, input_nc=None,
#                  submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, skip=True, nz=0):
#         """Construct a Unet submodule with skip connections.
#
#         Parameters:
#             outer_nc (int) -- the number of filters in the outer conv layer
#             inner_nc (int) -- the number of filters in the inner conv layer
#             input_nc (int) -- the number of channels in input images/features
#             submodule (UnetSkipConnectionBlock) -- previously defined submodules
#             outermost (bool)    -- if this module is the outermost module
#             innermost (bool)    -- if this module is the innermost module
#             norm_layer          -- normalization layer
#             user_dropout (bool) -- if use dropout layers.
#         """
#         super(xml, self).__init__()
#         self.outermost = outermost
#         self.innermost = innermost
#         self.nz = nz
#
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#         if input_nc is None:
#             input_nc = outer_nc
#
#         # downconv = nn.Conv2d(input_nc + self.nz, inner_nc, kernel_size=4,
#         #                      stride=2, padding=1, bias=use_bias)
#         downrelu = nn.LeakyReLU(0.2)
#         downnorm = norm_layer(inner_nc + self.nz)
#         uprelu = nn.ReLU()
#         upnorm = norm_layer(outer_nc)
#
#
#         up = nn.Upsample(scale_factor=2, mode='bilinear')
#         uppad = nn.ReflectionPad2d(1)
#         upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=0)
#
#
#         downconv = nn.Conv2d(input_nc + self.nz, inner_nc, kernel_size=3,
#                          stride=1, padding=1, bias=use_bias)
#         down = [downrelu, downconv, downnorm]
#         up = [uprelu, up, uppad, upconv, upnorm]
#
#
#         self.down = nn.Sequential(*down)
#         self.up = nn.Sequential(*up)
#         self.submodule = submodule
#
#     def forward(self, x, z):
#         z_layer = z[:, 0:self.nz]
#         z = z[:, self.nz:]
#
#         noise = z_layer.view(z_layer.size(0), z_layer.size(1), 1, 1).expand(z_layer.size(0), z_layer.size(1), x.size(2), x.size(3))
#         x_and_noise = torch.cat([x, noise], 1)
#         # print('x0 size is')
#         # print(x_and_noise.shape)
#
#         # print('this is down')
#
#         x1 = self.down(x_and_noise)
#         # print(x1.shape)
#
#         # print('this is submodule')
#         # print(z.shape)
#         if self.submodule is not None:
#             x2 = self.submodule(x1, z)
#         else:
#             x2 = x1
#
#         # print('x2 size is')
#         # print(x2.shape)
#         #
#         # print('this is up')
#         out = self.up(x2)
#
#         output = torch.cat([x, out], 1)
#
#
#         return output


# class UnetSkipConnectionBlock(nn.Module):
#     """Defines the Unet submodule with skip connection.
#         X -------------------identity----------------------
#         |-- downsampling -- |submodule| -- upsampling --|
#     """
#
#     def __init__(self, outer_nc, inner_nc, input_nc=None,
#                  submodule=None, outermost=False, innermost=False, norm='batch', use_dropout=False, skip=True, nz=0, downsize=True):
#         """Construct a Unet submodule with skip connections.
#
#         Parameters:
#             outer_nc (int) -- the number of filters in the outer conv layer
#             inner_nc (int) -- the number of filters in the inner conv layer
#             input_nc (int) -- the number of channels in input images/features
#             submodule (UnetSkipConnectionBlock) -- previously defined submodules
#             outermost (bool)    -- if this module is the outermost module
#             innermost (bool)    -- if this module is the innermost module
#             norm_layer          -- normalization layer
#             user_dropout (bool) -- if use dropout layers.
#         """
#         super(UnetSkipConnectionBlock, self).__init__()
#         self.outermost = outermost
#         self.innermost = innermost
#         self.nz = nz
#
#         norm_layer = get_norm_layer(norm_type=norm)
#
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#         if input_nc is None:
#             input_nc = outer_nc
#
#         # downconv = nn.Conv2d(input_nc + self.nz, inner_nc, kernel_size=4,
#         #                      stride=2, padding=1, bias=use_bias)
#         downrelu = nn.LeakyReLU(0.2)
#         # downnorm = norm_layer(inner_nc + self.nz)
#         uprelu = nn.ReLU()
#         # upnorm = norm_layer(outer_nc)
#
#         if outermost:
#             downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)
#             down = [downconv]
#
#             up = nn.Upsample(scale_factor=2, mode='bilinear')
#             uppad = nn.ReflectionPad2d(1)
#             upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=0)
#             up = [uprelu, up, uppad, upconv, nn.Tanh()]
#         elif innermost:
#             downconv = nn.Conv2d(input_nc + self.nz, inner_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)
#             down = [downrelu, downconv]
#
#             up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
#             uppad = nn.ReflectionPad2d(1)
#             upconv = nn.Conv2d(inner_nc + self.nz, outer_nc, kernel_size=3, stride=1, padding=0)
#             if norm == 'spectral':
#                 # print('this is the innerest')
#                 upnorm = norm_layer(nn.Conv2d(inner_nc + self.nz, outer_nc, kernel_size=3, stride=1, padding=0))
#                 up = [uprelu, up_sample, uppad, upnorm]
#                 # print(upnorm)
#             else:
#                 # print('Don\'t use spectral normalization' )
#                 upnorm = norm_layer(outer_nc)
#                 up = [uprelu, up_sample, uppad, upconv, upnorm]
#
#             # model = down + up
#
#             # self.f = nn.Conv2d(in_channels=input_nc, out_channels=input_nc // 8, kernel_size=1)
#             # self.g = nn.Conv2d(in_channels=input_nc, out_channels=input_nc // 8, kernel_size=1)
#             # self.h = nn.Conv2d(in_channels=input_nc, out_channels=input_nc, kernel_size=1)
#             # self.softmax = nn.Softmax(dim=2)
#             # self.gamma = nn.Parameter(torch.tensor(1e-9))
#         else:
#             # print('this is the middle')
#             # print(norm)
#             if downsize:
#                 if norm == 'spectral':
#                     # print('this use spectral normaltization')
#                     downnorm = norm_layer(nn.Conv2d(input_nc + self.nz, inner_nc, kernel_size=3, stride=2, padding=1, bias=use_bias))
#                     down = [downrelu, downnorm]
#                 else:
#                     # print('Don\'t use spectral normalization' )
#                     downconv = nn.Conv2d(input_nc + self.nz, inner_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)
#                     downnorm = norm_layer(inner_nc)
#                     down = [downrelu, downconv, downnorm]
#
#
#                 up_sampling = nn.Upsample(scale_factor=2, mode='bilinear')
#                 uppad = nn.ReflectionPad2d(1)
#                 if norm == 'spectral':
#                     # print('this use spectral normaltization')
#                     upnorm = norm_layer(nn.Conv2d(inner_nc * 2 + self.nz, outer_nc, kernel_size=3, stride=1, padding=0))
#                     up = [uprelu, up_sampling, uppad, upnorm]
#                 else:
#                     # print('Don\'t use spectral normalization' )
#                     upconv = nn.Conv2d(inner_nc * 2 + self.nz, outer_nc, kernel_size=3, stride=1, padding=0)
#                     upnorm = norm_layer(outer_nc)
#                     up = [uprelu, up_sampling, uppad, upconv, upnorm]
#             else:
#                 if norm == 'spectral':
#                     # print('this use spectral normaltization')
#                     # print('this use spectral normaltization')
#                     downnorm = norm_layer(nn.Conv2d(input_nc + self.nz, inner_nc, kernel_size=3, stride=1, padding=1, bias=use_bias))
#                     down = [downrelu, downnorm]
#                 else:
#                     # print('Don\'t use spectral normalization' )
#                     downconv = nn.Conv2d(input_nc + self.nz, inner_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
#                     downnorm = norm_layer(inner_nc)
#                     down = [downrelu, downconv, downnorm]
#
#                 if norm == 'spectral':
#                     upnorm = norm_layer(nn.Conv2d(inner_nc * 2 + self.nz, outer_nc, kernel_size=3, stride=1, padding=1))
#                     up = [uprelu, upnorm]
#                 else:
#                     # print('Don\'t use spectral normalization' )
#                     upconv = nn.Conv2d(inner_nc * 2 + self.nz, outer_nc, kernel_size=3, stride=1, padding=1)
#                     upnorm = norm_layer(outer_nc)
#                     up = [uprelu, upconv, upnorm]
#
#
#         self.down = nn.Sequential(*down)
#         self.up = nn.Sequential(*up)
#         self.submodule = submodule
#
#
#     def forward(self, x, z):
#         if self.outermost:
#             # print('this is outermost')
#             # print(x.shape)
#             # print('This is a new beginning')
#
#             x1 = self.down(x)
#             # print('----------------------------------')
#             # print(x1.shape)
#
#             x2 = self.submodule(x1, z)
#             output = self.up(x2)
#
#             # print('after up')
#             # print(z.shape)
#             #
#             # print('this is a end of one iteration')
#             # print(output.shape)
#             return output
#         # if self.innermost:
#         else:
#             # print(self.nz)
#             # step 1: compute self attention feature map
#             # batch_size, channels, height, width = x.size()
#             # # assert channels == self.in_channels
#             # f = self.f(x).view(batch_size, -1, height * width).permute(0, 2, 1)      # B * (H * W) * C//8
#             # g = self.g(x).view(batch_size, -1, height * width)                       # B * C//8 * (H * W)
#             #
#             # attention = torch.bmm(f, g)                                        # B * (H * W) * (H * W)
#             # attention = self.softmax(attention)
#             #
#             # h = self.h(x).view(batch_size, channels, -1)                       # B * C * (H * W)
#             #
#             # self_attention_map = torch.bmm(h, attention).view(batch_size, channels, height, width) # B * C * H * W
#
#             # step 2: add the noise plane if the layer is the innerest
#
#             # print('not outest')
#             # print(z.shape)
#             z_layer = z[:, 0:self.nz]
#             z = z[:, self.nz:]
#
#             noise = z_layer.view(z_layer.size(0), z_layer.size(1), 1, 1).expand(z_layer.size(0), z_layer.size(1), x.size(2), x.size(3))
#             x_and_noise = torch.cat([x, noise], 1)
#
#             # print('x0 size is')
#             # print(x_and_noise.shape)
#             # print('this is down')
#
#             x1 = self.down(x_and_noise)
#             # print('x1 size is ')
#             # print(x1.shape)
#
#             # print('this is submodule')
#             # print(z.shape)
#             if self.submodule is not None:
#                 x2 = self.submodule(x1, z)
#             else:
#                 x2 = x1
#
#             # print('x2 size is')
#             # print(x2.shape)
#             #
#             # print('this is up')
#
#             z_layer = z[:, 0:self.nz]
#             z = z[:, self.nz:]
#
#             noise = z_layer.view(z_layer.size(0), z_layer.size(1), 1, 1).expand(z_layer.size(0), z_layer.size(1), x2.size(2), x2.size(3))
#             x_and_noise = torch.cat([x2, noise], 1)
#             # print('This is up sampling')
#             out = self.up(x_and_noise)
#
#             output = torch.cat([x, out], 1)
#
#
#             return output



# class UnetSkipConnectionBlock(nn.Module):
#     """Defines the Unet submodule with skip connection.
#         X -------------------identity----------------------
#         |-- downsampling -- |submodule| -- upsampling --|
#     """
#
#     def __init__(self, outer_nc, inner_nc, input_nc=None,
#                  submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, skip=True, nz=0):
#         """Construct a Unet submodule with skip connections.
#
#         Parameters:
#             outer_nc (int) -- the number of filters in the outer conv layer
#             inner_nc (int) -- the number of filters in the inner conv layer
#             input_nc (int) -- the number of channels in input images/features
#             submodule (UnetSkipConnectionBlock) -- previously defined submodules
#             outermost (bool)    -- if this module is the outermost module
#             innermost (bool)    -- if this module is the innermost module
#             norm_layer          -- normalization layer
#             user_dropout (bool) -- if use dropout layers.
#         """
#         super(UnetSkipConnectionBlock, self).__init__()
#         self.outermost = outermost
#         self.innermost = innermost
#         self.nz = nz
#
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#         if input_nc is None:
#             input_nc = outer_nc
#
#         downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
#                              stride=2, padding=1, bias=use_bias)
#         downrelu = nn.LeakyReLU(0.2)
#         downnorm = norm_layer(inner_nc)
#         uprelu = nn.ReLU()
#         upnorm = norm_layer(outer_nc)
#
#         if outermost:
#             upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1)
#             down = [downconv]
#             up = [uprelu, upconv, nn.Tanh()]
#             # model = down + [submodule] + up
#             # self.model = nn.Sequential(*model)
#         elif innermost:
#             upconv = nn.ConvTranspose2d(inner_nc + self.nz, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downrelu, downconv]
#             up = [uprelu, upconv, upnorm]
#
#             # model = down + up
#
#             self.f = nn.Conv2d(in_channels=input_nc, out_channels=input_nc // 8, kernel_size=1)
#             self.g = nn.Conv2d(in_channels=input_nc, out_channels=input_nc // 8, kernel_size=1)
#             self.h = nn.Conv2d(in_channels=input_nc, out_channels=input_nc, kernel_size=1)
#             self.softmax = nn.Softmax(dim=2)
#             self.gamma = nn.Parameter(torch.tensor(1e-9))
#         else:
#             upconv = nn.ConvTranspose2d(inner_nc * 2 + self.nz, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             # downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
#                              # stride=2, padding=1, bias=use_bias)
#             down = [downrelu, downconv, downnorm]
#             up = [uprelu, upconv, upnorm]
#
#             # if use_dropout:
#             #     model = down + [submodule] + up + [nn.Dropout(0.5)]
#             # else:
#             #     model = down + [submodule] + up
#
#         # self.beta = nn.Parameter(torch.tensor(0.0))
#
#             # self.model = nn.Sequential(*model)
#
#         self.down = nn.Sequential(*down)
#         self.up = nn.Sequential(*up)
#         self.submodule = submodule
#
#
#     def forward(self, x, z):
#         if self.outermost:
#             x1 = self.down(x)
#             x2 = self.submodule(x1, z)
#             output = self.up(x2)
#
#             # print('')
#             # print('this is outermost')
#             # print(x.shape)
#             # print(output.shape)
#             return output
#         # if self.innermost:
#         else:
#             # print(self.nz)
#             # step 1: compute self attention feature map
#             # batch_size, channels, height, width = x.size()
#             # # assert channels == self.in_channels
#             # f = self.f(x).view(batch_size, -1, height * width).permute(0, 2, 1)      # B * (H * W) * C//8
#             # g = self.g(x).view(batch_size, -1, height * width)                       # B * C//8 * (H * W)
#             #
#             # attention = torch.bmm(f, g)                                        # B * (H * W) * (H * W)
#             # attention = self.softmax(attention)
#             #
#             # h = self.h(x).view(batch_size, channels, -1)                       # B * C * (H * W)
#             #
#             # self_attention_map = torch.bmm(h, attention).view(batch_size, channels, height, width) # B * C * H * W
#
#             # step 2: add the noise plane if the layer is the innerest
#             x1 = self.down(x)
#             if self.submodule is None:
#                 feature_map = x1
#                 # print('')
#                 # print('this is innerest')
#             else:
#                 # print('')
#                 # print('this is outside')
#                 feature_map = self.submodule(x1, z)
#
#             z_plane = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), feature_map.size(2), feature_map.size(3))
#
#             x_and_z_plane = torch.cat([feature_map, z_plane], 1)
#
#             # print(z_plane)
#             # print(feature_map)
#
#             # print('this is a debug {}'.format(x_and_z_plane))
#
#             # print(x.shape)
#             # print(feature_map.shape)
#             # # print(x_and_z_plane)
#             # print(x_and_z_plane.shape)
#
#             out = self.up(x_and_z_plane)
#
#
#             # print('gamma is {:0.9f}'.format(self.gamma))
#             # print(z_plane.shape)
#             # print(feature_map.shape)
#             # print(self_attention_map.shape)
#             # print(out.shape)
#
#             # step 3: combine the attention and encoding results with noise layer
#             # output = torch.cat([self.gamma * x, self.model(x)], 1)
#
#             # output = torch.cat([self.gamma * self_attention_map, out], 1)
#             output = torch.cat([x, out], 1)
#             # print('output\'s size is %d %d %d %d' % (output.shape[0], output.shape[1], output.shape[2], output.shape[3]))
#
#             return output
#         # else:
#         #     # return torch.cat([self.beta * x, self.model(x)], 1)
#         #     # y = self.beta * x
#         #     x1 = self.down(x)
#         #     x2 = self.submodule(x1, z)
#         #     x3 = self.up(x2)
#         #     output = torch.cat([x, x3], 1)
#         #     return output



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm='batch', nz_D=4):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.nz_D = nz_D

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1



        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            conv = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)


            if norm == 'spectral':
                sequence += [norm_layer(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))]
            else:
                sequence += [conv]
                sequence += [norm_layer(ndf * nf_mult)]

            sequence += [nn.LeakyReLU(0.2)]


        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        conv = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)


        if norm == 'spectral':
            sequence += [norm_layer(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias))]
        else:
            sequence += [conv]
            sequence += [norm_layer(ndf * nf_mult)]

        # sequence += [nn.LeakyReLU(0.2)]

        # sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

        # sequence += [nn.AdaptiveAvgPool2d(1)]

        self.model = nn.Sequential(*sequence)

        self.linear = nn.Linear(ndf * nf_mult, 1)

        self.embed_noise = nn.utils.spectral_norm(nn.Linear(self.nz_D, ndf * nf_mult))

        print('There are {:d} channels in Discriminator'.format(ndf * nf_mult))

    def forward(self, input, noise):
        """Standard forward."""
        out = self.model(input)

        # noise = noise.view(noise.size(0), noise.size(1), 1, 1).expand(noise.size(0), noise.size(1), input.size(2), input.size(3))
        out = torch.nn.functional.relu(out).view(out.size(0), out.size(1), -1).sum(2) # B * C
        out_linear = self.linear(out) # B * 1

        noise = self.embed_noise(noise) # B * nz ==> B * C (specific)
        label_noise = (noise * out).sum(1) # B * 1, noise shoule be B * C

        output = out_linear + label_noise # B * 1
        # print('this is D')
        # print(output.shape)
        return output


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
