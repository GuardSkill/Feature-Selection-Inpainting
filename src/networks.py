import torch
import torch.nn as nn
from libs.Modules import PartialModule, GatedModule, DeConvGatedModule, PartialResnetBlock, CFSModule, ResCFSBlock
import torch.nn.functional as F


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        # def init_weights(self, init_type='xavier', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0
        /models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x


class InpaintGeneratorGated(BaseNetwork):
    def __init__(self, residual_blocks=4, init_weights=True, upsampling_mode='nearest'):
        super(InpaintGeneratorGated, self).__init__()
        self.upsampling_mode = upsampling_mode

        self.enc_1 = CFSModule(in_channels=5, out_channels=64, kernel_size=7, padding=0)
        self.enc_2 = CFSModule(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=2)
        self.enc_3 = CFSModule(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.enc_4 = CFSModule(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)

        blocks = []
        for _ in range(residual_blocks):
            block = ResCFSBlock(512, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.dec_4 = CFSModule(in_channels=512 + 256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.dec_3 = CFSModule(in_channels=256 + 128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.dec_2 = CFSModule(in_channels=128 + 64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.dec_1 = CFSModule(in_channels=64 + 5, out_channels=3, kernel_size=3, padding=1, stride=1,
                               activ=None, instance_norm=False)

        if init_weights:
            self.init_weights(init_type='kaiming')

    def forward(self, x):
        h_dict = {}  # for the output of enc_N
        h_scale_dict = {}  # for recording the scale factor in convolution

        h_dict['h_0'] = x

        h_key_prev = 'h_0'
        for i in range(1, 5):  # from 1 to layer_size
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])
            # Propagate the feature meanwhile getting the feature map and mask map in encoder
            h_scale_dict[h_key] = (
                h_dict[h_key_prev].shape[2] / h_dict[h_key].shape[2],
                h_dict[h_key_prev].shape[3] / h_dict[h_key].shape[3])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(4)
        h = h_dict[h_key]  # last output in encoder
        h = self.middle(h)  # propagate the middle residual blocks

        for i in range(4, 0, -1):  # from layer_size to 1
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)
            sca_h_key = 'h_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=h_scale_dict[sca_h_key], mode=self.upsampling_mode)  # upsampling
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h = getattr(self, dec_l_key)(h)
        h = (torch.tanh(h) + 1) / 2
        return h


class EdgeGeneratorUnet(BaseNetwork):
    def __init__(self, residual_blocks=4, init_weights=True, input_channels=2, upsampling_mode='nearest'):
        super(EdgeGeneratorUnet, self).__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode

        self.enc_1 = PartialModule(in_channels=input_channels, out_channels=64, kernel_size=7, padding=0)
        # I = I-6 256-6=250
        self.enc_2 = PartialModule(in_channels=64, out_channels=128, kernel_size=6, padding=2, stride=2)
        # I = I/2  125
        self.enc_3 = PartialModule(in_channels=128, out_channels=256, kernel_size=4, padding=1, stride=2)
        # I = I/2  62
        self.enc_4 = PartialModule(in_channels=256, out_channels=512, kernel_size=4, padding=1, stride=2)
        # I = I/2  31
        blocks = []
        for _ in range(residual_blocks):
            block = PartialResnetBlock(512, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.dec_4 = PartialModule(in_channels=512 + 256, out_channels=256, kernel_size=3, padding=1, stride=1)

        self.dec_3 = PartialModule(in_channels=256 + 128, out_channels=128, kernel_size=3, padding=1, stride=1)

        self.dec_2 = PartialModule(in_channels=128 + 64, out_channels=64, kernel_size=3, padding=1, stride=1)

        self.dec_1 = PartialModule(in_channels=64 + input_channels, out_channels=1, kernel_size=3, padding=1, stride=1,
                                   activ=None, instance_norm=False)
        if init_weights:
            self.init_weights(init_type='kaiming')

    def forward(self, x, mask):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N
        h_scale_dict = {}  # for recording the scale factor in convolution

        h_dict['h_0'], h_mask_dict['h_0'] = x, mask

        h_key_prev = 'h_0'
        for i in range(1, 5):  # from 1 to layer_size
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev], h_mask_dict[h_key_prev])
            # Propagate the feature meanwhile getting the feature map and mask map in encoder
            h_scale_dict[h_key] = (
                h_dict[h_key_prev].shape[2] / h_dict[h_key].shape[2],
                h_dict[h_key_prev].shape[3] / h_dict[h_key].shape[3])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(4)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]  # last output in encoder
        [h, h_mask] = self.middle([h, h_mask])  # propagate the middle residual blocks

        for i in range(4, 0, -1):  # from layer_size to 1
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)
            sca_h_key = 'h_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=h_scale_dict[sca_h_key], mode=self.upsampling_mode)  # upsampling
            h_mask = F.interpolate(
                h_mask, scale_factor=h_scale_dict[sca_h_key], mode='nearest')

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)
        h = torch.sigmoid(h)
        return h


class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=4, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class DiscriminatorEnhanced(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )
        self.conv6 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6=self.conv6(conv5)
        outputs = conv6
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5,conv6]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
