from torch.functional import norm
import torch.nn as nn
import torch
from torch.nn.modules import loss
import torchvision
from torchvision.transforms.functional import invert

from utils import normalize
from utils import mean_std

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class SANet(nn.Module):
    def __init__(self, in_channel: int):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_channel, in_channel, (1, 1))
        self.g = nn.Conv2d(in_channel, in_channel, (1, 1))
        self.h = nn.Conv2d(in_channel, in_channel, (1, 1))
        self.softmax = nn.Softmax(-1)
        self.out_conv = nn.Conv2d(in_channel, in_channel, (1, 1))
    
    def forward(self, Fc : torch.Tensor, Fs : torch.Tensor):
        B, _, H, W = Fc.size()
        # the permute transpose the tensor along the two last axis
        F_Fc_norm = self.f(normalize(Fc)).view(B, -1, H * W).permute(0, 2, 1)
        B, _, H, W = Fs.size()
        G_Fs_norm = self.g(normalize(Fs)).view(B, -1, H * W)
        attention = self.softmax(torch.bmm(F_Fc_norm, G_Fs_norm))
        H_Fs = self.h(Fs).view(B, -1, H * W)
        out = torch.bmm(H_Fs, attention.permute(0, 2, 1))
        B, C, H, W = Fc.size()
        out = out.view(B, C, H, W)
        out = self.out_conv(out)
        out += Fc
        return out

class SelfAttentionModule(nn.Module):
    def __init__(self, in_channel : int):
        super(SelfAttentionModule, self).__init__()

        self.SAN1 = SANet(in_channel)
        self.SAN2 = SANet(in_channel)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') 
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_channel, in_channel, (3, 3))

    def forward(self, Fc : torch.Tensor, Fs : torch.Tensor):
        Fcsc_5 = self.SAN1(Fc[-1], Fs[-1])
        Fcsc_5_up = self.upsample(Fcsc_5)
        Fcsc_4 = self.SAN2(Fc[-2], Fs[-2])
        Fcsc_m = Fcsc_4 + Fcsc_5_up
        Fcsc_m = self.merge_conv_pad(Fcsc_m)
        Fcsc_m = self.merge_conv(Fcsc_m)
        
        return Fcsc_m

class MultiLevelStyleAttention(nn.Module):
    def __init__(self, encoder : nn.Sequential, decoder : nn.Sequential):
        super(MultiLevelStyleAttention, self).__init__()

        # Get the encoder layers
        encoder_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*encoder_layers[:4])
        self.enc_2 = nn.Sequential(*encoder_layers[4:11])
        self.enc_3 = nn.Sequential(*encoder_layers[11:18])
        self.enc_4 = nn.Sequential(*encoder_layers[18:31])
        self.enc_5 = nn.Sequential(*encoder_layers[31:44])

        # Transforms
        self.sa_module = SelfAttentionModule(512)
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # Fix the encoder parameters
        for n in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, n).parameters():
                param.requires_grad = False

    def get_encoder_features(self, x : torch.Tensor):
        '''
        x : batch of images
        '''
        results = [x]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        # extract [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1] from x
        return results[1:]
    
    def content_loss(self, _input : torch.Tensor, _target : torch.Tensor):
        assert (_input.size() == _target.size())
        assert (_target.requires_grad is False)
        return self.mse_loss(_input, _target)
    
    def style_loss(self, _input : torch.Tensor, _target : torch.Tensor):
        assert (_input.size() == _target.size())
        assert (_target.requires_grad is False)
        _input_mean, _input_std = mean_std(_input)
        _target_mean, _target_std = mean_std(_target)
        return self.mse_loss(_input_mean, _target_mean) \
             + self.mse_loss(_input_std, _target_std)

    def forward(self, Ic : torch.Tensor, Is : torch.Tensor, train : bool = True):
        Fs = self.get_encoder_features(Is)
        Fc = self.get_encoder_features(Ic)

        Ics = self.decoder(self.sa_module(Fc, Fs))

        if not train:
            return Ics

        Ics_feats = self.get_encoder_features(Ics)

        # Content loss
        Lc = self.content_loss(normalize(Ics_feats[-1]), normalize(Fc[-1])) \
           + self.content_loss(normalize(Ics_feats[-2]), normalize(Fc[-2]))
        
        # Style loss
        Ls = sum([self.style_loss(Ics_feats[i], Fs[i]) for i in range(5)])

        Icc = self.decoder(self.sa_module(Fc, Fc))
        Iss = self.decoder(self.sa_module(Fs, Fs))

        Icc_feats = self.get_encoder_features(Icc)
        Iss_feats = self.get_encoder_features(Iss)

        # identity1 loss
        loss_lambda1 = self.content_loss(Icc, Ic) + self.content_loss(Iss, Is)

        # identity2 loss
        loss_lambda2 = sum([self.content_loss(Icc_feats[i], Fc[i]) + self.content_loss(Iss_feats[i], Fs[i]) for i in range(5)])

        return Lc, Ls, loss_lambda1, loss_lambda2