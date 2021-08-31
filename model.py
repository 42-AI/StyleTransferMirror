import torch.nn as nn
import torch

from utils import normalize
from utils import mean_std

# Output the image from the content and style features
# passed through the attention layers
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

# The feature extractor network
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

# This is a Self Atention Network where the content image features (Fc)
# and the style image features (Fs) are match with the attention mechanism 
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

        # The permute transpose the tensor along the two last axis
        # F_Fc_norm is similar to the 'key' if you are familiar with attention mechanism
        F_Fc_norm = self.f(normalize(Fc)).view(B, -1, H * W).permute(0, 2, 1)
        B, _, H, W = Fs.size()

        # This is similar to the 'query'
        G_Fs_norm = self.g(normalize(Fs)).view(B, -1, H * W)

        # The attention mechanism
        attention = self.softmax(torch.bmm(F_Fc_norm, G_Fs_norm))

        # This is similar to the 'value'
        H_Fs = self.h(Fs).view(B, -1, H * W)

        # Finally this is the output calculation
        out = torch.bmm(H_Fs, attention.permute(0, 2, 1))

        B, C, H, W = Fc.size()

        # Reshape to the feature shapes 
        out = out.view(B, C, H, W)
        out = self.out_conv(out)

        # Skip connection with the content features
        out += Fc

        return out

# Combine the SANets
class SelfAttentionModule(nn.Module):
    def __init__(self, in_channel : int):
        super(SelfAttentionModule, self).__init__()

        # two SANets for the relu5 and relu4 layers
        self.SAN1 = SANet(in_channel)
        self.SAN2 = SANet(in_channel)

        # Other layers for combining
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') 
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_channel, in_channel, (3, 3))

    def forward(self, Fc : torch.Tensor, Fs : torch.Tensor):
        # First, the attentions on the last features (relu5 output)
        Fcsc_5 = self.SAN1(Fc[-1], Fs[-1])

        # Uspsampling to match the Fcsc_4 shape
        Fcsc_5_up = self.upsample(Fcsc_5)
        
        # Then, the attentions on the relu4 output
        Fcsc_4 = self.SAN2(Fc[-2], Fs[-2])
        
        # Finaly combination and convolution of both SANets
        Fcsc_m = Fcsc_4 + Fcsc_5_up
        Fcsc_m = self.merge_conv_pad(Fcsc_m)
        Fcsc_m = self.merge_conv(Fcsc_m)
        
        return Fcsc_m

# Comput the output images and the losses
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
            # Get the enc_i layers from self
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            
            # Propagate previous output through enc_i layers
            results.append(func(results[-1]))
        # extract [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1] from x
        return results[1:]
    
    # It is just a mse loss with safe garde
    def content_loss(self, _input : torch.Tensor, _target : torch.Tensor):
        assert (_input.size() == _target.size())
        assert (_target.requires_grad is False)
        return self.mse_loss(_input, _target)
    
    # Mse loss of the mean and standard deviation of the input and target
    def style_loss(self, _input : torch.Tensor, _target : torch.Tensor):
        assert (_input.size() == _target.size())
        assert (_target.requires_grad is False)
        _input_mean, _input_std = mean_std(_input)
        _target_mean, _target_std = mean_std(_target)
        return self.mse_loss(_input_mean, _target_mean) \
             + self.mse_loss(_input_std, _target_std)

    # If training, compute the losses, 
    # else, only the output styled images 
    def forward(self, Ic : torch.Tensor, Is : torch.Tensor, train : bool = True):
        # Extract features from the style and the content image
        Fs = self.get_encoder_features(Is)
        Fc = self.get_encoder_features(Ic)

        # The styled images
        Ics = self.decoder(self.sa_module(Fc, Fs))

        # If we are not training, stop here
        if not train:
            return Ics

        # Extract the features from the stylized images
        Ics_feats = self.get_encoder_features(Ics)

        # Content loss
        # We only use relu4 and relu5 output because 
        # this is where there is the most content value
        Lc = self.content_loss(normalize(Ics_feats[-1]), normalize(Fc[-1])) \
           + self.content_loss(normalize(Ics_feats[-2]), normalize(Fc[-2]))
        
        # Style loss
        # Here we calculate the style loss from all the relu layers because
        # the style can be extract from all the layers 
        # (even there is more style value on the first layers)
        Ls = sum([self.style_loss(Ics_feats[i], Fs[i]) for i in range(5)])
        # For the result yield by the save 159000.pt i have 
        # finetune for 50000 steps with above line replace by :
        # Ls = sum([self.style_loss(Ics_feats[i], Fs[i]) for i in range(3)])
        # By remvoing the last layers of the style images, we remove the 
        # style image content leakage on the transfered image.
        # I also added 3 to the style_weight on the config.json to 
        # put more value on the style because the sum will be smaller

        # Then the styled images with both same style and both same content
        Icc = self.decoder(self.sa_module(Fc, Fc))
        Iss = self.decoder(self.sa_module(Fs, Fs))

        # Extracting the features
        Icc_feats = self.get_encoder_features(Icc)
        Iss_feats = self.get_encoder_features(Iss)

        # Those two loss value are the inovation of this paper.
        # This is used to check that stylized an image with itself 
        # have the same features as the initial images

        # identity1 loss
        loss_lambda1 = self.content_loss(Icc, Ic) + self.content_loss(Iss, Is)

        # identity2 loss
        loss_lambda2 = sum([self.content_loss(Icc_feats[i], Fc[i]) + self.content_loss(Iss_feats[i], Fs[i]) for i in range(5)])

        return Lc, Ls, loss_lambda1, loss_lambda2