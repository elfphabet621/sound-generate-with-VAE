import torch
import torch.nn as nn
import os

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
class BottleNeck(nn.Module):
    def __init__(self, new_shape, latent_space_dim):
        super(BottleNeck, self).__init__()
        self.reshape_layer = Reshape(*(-1,new_shape))
        input_shape = 16*32*32
        self.mu = nn.Linear(input_shape, latent_space_dim)
        self.log_variance = nn.Linear(input_shape, latent_space_dim)    

    def sample_point_from_normal_distribution(self, mu, log_variance):
        device = mu.device.type
        epsilon = torch.normal(mean=0, std=1, size=mu.shape)
        sampled_point = mu + torch.exp(log_variance / 2) * epsilon.to(device)
        return sampled_point

    def forward(self, x):
        x = self.reshape_layer(x)
        mu = self.mu(x)
        log_variance = self.log_variance(x)
        sampled_point = self.sample_point_from_normal_distribution(mu, log_variance)
        return sampled_point, mu, log_variance
    
class ReverseBottleNeck(nn.Module):
    def __init__(self, new_shape, input_dim, latent_space_dim):
        super(ReverseBottleNeck, self).__init__()
        self.reverse_bottle_neck = nn.Sequential(nn.Linear(input_dim, latent_space_dim),
                Reshape(*new_shape))
        
    def forward(self, x):
        return self.reverse_bottle_neck(x)
    
class ConvLayer(nn.Module):
    def __init__(self, in_channels, conv_filters, conv_kernels, conv_strides, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels = in_channels, 
                        out_channels = conv_filters,
                        kernel_size = conv_kernels, 
                        stride = conv_strides, 
                        padding= padding)
        
        self.relu = nn.ReLU()

        self.batchnorm = nn.BatchNorm2d(conv_filters)
    
    def forward(self, x):
        return self.batchnorm(self.relu(self.conv2d(x)))
    
class ConvTranspose2dLayer(nn.Module):
    def __init__(self, in_channels, conv_filters, conv_kernels, conv_strides, padding, output_padding):
        super(ConvTranspose2dLayer, self).__init__()

        self.conv_transpose_layers = nn.ConvTranspose2d(in_channels= in_channels, 
                                            out_channels = conv_filters,
                                            kernel_size = conv_kernels, 
                                            stride = conv_strides, 
                                            padding=padding,
                                            output_padding=output_padding)
        
        self.relu = nn.ReLU()

        self.batchnorm = nn.BatchNorm2d(conv_filters)
    
    def forward(self, x):
        return self.batchnorm(self.relu(self.conv_transpose_layers(x)))
    
class VAE(nn.Module):
    """Autoencoder architecture include: Encoder and Decoder
    """
    def __init__(self, conv_filters, 
                 conv_kernels, conv_strides, latent_space_dim):
        
        super().__init__()
        self.conv_filters  = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.num_layers = len(conv_filters)
        self.sigmoid = nn.Sigmoid()

        self.conv_layers = nn.Sequential(*[ConvLayer(1 if i == 0 else self.conv_filters[i-1],
                                      self.conv_filters[i],
                                      self.conv_kernels[i],
                                      self.conv_strides[i],
                                      1)
                                for i in range(self.num_layers)])

        self.bottleneck = BottleNeck(new_shape=16*32*32,
                                latent_space_dim=self.latent_space_dim)

        self.reverse_bottleneck = ReverseBottleNeck(new_shape=(-1, 16, 32, 32), 
                                             input_dim=self.latent_space_dim, 
                                             latent_space_dim=16*32*32)
        
        self.conv_transpose_layers = nn.Sequential(*[ConvTranspose2dLayer(self.conv_filters[i], 
                                        1 if i==0 else self.conv_filters[i-1],
                                        self.conv_kernels[i], 
                                        self.conv_strides[i], 
                                        1,
                                        (1,0) if i == self.num_layers-1 else 1) 
                                for i in reversed(range(self.num_layers))])              

    def forward(self, x):
        x = self.conv_layers(x)
        x, mu, log_variance  = self.bottleneck(x)

        x = self.reverse_bottleneck(x)
        x = self.conv_transpose_layers(x)
        x = self.sigmoid(x)
        return x, mu, log_variance