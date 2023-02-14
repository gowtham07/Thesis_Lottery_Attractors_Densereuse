import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
#from train_file import save_alphas_for_all_layers, save_bns_for_bias

global countsss
global countsss_conv
countsss = -1
countsss_conv = -1
 

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(Linear, self).__init__(in_features, out_features, bias)        
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))
        
    def forward(self, input):
        
        W = self.weight_mask * self.weight
        if self.bias is not None:
            b = self.bias_mask * self.bias
        else:
            b = self.bias
        return F.linear(input, W, b)

class Linear_1(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_1, self).__init__(in_features, out_features, bias)        
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))
     
    def forward(self,input,save_alphas_for_all_layers,save_bns_for_bias,st):
        if st== True:
            if type(save_alphas_for_all_layers) == list:
                global countsss
                if countsss == 2: #change this as per linear layers
                    countsss = -1
                countsss = countsss + 1
                if countsss == 0:
                #  d = save_alphas_for_all_layers[2*countsss].shape
                #  w = save_alphas_for_all_layers[2*countsss].view(d[0],d[1],1,1)*self.weight.view(1024,40,7,7)
                #  w = w.view(1024,40*7*7)
                 a = save_alphas_for_all_layers[2*countsss].repeat(1,49)
                 w = a * self.weight
                 W = self.weight_mask * w
                else:
                 w = save_alphas_for_all_layers[2*countsss]*self.weight
                 W = self.weight_mask * w

                if  countsss <2: 
                    if self.bias is not None:
                        a = 1/(save_alphas_for_all_layers[2*countsss+1])
                        bi = save_alphas_for_all_layers[2*countsss+1]*(self.bias- save_bns_for_bias[2*countsss] + a*save_bns_for_bias[2*countsss+1])
                        b = self.bias_mask * bi
                    else:
                        b = self.bias
                else:
                    if self.bias is not None:
                        b = self.bias_mask * self.bias
                    else:
                        b = self.bias
            else:
                W = self.weight_mask * self.weight
                if self.bias is not None:
                 b = self.bias_mask * self.bias
                else:
                 b = self.bias
        else:   
            W = self.weight_mask * self.weight
            if self.bias is not None:
                b = self.bias_mask * self.bias
            else:
             b = self.bias    
                    
        return F.linear(input, W, b)        
   

# class Linear(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True):
#         super(Linear, self).__init__(in_features, out_features, bias)        
#         self.register_buffer('weight_mask', torch.ones(self.weight.shape))
#         if self.bias is not None:
#           self.register_buffer('bias_mask', torch.ones(self.bias.shape))

#     def forward(self, input):
#         W = self.weight
#         if self.bias is not None:
#             b = self.bias
#         else:
#             b = self.bias
#         return F.linear(input, W, b)        


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=False, padding_mode='zeros'):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode)
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self,input):
        W = self.weight_mask * self.weight
        if self.bias is not None:
            b = self.bias_mask * self.bias
        else:
            b = self.bias
        return self._conv_forward(input, W, b)

class Conv2d_1(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2d_1, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode)
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self,input,save_alphas_for_all_layers,save_bns_for_bias,st):
        # W = self.weight_mask * self.weight
        # if self.bias is not None:
        #     b = self.bias_mask * self.bias
        # else:
        #     b = self.bias
        # return self._conv_forward(input, W, b)
        if st== True:
            if type(save_alphas_for_all_layers) == list:
                global countsss_conv
                if countsss_conv == 3: #change this as per linear layers
                    countsss_conv = -1
                countsss_conv = countsss_conv + 1
                if countsss_conv == 0:
                 d = save_alphas_for_all_layers[2*countsss_conv].shape[0]
                 w = save_alphas_for_all_layers[2*countsss_conv].view(d,1,1,1)*self.weight
                 W = self.weight_mask * w
                else:
                 d = save_alphas_for_all_layers[2*countsss_conv].shape   
                 w = save_alphas_for_all_layers[2*countsss_conv].view(d[0],d[1],1,1)*self.weight
                 W = self.weight_mask * w

                if  countsss_conv <4: 
                    if self.bias is not None:
                        a = 1/(save_alphas_for_all_layers[2*countsss_conv+1])
                        bi = save_alphas_for_all_layers[2*countsss_conv+1]*((self.bias- save_bns_for_bias[2*countsss_conv]) + a*save_bns_for_bias[2*countsss_conv+1])
                        b = self.bias_mask * bi
                    else:
                        b = self.bias
                else:
                    if self.bias is not None:
                        b = self.bias_mask * self.bias
                    else:
                        b = self.bias
            else:
                W = self.weight_mask * self.weight
                if self.bias is not None:
                 b = self.bias_mask * self.bias
                else:
                 b = self.bias
        else:   
            W = self.weight_mask * self.weight
            if self.bias is not None:
                b = self.bias_mask * self.bias
            else:
             b = self.bias    
                    
        return self._conv_forward(input, W, b) 


class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        if self.affine:     
          self.register_buffer('weight_mask', torch.ones(self.weight.shape))
          self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.affine:
            W = self.weight_mask * self.weight
            b = self.bias_mask * self.bias
            # W = self.weight
            # b = self.bias
        else:
            W = self.weight
            b = self.bias

        return F.batch_norm(
            input, self.running_mean, self.running_var, W, b,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        if self.affine:     
            self.register_buffer('weight_mask', torch.ones(self.weight.shape))
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.affine:
            W = self.weight_mask * self.weight
            b = self.bias_mask * self.bias
        else:
            W = self.weight
            b = self.bias

        return F.batch_norm(
            input, self.running_mean, self.running_var, W, b,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class Identity1d(nn.Module):
    def __init__(self, num_features):
        super(Identity1d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W


class Identity2d(nn.Module):
    def __init__(self, num_features):
        super(Identity2d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features, 1, 1))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W



