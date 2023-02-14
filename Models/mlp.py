import torch
import torch.nn as nn
import numpy as np
from Layers import layers
from torch.nn import functional as F
from collections import OrderedDict


class mlp_bn(nn.Module):

    def __init__(self, input_shape, num_classes, dense_classifier,pretrained,nonlinearity=nn.ReLU()):
        super().__init__()
        self.size = np.prod(input_shape)
        self.num_classes = num_classes
        self.dense_classifier = dense_classifier
        self.nonlinearity = nonlinearity
       
        if self.dense_classifier == False:

          self.flatten = nn.Flatten()
          self.linear_1 = layers.Linear_1(self.size,64)
          self.linearity_1 = self.nonlinearity
          self.linear_2 = layers.Linear_1(64,32)
          self.linearity_2 = self.nonlinearity
          self.linear_3 = layers.Linear_1(32,10)
        else:
          self.flatten = nn.Flatten()
          self.linear_1 = layers.Linear_1(self.size,64)
          self.linearity_1 = self.nonlinearity
          self.linear_2 = layers.Linear_1(64,32)
          self.linearity_2 = self.nonlinearity
          self.linear_3 = nn.Linear(32,10)
          

    def forward(self, x,y,z,st):
        
        output = self.flatten(x)
        output = self.linear_1(output,y,z,st)
        output = self.linearity_1(output)
        output = self.linear_2(output,y,z,st)
        output = self.linearity_2(output)
        output = self.linear_3(output,y,z,st)
        
        return output

    

def fc(input_shape, num_classes, dense_classifier=False, pretrained=False, L=6, N=100, nonlinearity=nn.ReLU()):
  size = np.prod(input_shape)
  
  # Linear feature extractor
  modules = [nn.Flatten()]
  modules.append(layers.Linear(size, N))
  modules.append(nonlinearity)
  for i in range(L-2):
    modules.append(layers.Linear(N,N))
    modules.append(nonlinearity)

  # Linear classifier
  if dense_classifier:
    modules.append(nn.Linear(N, num_classes))
  else:
    modules.append(layers.Linear(N, num_classes))
  model = nn.Sequential(*modules)

  # Pretrained model
  if pretrained:
    print("WARNING: this model does not have pretrained weights.") 
  
  return model


def conv(input_shape, num_classes, dense_classifier=False, pretrained=False, L=7, N=32, nonlinearity=nn.ReLU()): 
  channels, width, height = input_shape
  
  # Convolutional feature extractor
  modules = []
  modules.append(layers.Conv2d(channels, N, kernel_size=3, padding=3//2))
  modules.append(nonlinearity)
  for i in range(L-2):
    modules.append(layers.Conv2d(N, N, kernel_size=3, padding=3//2))
    modules.append(nonlinearity)
      
  # Linear classifier
  modules.append(nn.Flatten())
  if dense_classifier:
    modules.append(nn.Linear(N * width * height, num_classes))
  else:
    modules.append(layers.Linear(N * width * height, num_classes))
  model = nn.Sequential(*modules)
  model.apply(weights_init)
  # Pretrained model
  if pretrained:
    print("WARNING: this model does not have pretrained weights.")
  
  return model

def mlps(input_shape, num_classes, dense_classifier,pretrained,nonlinearity=nn.ReLU()):
  size = np.prod(input_shape)
  
  # modules = []
  # modules.append(nn.Flatten())
  # modules.append(layers.Linear(size,64))
  # modules.append(layers.BatchNorm1d(64))
  # modules.append(nonlinearity)
  # modules.append(layers.Linear(64,32))
  # modules.append(layers.BatchNorm1d(32))
  # modules.append(nonlinearity)
  # modules.append(layers.Linear(32,num_classes))
  # # modules.append(layers.BatchNorm1d(10))
  # model = nn.Sequential(*modules)

 
  
  if dense_classifier:
   model = nn.Sequential(OrderedDict([
          ("flatten", nn.Flatten()),
          ("linear_1", layers.Linear(size,64)),
          ("bn_1", layers.BatchNorm1d(64)),
          ("linearity_1", nonlinearity),
          ("linear_2", layers.Linear(64,32)),
          ("bn_2", layers.BatchNorm1d(32)),
          ("linearity_2", nonlinearity),
          ("linear_3", nn.Linear(32,10)),
        ]))
  else:
       model = nn.Sequential(OrderedDict([
          ("flatten", nn.Flatten()),
          ("linear_1", layers.Linear(size,64)),
          ("bn_1", layers.BatchNorm1d(64)),
          ("linearity_1", nonlinearity),
          ("linear_2", layers.Linear(64,32)),
          ("bn_2", layers.BatchNorm1d(32)),
          ("linearity_2", nonlinearity),
          ("linear_3", layers.Linear(32,10)),
        ]))

  # model = nn.Sequential(OrderedDict([
  #         ("flatten", nn.Flatten()),
  #         ("linear_1", layers.Linear(size,720)),
  #         ("bn_1", layers.BatchNorm1d(720)),
  #         ("linearity_1", nonlinearity),
  #         ("linear_2", layers.Linear(720,360)),
  #         ("bn_2", layers.BatchNorm1d(360)),
  #         ("linearity_2", nonlinearity),
  #         ("linear_3", layers.Linear(360,180)),
  #         ("bn_3", layers.BatchNorm1d(180)),
  #         ("linearity_3", nonlinearity),
  #         ("linear_4", layers.Linear(180,90)),
  #         ("bn_4", layers.BatchNorm1d(90)),
  #         ("linearity_4", nonlinearity),
  #         ("linear_5", layers.Linear(90,45)),
  #         ("bn_5", layers.BatchNorm1d(45)),
  #         ("linearity_5", nonlinearity),
  #         ("linear_6", nn.Linear(45,10)),
  #       ]))

  model.apply(weights_init)      
  return model

def mlps_bn(input_shape, num_classes, dense_classifier,pretrained,nonlinearity=nn.ReLU()):
  size = np.prod(input_shape)
  # if dense_classifier:
  #  model = nn.Sequential(OrderedDict([
  #         ("flatten", nn.Flatten()),
  #         ("linear_1", layers.Linear_1(size,64)),
  #         ("linearity_1", nonlinearity),
  #         ("linear_2", layers.Linear_1(64,32)),
  #         ("linearity_2", nonlinearity),
  #         ("linear_3", nn.Linear(32,10)),
  #       ]))
  # else:
  #      model = nn.Sequential(OrderedDict([
  #         ("flatten", nn.Flatten()),
  #         ("linear_1", layers.Linear_1(size,64)),
  #         ("linearity_1", nonlinearity),
  #         ("linear_2", layers.Linear_1(64,32)),
  #         ("linearity_2", nonlinearity),
  #         ("linear_3", layers.Linear_1(32,10)),
  #       ]))
  model = mlp_bn(input_shape, num_classes, dense_classifier,pretrained,nonlinearity=nn.ReLU())
  model.apply(weights_init)      
  return model 

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
        
#         # Declare all the layers for feature extraction
#         self.features = nn.Sequential(
#             layers.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             layers.BatchNorm2d(5),
#             layers.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1),
#             nn.MaxPool2d(2, 2),
#             nn.ReLU(inplace=True),
#             layers.BatchNorm2d(10),
#             layers.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             layers.BatchNorm2d(20),
#             layers.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1),
#             nn.MaxPool2d(2, 2),
#             nn.ReLU(inplace=True),
#             layers.BatchNorm2d(40),
#         )
        
#         # Declare all the layers for classification
#         self.classifier = nn.Sequential(
#             layers.Linear(7 * 7 * 40, 1024),
#             nn.ReLU(inplace=True),
#             layers.BatchNorm1d(1024),
#             layers.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             layers.BatchNorm1d(512),
#             layers.Linear(512, 10)
#         )
#     def forward(self, x):
#         # Apply the feature extractor in the input
#         x = self.features(x)
        
#         # Squeeze the three spatial dimentions in one
#         x = x.view(-1, 7 * 7 * 40)
        
#         # Classifiy the image
#         x = self.classifier(x)
#         return x
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = layers.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.linearity_1 = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.batchnorm_1 = layers.BatchNorm2d(5)
        self.conv_2 = layers.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        self.maxpool_1 = nn.MaxPool2d(2, 2)
        self.linearity_2= nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.batchnorm_2 = layers.BatchNorm2d(10)  
        self.conv_3 = layers.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1)
        self.linearity_3 = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.batchnorm_3 = layers.BatchNorm2d(20)
        self.conv_4 = layers.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1)
        self.maxpool_2 = nn.MaxPool2d(2, 2)
        self.linearity_4 = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.batchnorm_4 = layers.BatchNorm2d(40)
        self.linear_1 = layers.Linear(7 * 7 * 40, 1024)
        self.linearity_5 = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.batchnorm_5 = layers.BatchNorm1d(1024)
        self.linear_2 = layers.Linear(1024, 512)
        self.linearity_6 = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.batchnorm_6 = layers.BatchNorm1d(512)
        self.linear_3 = layers.Linear(512, 10)

    def forward(self, x):
        
        output = self.conv_1(x)
        output = self.linearity_1(output)
        output = self.batchnorm_1(output)
        output = self.conv_2(output)
        output = self.maxpool_1(output)
        output = self.linearity_2(output)
        output = self.batchnorm_2(output)
        output = self.conv_3(output)
        output = self.linearity_3(output)
        output = self.batchnorm_3(output)
        output = self.conv_4(output)
        output = self.maxpool_2(output)

        output = self.linearity_4(output)
        
        output = self.batchnorm_4(output)
        output = output.view(-1, 7 * 7 * 40)
        output = self.linear_1(output)
        output = self.linearity_5(output)
        output = self.batchnorm_5(output)
        output = self.linear_2(output)
        output = self.linearity_6(output)
        output = self.batchnorm_6(output)
        output = self.linear_3(output)
        
        return output

class Net_cifar10(nn.Module):
    def __init__(self):
        super(Net_cifar10, self).__init__()
        self.conv_1 = layers.Conv2d(in_channels=3, out_channels=5, kernel_size=3, padding=1)
        self.linearity_1 = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.batchnorm_1 = layers.BatchNorm2d(5)
        self.conv_2 = layers.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        self.maxpool_1 = nn.MaxPool2d(2, 2)
        self.linearity_2= nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.batchnorm_2 = layers.BatchNorm2d(10)  
        self.conv_3 = layers.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1)
        self.linearity_3 = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.batchnorm_3 = layers.BatchNorm2d(20)
        self.conv_4 = layers.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1)
        self.maxpool_2 = nn.MaxPool2d(2, 2)
        self.linearity_4 = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.batchnorm_4 = layers.BatchNorm2d(40)
        self.linear_1 = layers.Linear(8 * 8 * 40, 1024)
        self.linearity_5 = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.batchnorm_5 = layers.BatchNorm1d(1024)
        self.linear_2 = layers.Linear(1024, 512)
        self.linearity_6 = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.batchnorm_6 = layers.BatchNorm1d(512)
        self.linear_3 = layers.Linear(512, 10)

    def forward(self, x):
        
        output = self.conv_1(x)
        output = self.linearity_1(output)
        output = self.batchnorm_1(output)
        output = self.conv_2(output)
        output = self.maxpool_1(output)
        output = self.linearity_2(output)
        output = self.batchnorm_2(output)
        output = self.conv_3(output)
        output = self.linearity_3(output)
        output = self.batchnorm_3(output)
        output = self.conv_4(output)
        output = self.maxpool_2(output)

        output = self.linearity_4(output)
        
        output = self.batchnorm_4(output)
        output = output.view(-1, 8 * 8 * 40)
        output = self.linear_1(output)
        output = self.linearity_5(output)
        output = self.batchnorm_5(output)
        output = self.linear_2(output)
        output = self.linearity_6(output)
        output = self.batchnorm_6(output)
        output = self.linear_3(output)
        
        return output 

class conv_net_bn(nn.Module):

    def __init__(self, input_shape, num_classes, dense_classifier,pretrained,nonlinearity=nn.ReLU()):
        super(conv_net_bn,self).__init__()
        
        self.num_classes = num_classes
        self.dense_classifier = dense_classifier
        self.nonlinearity = nonlinearity
       
        if self.dense_classifier == False:

          
          self.conv_1 = layers.Conv2d_1(in_channels=1, out_channels=5, kernel_size=3, padding=1)
          self.linearity_1 = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
          
          self.conv_2 = layers.Conv2d_1(in_channels=5, out_channels=10, kernel_size=3, padding=1)
          self.maxpool_1 = nn.MaxPool2d(2, 2)
          self.linearity_2= nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
          
          self.conv_3 = layers.Conv2d_1(in_channels=10, out_channels=20, kernel_size=3, padding=1)
          self.linearity_3 = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
          self.conv_4 = layers.Conv2d_1(in_channels=20, out_channels=40, kernel_size=3, padding=1)
          self.maxpool_2 = nn.MaxPool2d(2, 2)
          self.linearity_4 = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
          self.linear_1 = layers.Linear_1(7 * 7 * 40, 1024)
          self.linearity_5 = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
          self.linear_2 = layers.Linear_1(1024, 512)
          self.linearity_6 = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01,inplace=True)
          self.linear_3 = layers.Linear_1(512, 10)
          
        else:
          self.conv_1 = layers.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1),
          self.linearity_1 = nn.ReLU(inplace=True),
          
          self.conv_2 = layers.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1),
          self.maxpool_1 = nn.MaxPool2d(2, 2),
          self.linearity_2= nn.ReLU(inplace=True),
          
          self.conv_3 = layers.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1),
          self.linearity_3 = nn.ReLU(inplace=True),
          self.conv_4 = layers.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1),
          self.maxpool_2 = nn.MaxPool2d(2, 2),
          self.linearity_4 = nn.ReLU(inplace=True),
          self.linear_1 = layers.Linear(7 * 7 * 40, 1024),
          self.linearity_5 = nn.ReLU(inplace=True),
          self.linear_2 = layers.Linear(1024, 512),
          self.linearity_6 = nn.ReLU(inplace=True),
          self.linear_3 = nn.Linear(512, 10)
          

    def forward(self, x,a,b,y,z,st):
        
        output = self.conv_1(x,a,b,st)
        output = self.linearity_1(output)
        output = self.conv_2(output,a,b,st)
        output = self.maxpool_1(output)
        output = self.linearity_2(output)
        output = self.conv_3(output,a,b,st)
        output = self.linearity_3(output)
        output = self.conv_4(output,a,b,st)
        output = self.maxpool_2(output)
        output = self.linearity_4(output)
        output = output.view(-1, 7 * 7 * 40)
        output = self.linear_1(output,y,z,st)
        output = self.linearity_5(output)
        output = self.linear_2(output,y,z,st)
        output = self.linearity_6(output)
        output = self.linear_3(output,y,z,st)
        
        return output

def conv_net(input_shape, num_classes, dense_classifier,pretrained,nonlinearity=nn.ReLU()):
  
  model = Net()
  model.apply(weights_init)      
  return model

def conv_net_cifar10(input_shape, num_classes, dense_classifier,pretrained,nonlinearity=nn.ReLU()):
  
  model = Net_cifar10()
  model.apply(weights_init)      
  return model 

def conv_netbn(input_shape, num_classes, dense_classifier,pretrained,nonlinearity=nn.ReLU()):
  
  model = conv_net_bn(input_shape, num_classes, dense_classifier,pretrained,nonlinearity=nn.ReLU())
  model.apply(weights_init)      
  return model 

def weights_init(m):
    if (type(m) == layers.Conv2d) or (type(m) == layers.Conv2d_1):
        
        torch.nn.init.xavier_uniform(m.weight)
    if (type(m) == layers.Linear) or (type(m) == layers.Linear_1) :    
        torch.nn.init.xavier_uniform(m.weight)