from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class PoseFeature(nn.Module):
    def __init__(self, num_points = 6890):
        super(PoseFeature, self).__init__()
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.norm1 = torch.nn.InstanceNorm1d(64)
        self.norm2 = torch.nn.InstanceNorm1d(128)
        self.norm3 = torch.nn.InstanceNorm1d(1024)

        self.num_points = num_points

    def forward(self, x):

        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)

        return x


class SPAdaIN(nn.Module):
    def __init__(self,norm,input_nc,planes):
        super(SPAdaIN,self).__init__()
        self.conv_weight = nn.Conv1d(input_nc, planes, 1)
        self.conv_bias = nn.Conv1d(input_nc, planes, 1)
        self.norm = norm(planes)
    
    def forward(self,x,addition):

        x = self.norm(x)
        weight = self.conv_weight(addition)
        bias = self.conv_bias(addition)
        out =  weight * x + bias

        return out

class SPAdaIN1(nn.Module):
    def __init__(self,norm,input_nc,planes):
        super(SPAdaIN1,self).__init__()
        self.conv_weight = nn.Conv1d(input_nc, planes, 1)
        self.conv_bias = nn.Conv1d(input_nc, planes, 1)
        self.norm = norm(planes)
    
    def forward(self,x,addition):
        
        weight= self.conv_weight(addition)
        bias = self.conv_bias(addition)
        out =  weight * x + bias

        return out

class SPAdaINResBlock1(nn.Module):
    def __init__(self,input_nc,planes,norm=nn.InstanceNorm1d,conv_kernel_size=1,padding=0):
        super(SPAdaINResBlock1,self).__init__()
        self.spadain1 = SPAdaIN1(norm=norm,input_nc=input_nc,planes=planes)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.spadain2 = SPAdaIN1(norm=norm,input_nc=input_nc,planes=planes)
        self.conv2 = nn.Conv1d(planes,planes,kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.spadain_res = SPAdaIN1(norm=norm,input_nc=input_nc,planes=planes)
        self.conv_res=nn.Conv1d(planes,planes,kernel_size=conv_kernel_size, stride=1, padding=padding)

    def forward(self,x,addition):
        #print(x.shape)
        out = self.spadain1(x,addition)
        #print(out.shape)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.spadain2(out,addition)
        out = self.relu(out)
        out = self.conv2(out)

        residual = x
        residual = self.spadain_res(residual,addition)
        residual = self.relu(residual)
        residual = self.conv_res(residual)

        out = out + residual

        return  out

class SPAdaINResBlock(nn.Module):
    def __init__(self,input_nc,planes,norm=nn.InstanceNorm1d,conv_kernel_size=1,padding=0):
        super(SPAdaINResBlock,self).__init__()
        self.spadain1 = SPAdaIN(norm=norm,input_nc=input_nc,planes=planes)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.spadain2 = SPAdaIN(norm=norm,input_nc=input_nc,planes=planes)
        self.conv2 = nn.Conv1d(planes,planes,kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.spadain_res = SPAdaIN(norm=norm,input_nc=input_nc,planes=planes)
        self.conv_res=nn.Conv1d(planes,planes,kernel_size=conv_kernel_size, stride=1, padding=padding)

    def forward(self,x,addition):

        out = self.spadain1(x,addition)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.spadain2(out,addition)
        out = self.relu(out)
        out = self.conv2(out)

        residual = x
        residual = self.spadain_res(residual,addition)
        residual = self.relu(residual)
        residual = self.conv_res(residual)

        out = out + residual

        return  out


class Decoder(nn.Module):
    def __init__(self, bottleneck_size = 1024):
        self.bottleneck_size = bottleneck_size
        super(Decoder, self).__init__()

        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1) 
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1) 
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1) 
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1) 

        self.spadain_block1 = SPAdaINResBlock1(input_nc=3 ,planes=self.bottleneck_size)
        self.spadain_block2 = SPAdaINResBlock(input_nc=3 ,planes=self.bottleneck_size//2)
        self.spadain_block3 = SPAdaINResBlock(input_nc=3 ,planes=self.bottleneck_size//4)
        
        self.norm1 = torch.nn.InstanceNorm1d(self.bottleneck_size)
        self.norm2 = torch.nn.InstanceNorm1d(self.bottleneck_size//2)
        self.norm3 = torch.nn.InstanceNorm1d(self.bottleneck_size//4)
        self.th = nn.Tanh()
        

    def forward(self, x, addition):

        x = self.conv1(x)
        x = self.spadain_block1(x,addition)
        x = self.conv2(x)
        x = self.spadain_block2(x,addition)
        x = self.conv3(x)
        x = self.spadain_block3(x,addition)        
        x = 2*self.th(self.conv4(x))

        return x

        

class NPT(nn.Module):
    def __init__(self, num_points = 6890, bottleneck_size = 1024):
        super(NPT, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        
        self.encoder = PoseFeature(num_points = num_points)
        self.decoder = Decoder(bottleneck_size = self.bottleneck_size+3)

    def forward(self, x1, x2):

        x1 = self.encoder(x1)
        y = torch.cat((x1, x2), 1)
        out =self.decoder(y,x2)

        return out.transpose(2,1)