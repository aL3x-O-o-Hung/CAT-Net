import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class CrossSliceAttention(nn.Module):
    def __init__(self,input_channels):
        super(CrossSliceAttention,self).__init__()
        self.linear_q=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1),bias=False)
        self.linear_k=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1),bias=False)
        self.linear_v=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1),bias=False)

    def forward(self,pooled_features,features):
        q=self.linear_q(pooled_features)
        q=q.view(q.size(0),-1)
        k=self.linear_k(pooled_features)
        k=k.view(k.size(0),-1)
        v=self.linear_v(features)
        x=torch.matmul(q,k.permute(1,0))/np.sqrt(q.size(1))
        x=torch.softmax(x,dim=1)
        out=torch.zeros_like(v)
        for i in range(x.size(0)):
            temp=x[i,:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out[i,:,:,:]=torch.sum(temp*v,dim=0).clone()
        return out


class MultiHeadedCrossSliceAttentionModule(nn.Module):
    def __init__(self,input_channels,heads=3,pool_kernel_size=(4,4),input_size=(1128,128),batch_size=20,pool_method='avgpool'):
        super(MultiHeadedCrossSliceAttentionModule,self).__init__()
        self.attentions=[]
        self.linear1=nn.Conv2d(in_channels=heads*input_channels,out_channels=input_channels,kernel_size=(1,1))
        self.norm1=nn.LayerNorm([batch_size,input_channels,input_size[0],input_size[1]])
        self.linear2=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1))
        self.norm2=nn.LayerNorm([batch_size,input_channels,input_size[0],input_size[1]])

        if pool_method=="maxpool":
            self.pool=nn.MaxPool2d(kernel_size=pool_kernel_size)
        elif pool_method=="avgpool":
            self.pool=nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            assert (False)  # not implemented yet

        for i in range(heads):
            self.attentions.append(CrossSliceAttention(input_channels))
        self.attentions=nn.Sequential(*self.attentions)

    def forward(self,pooled_features,features):

        for i in range(len(self.attentions)):
            x_=self.attentions[i](pooled_features,features)
            if i==0:
                x=x_
            else:
                x=torch.cat((x,x_),dim=1)
        out=self.linear1(x)
        x=F.gelu(out)+features
        out_=self.norm1(x)
        out=self.linear2(out_)
        x=F.gelu(out)+out_
        out=self.norm2(x)
        pooled_out=self.pool(out)
        return pooled_out,out


class PositionalEncoding(nn.Module):
    def __init__(self,d_model,is_pe_learnable=True,max_len=20):
        super(PositionalEncoding,self).__init__()

        position=torch.arange(max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe=torch.zeros(max_len,d_model,1,1)
        pe[:,0::2,0,0]=torch.sin(position*div_term)
        pe[:,1::2,0,0]=torch.cos(position*div_term)
        self.pe=nn.Parameter(pe.clone(),is_pe_learnable)
        #self.register_buffer('pe',self.pe)

    def forward(self,x):
        return x+self.pe[:x.size(0),:,:,:]

    def get_pe(self):
        return self.pe[:,:,0,0]
