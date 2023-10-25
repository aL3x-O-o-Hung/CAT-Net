from CAT_module import *


class ConvBlock(nn.Module):
    def __init__(self,input_channels,output_channels,max_pool,return_single=False):
        super(ConvBlock,self).__init__()
        self.max_pool=max_pool
        self.conv=[]
        self.conv.append(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.InstanceNorm2d(output_channels))
        self.conv.append(nn.LeakyReLU())
        self.conv.append(nn.Conv2d(in_channels=output_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.InstanceNorm2d(output_channels))
        self.conv.append(nn.LeakyReLU())
        self.return_single=return_single
        if max_pool:
            self.pool=nn.MaxPool2d(2,stride=2,dilation=(1,1))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x):
        x=self.conv(x)
        b=x
        if self.max_pool:
            x=self.pool(x)
        if self.return_single:
            return x
        else:
            return x,b


class DeconvBlock(nn.Module):
    def __init__(self,input_channels,output_channels,intermediate_channels=-1):
        super(DeconvBlock,self).__init__()
        input_channels=int(input_channels)
        output_channels=int(output_channels)
        if intermediate_channels<0:
            intermediate_channels=output_channels*2
        else:
            intermediate_channels=input_channels
        self.upconv=[]
        self.upconv.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.upconv.append(nn.Conv2d(in_channels=input_channels,out_channels=intermediate_channels//2,kernel_size=3,stride=1,padding=1))
        self.conv=ConvBlock(intermediate_channels,output_channels,False)
        self.upconv=nn.Sequential(*self.upconv)

    def forward(self,x,b):
        x=self.upconv(x)
        x=torch.cat((x,b),dim=1)
        x,_=self.conv(x)
        return x

class UNetDecoder(nn.Module):
    def __init__(self,num_layers,base_num):
        super(UNetDecoder,self).__init__()
        self.conv=[]
        self.num_layers=num_layers
        for i in range(num_layers-1,0,-1):
            self.conv.append(DeconvBlock(base_num*(2**i),base_num*(2**(i-1))))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x,b):
        for i in range(self.num_layers-1):
            x=self.conv[i](x,b[i])
        return x

class CrossSliceUNetEncoder(nn.Module):
    def __init__(self,input_channels,num_layers,base_num,num_attention_blocks=3,heads=4,pool_kernel_size=(4,4),input_size=(128,128),batch_size=20,pool_method='avgpool',is_pe_learnable=True):
        super(CrossSliceUNetEncoder,self).__init__()
        self.conv=[]
        self.num_layers=num_layers
        self.num_attention_blocks=num_attention_blocks
        for i in range(num_layers):
            if i==0:
                self.conv.append(ConvBlock(input_channels,base_num,True))
            else:
                self.conv.append(ConvBlock(base_num*(2**(i-1)),base_num*(2**i),(i!=num_layers-1)))
        self.conv=nn.Sequential(*self.conv)
        self.pools=[]
        self.pes=[]
        self.attentions=[]
        for i in range(num_layers):
            if pool_method=='maxpool':
                self.pools.append(nn.MaxPool2d(kernel_size=pool_kernel_size))
            elif pool_method=='avgpool':
                self.pools.append(nn.AvgPool2d(kernel_size=pool_kernel_size))
            else:
                assert (False)  # not implemented yet

            self.pes.append(PositionalEncoding(base_num*(2**i),is_pe_learnable))
            temp=[]
            for j in range(num_attention_blocks):
                temp.append(MultiHeadedCrossSliceAttentionModule(base_num*(2**i),heads,pool_kernel_size,input_size,batch_size,pool_method))
            input_size=(input_size[0]//2,input_size[1]//2)
            self.attentions.append(nn.Sequential(*temp))
        self.attentions=nn.Sequential(*self.attentions)
        self.pes=nn.Sequential(*self.pes)

    def forward(self,x):
        b=[]
        for i in range(self.num_layers):
            x,block=self.conv[i](x)
            if i!=self.num_layers-1:
                block=self.pes[i](block)
                block_pool=self.pools[i](block)
                for j in range(self.num_attention_blocks):
                    block_pool,block=self.attentions[i][j](block_pool,block)
            else:
                x=self.pes[i](x)
                x_pool=self.pools[i](x)
                for j in range(self.num_attention_blocks):
                    x_pool,x=self.attentions[i][j](x_pool,x)
            b.append(block)
        b=b[:-1]
        b=b[::-1]
        return x,b



class CrossSliceAttentionUNet(nn.Module):
    def __init__(self,input_channels,num_classes,num_layers,heads=3,num_attention_blocks=2,base_num=64,pool_kernel_size=(4,4),input_size=(128,128),batch_size=20,pool_method="avgpool",is_pe_learnable=True):
        super(CrossSliceAttentionUNet,self).__init__()
        self.encoder=CrossSliceUNetEncoder(input_channels,num_layers,base_num,num_attention_blocks,heads,pool_kernel_size,input_size,batch_size,pool_method,is_pe_learnable)
        self.decoder=UNetDecoder(num_layers,base_num)
        self.base_num=base_num
        self.input_channels=input_channels
        self.num_classes=num_classes
        self.conv_final=nn.Conv2d(in_channels=base_num,out_channels=num_classes,kernel_size=(1,1))

    def forward(self,x):
        x,b=self.encoder(x)
        x=self.decoder(x,b)
        x=self.conv_final(x)
        return x



class CrossSliceUNetPlusPlus(nn.Module):
    def __init__(self,input_channels,num_classes,num_layers,heads=3,num_attention_blocks=2,base_num=64,pool_kernel_size=(4,4),input_size=(128,128),batch_size=20,pool_method="maxpool",is_pe_learnable=True):
        super(CrossSliceUNetPlusPlus).__init__()
        self.num_layers=num_layers
        self.num_attention_blocks=num_attention_blocks
        nb_filter=[]
        for i in range(num_layers):
            nb_filter.append(base_num*(2**i))
        self.pool=nn.MaxPool2d(2,2)
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv=[]
        for i in range(num_layers):
            temp_conv=[]
            for j in range(num_layers-i):
                if j==0:
                    if i==0:
                        inp=input_channels
                    else:
                        inp=nb_filter[i-1]
                else:
                    inp=nb_filter[i]*j+nb_filter[i+1]
                temp_conv.append(ConvBlock(inp,nb_filter[i],False,True))
            self.conv.append(nn.Sequential(*temp_conv))
        self.conv=nn.Sequential(*self.conv)
        self.pools=[]
        self.pes=[]
        self.attentions=[]
        for i in range(num_layers):
            if pool_method=='maxpool':
                self.pools.append(nn.MaxPool2d(kernel_size=pool_kernel_size))
            elif pool_method=='avgpool':
                self.pools.append(nn.AvgPool2d(kernel_size=pool_kernel_size))
            else:
                assert (False)  # not implemented yet

            self.pes.append(PositionalEncoding(base_num*(2**i),is_pe_learnable))
            temp=[]
            for j in range(num_attention_blocks):
                temp.append(MultiHeadedCrossSliceAttentionModule(base_num*(2**i),heads,pool_kernel_size,input_size,batch_size,pool_method))
            input_size=(input_size[0]//2,input_size[1]//2)
            self.attentions.append(nn.Sequential(*temp))
        self.attentions=nn.Sequential(*self.attentions)
        self.pes=nn.Sequential(*self.pes)
        self.final=[]
        for i in range(num_layers-1):
            self.final.append(nn.Conv2d(nb_filter[0],num_classes,kernel_size=(1,1)))
        self.final=nn.Sequential(*self.final)

    def forward(self,inputs):
        x=[]
        for i in range(self.num_layers):
            temp=[]
            for j in range(self.num_layers-i):
                temp.append([])
            x.append(temp)
        x[0][0].append(self.conv[0][0](inputs))
        for s in range(1,self.num_layers):
            for i in range(s+1):
                if i==0:
                    x[s-i][i].append(self.conv[s-i][i](self.pool(x[s-i-1][i][0])))
                else:
                    for j in range(i):
                        if j==0:
                            block=x[s-i][j][0]
                            block_pool=self.pools[s-i](block)
                            for k in range(self.num_attention_blocks):
                                block_pool,block=self.attentions[s-i][k](block_pool,block)
                            temp_x=block
                            #print(s-i,j)
                        else:
                            temp_x=torch.cat((temp_x,x[s-i][j][0]),dim=1)
                            #print(s-i,j)
                    temp_x=torch.cat((temp_x,self.up(x[s-i+1][i-1][0])),dim=1)
                    #print('up',s-i+1,i-1,temp_x.size(),self.up(x[s-i+1][i-1][0]).size())
                    x[s-i][i].append(self.conv[s-i][i](temp_x))
        if self.training:
            res=[]
            for i in range(self.num_layers-1):
                res.append(self.final[i](x[0][i+1][0]))
            return res
        else:
            return self.final[-1](x[0][-1][0])