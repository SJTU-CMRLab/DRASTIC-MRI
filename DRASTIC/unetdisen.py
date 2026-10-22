"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn

from modelutils import ModalityEncoderNew,SPADENew,AnatomyEncoderEncNew,AnatomyEncoderDecNew
from unet import Unet,UnetCond


class DisenNetwork(nn.Module):
    def __init__(
        self,
        in_chans: int=2,
        out_chans: int=2,
        image_size = (256,384)
    ):
        super().__init__()
        content_chans = 4
        self.content_encoder = Unet(in_chans=in_chans,out_chans=content_chans,chans=16,num_pool_layers=4)
        # 功能：将输入图像编码为content map，其size与图像相同，channel数由content_chans确定
        self.content_denoiser = Unet(in_chans=content_chans,out_chans=content_chans,chans=32,num_pool_layers=4)
        # 功能：对content map进行去噪和去伪影
        self.style_encoder = ModalityEncoderNew(img_num_ch=in_chans,s_num_ch=0,z_size=16,is_cond=True)
        # 功能：将输入图像编码为style code
        self.decoder = SPADENew(image_size=image_size,in_num_ch=in_chans, s_num_ch=content_chans,z_size=16,is_cond=True,output_activation='no')
        # 功能：用style code和content code组合得到图像
    def sample(self, z_mean, z_log_var):
        eps = torch.normal(0, 1, size=(z_mean.shape[0], z_mean.shape[1])).to(z_mean.device)
        zi = z_mean + eps * torch.exp(0.5 * z_log_var)
        return zi
    
    def forward(self, image: torch.Tensor,inputs_type=None) -> torch.Tensor:
        cont = self.content_encoder(image)
        sty_mean, sty_log_var = self.style_encoder(image,si=None,inputs_type=inputs_type)
        sty_mean = sty_mean.clamp(-1000.0,1000.0)
        sty_log_var = sty_log_var.clamp(-10.0,10.0)
        sty = self.sample(sty_mean,sty_log_var)
        #if torch.any(torch.isnan(sty)):
        #    sty = torch.where(torch.isnan(sty), 0, sty)
        cont_denoised = self.content_denoiser(cont)
        output = self.decoder(cont_denoised,sty,inputs_type=inputs_type)
        return output,cont,sty

class DisenNetworkV2_2(nn.Module):
    def __init__(
        self,
        in_chans: int=2,
        out_chans: int=2,
        image_size = (256,384)
    ):
        super().__init__()
        content_chans = 4
        self.content_encoder = Unet(in_chans=in_chans,out_chans=content_chans,chans=8,num_pool_layers=4)
        # 功能：将输入图像编码为content map，其size与图像相同，channel数由content_chans确定
        self.content_denoiser = Unet(in_chans=content_chans,out_chans=content_chans,chans=8,num_pool_layers=4)
        # 功能：对content map进行去噪和去伪影
        self.style_encoder = ModalityEncoderNew(img_num_ch=in_chans,s_num_ch=0,z_size=16,first_num_ch=8,is_cond=True)
        # 功能：将输入图像编码为style code
        self.decoder = SPADENew(image_size=image_size,in_num_ch=in_chans, s_num_ch=content_chans,z_size=16,is_cond=True,output_activation='no')
        # 功能：用style code和content code组合得到图像
    def sample(self, z_mean, z_log_var):
        eps = torch.normal(0, 1, size=(z_mean.shape[0], z_mean.shape[1])).to(z_mean.device)
        zi = z_mean + eps * torch.exp(0.5 * z_log_var)
        return zi

    def forward(self, image: torch.Tensor,inputs_type=None) -> torch.Tensor:
        cont = self.content_encoder(image)
        sty_mean, sty_log_var = self.style_encoder(image,si=None,inputs_type=inputs_type)
        sty_mean = sty_mean.clamp(-1000.0,1000.0)
        sty_log_var = sty_log_var.clamp(-10.0,10.0)
        sty = self.sample(sty_mean,sty_log_var)
        #if torch.any(torch.isnan(sty)):
        #    sty = torch.where(torch.isnan(sty), 0, sty)
        cont_denoised = self.content_denoiser(cont)
        output = self.decoder(cont_denoised,sty,inputs_type=inputs_type)
        return output,cont,sty

class DisenNetworkV3_2(nn.Module):
    def __init__(
        self,
        in_chans: int=2,
        out_chans: int=2,
        image_size = (256,384)
    ):
        super().__init__()
        content_chans = 4
        self.content_encoder = Unet(in_chans=in_chans,out_chans=content_chans,chans=8,num_pool_layers=4)
        # 功能：将输入图像编码为content map，其size与图像相同，channel数由content_chans确定
        self.content_denoiser = Unet(in_chans=content_chans,out_chans=content_chans,chans=8,num_pool_layers=4)
        # 功能：对content map进行去噪和去伪影
        self.style_encoder = ModalityEncoderNew(img_num_ch=in_chans,s_num_ch=0,z_size=16,first_num_ch=8,is_cond=False)
        # 功能：将输入图像编码为style code
        self.decoder = SPADENew(image_size=image_size,in_num_ch=in_chans, s_num_ch=content_chans,z_size=16,is_cond=False,output_activation='no')
        # 功能：用style code和content code组合得到图像
    def sample(self, z_mean, z_log_var):
        eps = torch.normal(0, 1, size=(z_mean.shape[0], z_mean.shape[1])).to(z_mean.device)
        zi = z_mean + eps * torch.exp(0.5 * z_log_var)
        return zi

    def forward(self, image: torch.Tensor,inputs_type=None) -> torch.Tensor:
        cont = self.content_encoder(image)
        sty_mean, sty_log_var = self.style_encoder(image,si=None,inputs_type=None)
        sty_mean = sty_mean.clamp(-1000.0,1000.0)
        sty_log_var = sty_log_var.clamp(-10.0,10.0)
        sty = self.sample(sty_mean,sty_log_var)

        cont_denoised = self.content_denoiser(cont)
        output = self.decoder(cont_denoised,sty,inputs_type=None)
        return output,cont,sty

class DisenNetworkV3_3(nn.Module):
    def __init__(
        self,
        in_chans: int=2,
        out_chans: int=2,
        image_size = (256,384)
    ):
        super().__init__()
        content_chans = 4
        self.content_encoder = Unet(in_chans=in_chans,out_chans=content_chans,chans=8,num_pool_layers=4)
        # 功能：将输入图像编码为content map，其size与图像相同，channel数由content_chans确定
        #self.content_denoiser = Unet(in_chans=content_chans,out_chans=content_chans,chans=8,num_pool_layers=4)
        # 功能：对content map进行去噪和去伪影
        self.style_encoder = ModalityEncoderNew(img_num_ch=in_chans,s_num_ch=0,z_size=16,first_num_ch=8,is_cond=False)
        # 功能：将输入图像编码为style code
        self.decoder = UnetCond(in_chans=content_chans,out_chans=out_chans,chans=16,num_pool_layers=4)
        # 功能：用style code和content code组合得到图像
    def sample(self, z_mean, z_log_var):
        eps = torch.normal(0, 1, size=(z_mean.shape[0], z_mean.shape[1])).to(z_mean.device)
        zi = z_mean + eps * torch.exp(0.5 * z_log_var)
        return zi

    def forward(self, image: torch.Tensor,inputs_type=None) -> torch.Tensor:
        cont = self.content_encoder(image)
        sty_mean, sty_log_var = self.style_encoder(image,si=None,inputs_type=None)
        sty_mean = sty_mean.clamp(-1000.0,1000.0)
        sty_log_var = sty_log_var.clamp(-10.0,10.0)
        sty = self.sample(sty_mean,sty_log_var)
        
        output = self.decoder(cont,sty)
        return output,cont,sty

#V3_4_2 是目前的有效版本 2024.11.25
class DisenNetworkV3_4(nn.Module):
    def __init__(
        self,
        in_chans: int=2,
        out_chans: int=2,
        image_size = (256,384)
    ):
        super().__init__()
        content_chans = 4
        self.content_encoder = Unet(in_chans=in_chans,out_chans=content_chans,chans=8,num_pool_layers=4)
        # 功能：将输入图像编码为content map，其size与图像相同，channel数由content_chans确定
        #self.content_denoiser = Unet(in_chans=content_chans,out_chans=content_chans,chans=8,num_pool_layers=4)
        # 功能：对content map进行去噪和去伪影
        self.style_encoder = ModalityEncoderNew(img_num_ch=in_chans,s_num_ch=0,z_size=16,first_num_ch=8,is_cond=True)
        # 功能：将输入图像编码为style code
        self.decoder = UnetCond(in_chans=content_chans,out_chans=out_chans,chans=16,num_pool_layers=4)
        # 功能：用style code和content code组合得到图像

    def forward(self, image: torch.Tensor,inputs_type=None) -> torch.Tensor:
        cont = self.content_encoder(image)
        sty_1, sty_2 = self.style_encoder(image,si=None,inputs_type=inputs_type) #似乎也应将content输入到style_encoder
        sty = sty_1 + sty_2  #只用sty_1或者改成随机变量？
        
        output = self.decoder(cont,sty)
        return output,cont,sty

#在v3.4.2基础上增加content参数量 待实验
class DisenNetworkV3_4_4(nn.Module):
    def __init__(
        self,
        in_chans: int=2,
        out_chans: int=2,
        image_size = (256,384)
    ):
        super().__init__()
        content_chans = 4
        self.content_encoder = Unet(in_chans=in_chans,out_chans=content_chans,chans=16,num_pool_layers=4)
        # 功能：将输入图像编码为content map，其size与图像相同，channel数由content_chans确定
        #self.content_denoiser = Unet(in_chans=content_chans,out_chans=content_chans,chans=8,num_pool_layers=4)
        # 功能：对content map进行去噪和去伪影
        self.style_encoder = ModalityEncoderNew(img_num_ch=in_chans,s_num_ch=0,z_size=16,first_num_ch=8,is_cond=True)
        # 功能：将输入图像编码为style code
        self.decoder = UnetCond(in_chans=content_chans,out_chans=out_chans,chans=16,num_pool_layers=4)
        # 功能：用style code和content code组合得到图像

    def forward(self, image: torch.Tensor,inputs_type=None) -> torch.Tensor:
        cont = self.content_encoder(image)
        sty_1, sty_2 = self.style_encoder(image,si=None,inputs_type=inputs_type) #似乎也应将content输入到style_encoder
        sty = sty_1 + sty_2  #只用sty_1或者改成随机变量？
        output = self.decoder(cont,sty)
        return output,cont,sty


class DisenNetworkV3_4_3(nn.Module):
    def __init__(
        self,
        in_chans: int=2,
        out_chans: int=2,
        image_size = (256,384)
    ):
        super().__init__()
        content_chans = 4
        self.content_encoder = Unet(in_chans=in_chans,out_chans=content_chans,chans=12,num_pool_layers=4)
        # 功能：将输入图像编码为content map，其size与图像相同，channel数由content_chans确定
        #self.content_denoiser = Unet(in_chans=content_chans,out_chans=content_chans,chans=8,num_pool_layers=4)
        # 功能：对content map进行去噪和去伪影
        self.style_encoder = ModalityEncoderNew(img_num_ch=in_chans,s_num_ch=0,z_size=16,first_num_ch=8,is_cond=True)
        # 功能：将输入图像编码为style code
        self.decoder = UnetCond(in_chans=content_chans,out_chans=out_chans,chans=24,num_pool_layers=4)
        # 功能：用style code和content code组合得到图像

    def forward(self, image: torch.Tensor,inputs_type=None) -> torch.Tensor:
        cont = self.content_encoder(image)
        sty_1, sty_2 = self.style_encoder(image,si=None,inputs_type=inputs_type)
        sty = sty_1 + sty_2
        
        output = self.decoder(cont,sty)
        return output,cont,sty
#Copied from V3_4 使用幅值解耦 style encoder中也输入content
class DisenNetworkV3_6(nn.Module):
    def __init__(
        self,
        in_chans: int=2,
        out_chans: int=2,
        image_size = (256,384),
        cont_input = False
    ):
        super().__init__()
        content_chans = 4
        self.content_encoder = Unet(in_chans=1,out_chans=content_chans,chans=8,num_pool_layers=4)
        # 功能：将输入图像编码为content map，其size与图像相同，channel数由content_chans确定
        #self.content_denoiser = Unet(in_chans=content_chans,out_chans=content_chans,chans=8,num_pool_layers=4)
        # 功能：对content map进行去噪和去伪影  first_num_ch从8改成16
        self.cont_input = cont_input
        if self.cont_input:  #是否把content也作为输入 默认false
            self.style_encoder = ModalityEncoderNew(img_num_ch=1, s_num_ch=4, first_num_ch=16, z_size=16, is_cond=True)
        else:
            self.style_encoder = ModalityEncoderNew(img_num_ch=1, s_num_ch=0, first_num_ch=16, z_size=16, is_cond=True)
        #self.style_encoder = ModalityEncoderNew(img_num_ch=1,s_num_ch=4,z_size=16,first_num_ch=8,is_cond=True)
        # 功能：将输入图像编码为style code
        self.decoder = UnetCond(in_chans=content_chans,out_chans=1,chans=16,num_pool_layers=4)
        # 功能：用style code和content code组合得到图像
        self.phase_model = Unet(in_chans=1,out_chans=1,chans=8,num_pool_layers=4)
    def to_mag_phase(self,image:torch.Tensor):#input: Nb*2*Nx*Ny
        try:
            image = torch.view_as_complex(image.permute(0,2,3,1))  #
        except RuntimeError:
            image = torch.view_as_complex(image.permute(0,2,3,1).contiguous()) 
        image_mag = image.abs()
        image_phase = image.angle()
        return image_mag.unsqueeze(1), image_phase.unsqueeze(1)

    def forward(self, image: torch.Tensor,inputs_type=None) -> torch.Tensor:
        #输入转换成相位和幅值
        image_mag, image_phase = self.to_mag_phase(image)
        image_phase = image_phase + self.phase_model(image_phase)  #residual connection
        cont = self.content_encoder(image_mag)
        if self.cont_input:
            sty_1, sty_2 = self.style_encoder(image_mag,si=cont,inputs_type=inputs_type)
        else:
            sty_1, sty_2 = self.style_encoder(image_mag,si=None,inputs_type=inputs_type) #似乎也应将content输入到style_encoder
        sty = sty_1 + sty_2  #只用sty_1或者改成随机变量？
        
        output = self.decoder(cont,sty)
        output_real = output * torch.cos(image_phase)
        output_imag = output * torch.sin(image_phase)
        output = torch.concat([output_real,output_imag],dim=1)
        return output,cont,sty



from torch.nn import Linear, ReLU, Sequential, Dropout, Softmax
class MLP(torch.nn.Module):
    def __init__(self, input_n, output_n, num_layer=3, layer_list=[128, 64, 16], dropout=0.0):
        super(MLP, self).__init__()
        self.input_n = input_n
        self.output_n = output_n
        self.num_layer = num_layer
        self.layer_list = layer_list
        # 输入层
        self.input_layer = Sequential(
            Linear(input_n, layer_list[0], bias=False),
            ReLU()
        )
        # 隐藏层
        self.hidden_layer = Sequential()
        for index in range(num_layer-1):
            self.hidden_layer.extend([
            Linear(layer_list[index], layer_list[index+1], bias=False),
            ReLU()
            ])
            self.dropout = Dropout(dropout)
        # 输出层
        self.output_layer = Sequential(
            Linear(layer_list[-1], output_n, bias=False),
            Softmax(dim=1),
        )
    def forward(self, x):
        input = self.input_layer(x)
        hidden = self.hidden_layer(input)
        hidden = self.dropout(hidden)
        output = self.output_layer(hidden)
        return output


from topKDebug import TopK2,TopK3,TopK3_1,TopK4

class SamplingTrajEstimationV4_4(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        input_n =  kwargs.get('input_n')
        self.network = MLP(input_n=input_n,output_n=256,num_layer=2,layer_list=[64,128])
        self.fc = nn.Linear(in_features=256,out_features=256)
        self.sigmoid = nn.Sigmoid()

        self.topk = TopK2.apply
    def forward(self,ksp,x,num_lines=64,total_lines=128,mode='train'):
        x = self.network(x)  #2,256,8,12
        x = self.fc(x)
        acs = torch.zeros_like(x)
        acs[:,120:136] = 10
        p = self.sigmoid(x+acs)  #网络估计得到分布参数，其中ACS region概率必为1
        #去掉外周多余的线
        m = torch.zeros_like(p)
        for idx in range(x.shape[0]):
            m[idx,int(128-total_lines[idx]//2):int(128+total_lines[idx]//2)] = 1.0

        p = p * m

        mask = self.topk(p,num_lines[0])

        mask = torch.tile(mask.unsqueeze(-1),[1,1,ksp.shape[-2]])  #原来是1，1，384
        if ksp.shape[0] == 1:
            mask = mask.unsqueeze(0)
        else:
            mask = mask.unsqueeze(1)    
        masked_ksp = torch.view_as_real(torch.view_as_complex(ksp) * mask)
        
        return masked_ksp,mask 



     
#4.6 增加对PE lines=320的支持
class SamplingTrajEstimationV4_6(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        input_n =  kwargs.get('input_n')
        self.network = MLP(input_n=input_n,output_n=256,num_layer=2,layer_list=[64,128])
        self.fc = nn.Linear(in_features=256,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=320)
        self.sigmoid = nn.Sigmoid()
        self.topk = TopK3.apply
    def forward(self,ksp,x,num_lines=64,total_lines=128,mode='train'):
        x = self.network(x)  #2,256,8,12
        if ksp.shape[2] == 256:
            x = self.fc(x)
            acs = torch.zeros_like(x)
            acs[:,120:136] = 10
            p = self.sigmoid((x+acs))  #网络估计得到分布参数，其中ACS region概率必为1
            #去掉外周多余的线
            m = torch.zeros_like(p)
            for idx in range(x.shape[0]):
                m[idx,int(128-total_lines[idx]//2):int(128+total_lines[idx]//2)] = 1.0
            p = p * m
        elif ksp.shape[2] == 320:
            x = self.fc2(x)
            acs = torch.zeros_like(x)
            acs[:,152:168] = 10
            p = self.sigmoid((x+acs))  #网络估计得到分布参数，其中ACS region概率必为1
            #去掉外周多余的线
            m = torch.zeros_like(p)
            for idx in range(x.shape[0]):
                m[idx,int(160-total_lines[idx]//2):int(160+total_lines[idx]//2)] = 1.0
            p = p * m
        
        mask = self.topk(p,num_lines[0])
        mask = torch.tile(mask.unsqueeze(-1),[1,1,ksp.shape[-2]])  #原来是1，1，384
        if ksp.shape[0] == 1:
            mask = mask.unsqueeze(0)
        else:
            mask = mask.unsqueeze(1)    
        masked_ksp = torch.view_as_real(torch.view_as_complex(ksp) * mask)
        
        return masked_ksp,mask


#4.7 增加对自定义MLP大小的支持
class SamplingTrajEstimationV4_7(nn.Module):
    def __init__(self,layer_list=[64,128], **kwargs) -> None:
        super().__init__()
        input_n =  kwargs.get('input_n')
        self.network = MLP(input_n=input_n,output_n=256,num_layer=len(layer_list),layer_list=layer_list)
        self.fc = nn.Linear(in_features=256,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=320)
        self.sigmoid = nn.Sigmoid()
        self.topk = TopK3.apply
    def forward(self,ksp,x,num_lines=64,total_lines=128,mode='train'):
        x = self.network(x)  #2,256
        if ksp.shape[2] == 256:
            x = self.fc(x)
            acs = torch.zeros_like(x)
            acs[:,120:136] = 10
            p = self.sigmoid((x+acs))  #网络估计得到分布参数，其中ACS region概率必为1
            #去掉外周多余的线
            m = torch.zeros_like(p)
            for idx in range(x.shape[0]):
                m[idx,int(128-total_lines[idx]//2):int(128+total_lines[idx]//2)] = 1.0
            p = p * m
        elif ksp.shape[2] == 320:
            x = self.fc2(x)
            acs = torch.zeros_like(x)
            acs[:,152:168] = 10
            p = self.sigmoid((x+acs))  #网络估计得到分布参数，其中ACS region概率必为1
            #去掉外周多余的线
            m = torch.zeros_like(p)
            for idx in range(x.shape[0]):
                m[idx,int(160-total_lines[idx]//2):int(160+total_lines[idx]//2)] = 1.0
            p = p * m
        
        mask = self.topk(p,num_lines[0])
        mask = torch.tile(mask.unsqueeze(-1),[1,1,ksp.shape[-2]])  #原来是1，1，384
        if ksp.shape[0] == 1:
            mask = mask.unsqueeze(0)
        else:
            mask = mask.unsqueeze(1)    
        masked_ksp = torch.view_as_real(torch.view_as_complex(ksp) * mask)
        
        return masked_ksp,mask



class MLP2(torch.nn.Module):
    def __init__(self, input_n, output_n, num_layer=3, layer_list=[128, 64, 16], dropout=0.0, softmax=True):
        super(MLP2, self).__init__()
        self.input_n = input_n
        self.output_n = output_n
        self.num_layer = num_layer
        self.layer_list = layer_list
        # 输入层
        self.input_layer = Sequential(
            Linear(input_n, layer_list[0], bias=False),
            ReLU()
        )
        # 隐藏层
        self.hidden_layer = Sequential()
        for index in range(num_layer-1):
            self.hidden_layer.extend([
            Linear(layer_list[index], layer_list[index+1], bias=False),
            ReLU()
            ])
            self.dropout = Dropout(dropout)
        # 输出层
        if softmax:
            self.output_layer = Sequential(
                Linear(layer_list[-1], output_n, bias=False),
                Softmax(dim=1),
            )
        else:
            self.output_layer = Sequential(
                Linear(layer_list[-1], output_n, bias=False),
            )
    def forward(self, x):
        input = self.input_layer(x)
        hidden = self.hidden_layer(input)
        #hidden = self.dropout(hidden)
        output = self.output_layer(hidden)
        return output


# Uses MLP2, adds LayerNorm, and allows the MLP’s Softmax to be turned off; it also returns probabilities and optionally
class SamplingTrajEstimationV4_7_2(nn.Module):
    def __init__(self,layer_list=[64,128], **kwargs) -> None:
        super().__init__()
        input_n =  kwargs.get('input_n')
        self.anatomy_code =  kwargs.get('anatomy_code')
        softmax =  kwargs.get('softmax')
        if softmax is None:
            softmax = True
        self.network = MLP2(input_n=input_n,output_n=256,num_layer=len(layer_list),layer_list=layer_list,softmax=softmax)
        
        self.fc = nn.Linear(in_features=256,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=320)
        self.sigmoid = nn.Sigmoid()
        self.topk = TopK3_1.apply
        self.norm = nn.LayerNorm(256)
    def forward(self,ksp,x,num_lines=64,total_lines=128,mode='train'):
        x = self.network(x) #2,256
        x = self.norm(x)
        anatomy_code = x
        if ksp.shape[2] == 256:
            x = self.fc(x)            
            acs = torch.zeros_like(x)
            acs[:,120:136] = 2.0
            p = self.sigmoid((x+acs))  #网络估计得到分布参数，其中ACS region概率必为1
            #去掉外周多余的线
            m = torch.zeros_like(p)
            for idx in range(x.shape[0]):
                m[idx,int(128-total_lines[idx]//2):int(128+total_lines[idx]//2)] = 1.0
            p = p * m
        elif ksp.shape[2] == 320:
            x = self.fc2(x)
            acs = torch.zeros_like(x)
            acs[:,152:168] = 2.0
            p = self.sigmoid((x+acs))  #网络估计得到分布参数，其中ACS region概率必为1
            #去掉外周多余的线
            m = torch.zeros_like(p)
            for idx in range(x.shape[0]):
                m[idx,int(160-total_lines[idx]//2):int(160+total_lines[idx]//2)] = 1.0
            p = p * m
        mask = self.topk(p,num_lines[0])
        mask = torch.tile(mask.unsqueeze(-1),[1,1,ksp.shape[-2]])  #原来是1，1，384
        if ksp.shape[0] == 1:
            mask = mask.unsqueeze(0)
        else:
            mask = mask.unsqueeze(1)    
        masked_ksp = torch.view_as_real(torch.view_as_complex(ksp) * mask)
        if self.anatomy_code is not None:
            return masked_ksp,mask,p,anatomy_code
        return masked_ksp,mask,p


#4.8 增加对2D trajectory大小的支持
class SamplingTrajEstimationV4_8(nn.Module):
    def __init__(self,layer_list=[64,128],pattern='Uniform4',**kwargs) -> None:
        super().__init__()
        self.pattern = pattern
        if 'Uniform' in self.pattern:
            input_n =  kwargs.get('input_n')
            self.network = MLP(input_n=input_n,output_n=256,num_layer=len(layer_list),layer_list=layer_list)
            self.fc = nn.Linear(in_features=256,out_features=256)
            self.fc2 = nn.Linear(in_features=256,out_features=320)
            self.sigmoid = nn.Sigmoid()
            self.topk = TopK3.apply
        elif 'Random' in self.pattern:
            input_n =  kwargs.get('input_n')
            self.network = MLP(input_n=input_n,output_n=256,num_layer=len(layer_list),layer_list=layer_list)
            self.fc = nn.Linear(in_features=256,out_features=256*384)
            self.fc2 = nn.Linear(in_features=256,out_features=320*384)
            self.fc3 = nn.Linear(in_features=256,out_features=256*256)
            self.sigmoid = nn.Sigmoid()
            self.topk = TopK3.apply
    def forward(self,ksp,x,num_lines=64,total_lines=128,mode='train',num_points=6000):
        if 'Uniform' in self.pattern:
            x = self.network(x)  #2,256
            if ksp.shape[2] == 256:
                x = self.fc(x)
                acs = torch.zeros_like(x)
                acs[:,120:136] = 10
                p = self.sigmoid((x+acs))  #网络估计得到分布参数，其中ACS region概率必为1
                #去掉外周多余的线
                m = torch.zeros_like(p)
                for idx in range(x.shape[0]):
                    m[idx,int(128-total_lines[idx]//2):int(128+total_lines[idx]//2)] = 1.0
                p = p * m
            elif ksp.shape[2] == 320:
                x = self.fc2(x)
                acs = torch.zeros_like(x)
                acs[:,152:168] = 10
                p = self.sigmoid((x+acs))  #网络估计得到分布参数，其中ACS region概率必为1
                #去掉外周多余的线
                m = torch.zeros_like(p)
                for idx in range(x.shape[0]):
                    m[idx,int(160-total_lines[idx]//2):int(160+total_lines[idx]//2)] = 1.0
                p = p * m
            
            mask = self.topk(p,num_lines[0])
            
            mask = torch.tile(mask.unsqueeze(-1),[1,1,ksp.shape[-2]])  #原来是1，1，384
            if ksp.shape[0] == 1:
                mask = mask.unsqueeze(0)
            else:
                mask = mask.unsqueeze(1)    
            masked_ksp = torch.view_as_real(torch.view_as_complex(ksp) * mask)
        elif 'Random' in self.pattern:
            nb,_,npe,nro,_ = ksp.shape
            x = self.network(x)  #2,256
            if ksp.shape[2] == 256 and ksp.shape[3] == 384:
                x = self.fc(x)
                x = x.reshape(nb,npe,nro)
                acs = torch.zeros_like(x)
                acs[:,128-8:128+8,192-8:192+8] = 10
                p = self.sigmoid((x+acs))  #网络估计得到分布参数，其中ACS region概率必为1
                #去掉外周多余的线
                m = torch.zeros_like(p)
                for idx in range(x.shape[0]):
                    m[idx,int(128-total_lines[idx]//2):int(128+total_lines[idx]//2),:] = 1.0
                p = p * m
            elif npe == 256 and nro == 256:
                x = self.fc3(x)
                x = x.reshape(nb,npe,nro)
                acs = torch.zeros_like(x)
                acs[:,128-8:128+8,128-8:128+8] = 10
                p = self.sigmoid((x+acs))  #网络估计得到分布参数，其中ACS region概率必为1
                #去掉外周多余的线
                m = torch.zeros_like(p)
                for idx in range(x.shape[0]):
                    m[idx,int(128-total_lines[idx]//2):int(128+total_lines[idx]//2),:] = 1.0
                p = p * m
            elif ksp.shape[2] == 320:
                x = self.fc2(x)
                x = x.reshape(nb,npe,nro)
                acs = torch.zeros_like(x)
                acs[:,152:168,192-8:192+8] = 10
                p = self.sigmoid((x+acs))  #网络估计得到分布参数，其中ACS region概率必为1
                #去掉外周多余的线
                m = torch.zeros_like(p)
                for idx in range(x.shape[0]):
                    m[idx,int(160-total_lines[idx]//2):int(160+total_lines[idx]//2),:] = 1.0
                p = p * m
            p = p.reshape(nb,npe*nro)
            mask = self.topk(p,num_points[0])
            mask = mask.reshape(nb,npe,nro)
        #    mask = torch.tile(mask.unsqueeze(-1),[1,1,ksp.shape[-2]])  
            if ksp.shape[0] == 1:
                mask = mask.unsqueeze(0)
            else:
                mask = mask.unsqueeze(1)    
            masked_ksp = torch.view_as_real(torch.view_as_complex(ksp) * mask)
        return masked_ksp,mask


class DisenNetworkV3_5(nn.Module):
    def __init__(
        self,
        in_chans: int=2,
        out_chans: int=2,
        image_size = (256,384)
    ):
        super().__init__()
        content_chans = 4
        #self.content_encoder = Unet(in_chans=in_chans,out_chans=content_chans,chans=8,num_pool_layers=4)
        self.content_encoder_enc = AnatomyEncoderEncNew(in_num_ch=2,first_num_ch=8,is_cond=True)
        self.content_encoder_dec = AnatomyEncoderDecNew(first_num_ch=8,out_num_ch=4,is_cond=True)
        # 功能：将输入图像编码为content map，其size与图像相同，channel数由content_chans确定
        #self.content_denoiser = Unet(in_chans=content_chans,out_chans=content_chans,chans=8,num_pool_layers=4)
        # 功能：对content map进行去噪和去伪影
        self.style_encoder = ModalityEncoderNew(img_num_ch=in_chans,s_num_ch=0,z_size=16,first_num_ch=8,is_cond=True)
        # 功能：将输入图像编码为style code
        self.decoder = UnetCond(in_chans=content_chans,out_chans=out_chans,chans=16,num_pool_layers=4)
        # 功能：用style code和content code组合得到图像
    def sample(self, z_mean, z_log_var):
        eps = torch.normal(0, 1, size=(z_mean.shape[0], z_mean.shape[1])).to(z_mean.device)
        zi = z_mean + eps * torch.exp(0.5 * z_log_var)
        return zi

    def forward(self, image: torch.Tensor,inputs_type=None) -> torch.Tensor:
        cont_latent = self.content_encoder_enc(image,inputs_type=inputs_type)

        sty_1, sty_2 = self.style_encoder(image,si=None,inputs_type=inputs_type)
        #sty = sty_1 + sty_2
        #sty = self.sample(sty_1,sty_2)
        sty = sty_1
        #sty = sty.clamp(-1000,1000)
        cont = self.content_encoder_dec(cont_latent,inputs_type=inputs_type)

        output = self.decoder(cont[0],sty)
        return output,cont[0],sty


if __name__ == '__main__':
    '''
    #style encoder 测试代码
    style_encoder = ModalityEncoderNew(img_num_ch=2,s_num_ch=0,is_cond=True,input_size=(256,384)).cuda()
    input_t = torch.zeros((2,2,256,384)).cuda()
    type_t  = torch.zeros((2,1),dtype=torch.float32).cuda()
    output_t = style_encoder(xi=input_t,si=None,inputs_type=type_t)
    print(output_t[0].shape)  #2,16
    print(output_t[1].shape)  #2,16
    '''
    '''
    #decoder 测试代码
    decoder = SPADENew(image_size=(256,384),in_num_ch=2,z_size=16,is_cond=True,s_num_ch=4).cuda()
    input_t = torch.zeros((2,4,256,384)).cuda()
    s_t = torch.zeros((2,16)).cuda()
    type_t  = torch.zeros((2,1),dtype=torch.float32).cuda()
    output_t,_ = decoder(input_t,s_t,type_t)
    print(output_t.shape)  #2,256,384
    '''
    #模块整体测试代码
    net = DisenNetwork().cuda()
    input_t = torch.zeros((2,2,256,384)).cuda()
    type_t  = torch.zeros((2,1),dtype=torch.float32).cuda()
    res,cont,sty = net(input_t,type_t)
    print("Done")