import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src




### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, compress_rate, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        mid_ch1 = int((1-compress_rate[0])*mid_ch)
        mid_ch2 = int((1-compress_rate[1])*mid_ch)
        mid_ch3 = int((1-compress_rate[2])*mid_ch)
        mid_ch4 = int((1-compress_rate[3])*mid_ch)
        mid_ch5 = int((1-compress_rate[4])*mid_ch)
        mid_ch6 = mid_ch
        if mid_ch1<1:
            mid_ch1=1
        if mid_ch2<1:
            mid_ch2=1
        if mid_ch3<1:
            mid_ch3=1
        if mid_ch4<1:
            mid_ch4=1
        if mid_ch5<1:
            mid_ch5=1
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch1,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch1,mid_ch2,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch2,mid_ch3,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch3,mid_ch4,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch4,mid_ch5,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch5,mid_ch6,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch6,mid_ch6,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch6*2,mid_ch5,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch5*2,mid_ch4,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch4*2,mid_ch3,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch3*2,mid_ch2,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch2*2,mid_ch1,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch1*2,out_ch,dirate=1)


    def forward(self,x):

        hx = x#1,3,288,288
        hxin = self.rebnconvin(hx)#1,64,288,288

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)
        # cat_7_6 = torch.cat(hx7,hx6)#提取这里的feature map确定mid_ch6*2

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)


        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        #提取这里的feature map确定mid_ch3*2

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        #提取这里的feature map确定mid_ch2*2


        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        #提取这里的feature map确定mid_ch1*2
        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self,compress_rate, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()
        mid_ch1 = int((1-compress_rate[0])*mid_ch)
        mid_ch2 = int((1-compress_rate[1])*mid_ch)
        mid_ch3 = int((1-compress_rate[2])*mid_ch)
        mid_ch4 = int((1-compress_rate[3])*mid_ch)
        mid_ch5 = mid_ch
        if mid_ch1<1:
            mid_ch1=1
        if mid_ch2<1:
            mid_ch2=1
        if mid_ch3<1:
            mid_ch3=1
        if mid_ch4<1:
            mid_ch4=1

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch1,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch1,mid_ch2,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch2,mid_ch3,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch3,mid_ch4,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch4,mid_ch5,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch5,mid_ch5,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch5*2,mid_ch4,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch4*2,mid_ch3,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch3*2,mid_ch2,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch2*2,mid_ch1,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch1*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self,compress_rate,in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()
        mid_ch1 = int((1-compress_rate[0])*mid_ch)
        mid_ch2 = int((1-compress_rate[1])*mid_ch)
        mid_ch3 = int((1-compress_rate[2])*mid_ch)
        mid_ch4 = mid_ch

        if mid_ch1<1:
            mid_ch1=1
        if mid_ch2<1:
            mid_ch2=1
        if mid_ch3<1:
            mid_ch3=1

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch1,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch1,mid_ch2,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch2,mid_ch3,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch3,mid_ch4,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch4,mid_ch4,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch4*2,mid_ch3,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch3*2,mid_ch2,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch2*2,mid_ch1,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch1*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, compress_rate,in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()
        mid_ch1 = int((1-compress_rate[0])*mid_ch)
        mid_ch2 = int((1-compress_rate[1])*mid_ch)
        mid_ch3 = mid_ch

        if mid_ch1<1:
            mid_ch1=1
        if mid_ch2<1:
            mid_ch2=1

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch1,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch1,mid_ch2,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch2,mid_ch3,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch3,mid_ch3,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch3*2,mid_ch2,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch2*2,mid_ch1,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch1*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self,compress_rate, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()
        mid_ch1 = int((1-compress_rate[0])*mid_ch)
        mid_ch2 = int((1-compress_rate[1])*mid_ch)
        mid_ch3 = mid_ch

        if mid_ch1<1:
            mid_ch1=1
        if mid_ch2<1:
            mid_ch2=1

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch1,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch1,mid_ch2,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch2,mid_ch3,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch3,mid_ch3,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch3*2,mid_ch2,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch2*2,mid_ch1,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch1*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


def adapt_channel(compress_rate):

    stage1_inter = compress_rate[:5]
    stage2_inter = compress_rate[5:9]
    stage3_inter = compress_rate[9:12]
    stage4_inter = compress_rate[12:14]
    stage5_inter = compress_rate[14:16]
    stage6_inter = compress_rate[16:18]

    stage1d_inter = compress_rate[18:23]
    stage2d_inter = compress_rate[23:27]
    stage3d_inter = compress_rate[27:30]
    stage4d_inter = compress_rate[30:32]
    stage5d_inter = compress_rate[32:34]

    exter = compress_rate[34:39]

    stage_en, stage_de, exter = [[stage1_inter, stage2_inter, 
                                stage3_inter, stage4_inter, 
                                stage5_inter, stage6_inter], 
                                [stage1d_inter, stage2d_inter, 
                                stage3d_inter, stage4d_inter, stage5d_inter],
                                [exter]]

    return stage_en, stage_de, exter


### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self,in_ch=3,out_ch=1,compress_rate=[0.]*100):
        super(U2NETP,self).__init__()
        self.best_loss = 999999

        stage_en, stage_de, exter = adapt_channel(compress_rate)

        compress_rate_stage1_inter = stage_en[0]
        compress_rate_stage2_inter = stage_en[1]
        compress_rate_stage3_inter = stage_en[2]
        compress_rate_stage4_inter = stage_en[3]
        compress_rate_stage5_inter = stage_en[4]
        compress_rate_stage6_inter = stage_en[5]

        compress_rate_stage1d_inter = stage_de[0]
        compress_rate_stage2d_inter = stage_de[1]
        compress_rate_stage3d_inter = stage_de[2]
        compress_rate_stage4d_inter = stage_de[3]
        compress_rate_stage5d_inter = stage_de[4]

        compress_rate_exter = exter[0]

        mid_ch=64
        mid_ch1 = int((1-compress_rate_exter[0])*mid_ch)
        mid_ch2 = int((1-compress_rate_exter[1])*mid_ch)
        mid_ch3 = int((1-compress_rate_exter[2])*mid_ch)
        mid_ch4 = int((1-compress_rate_exter[3])*mid_ch)
        mid_ch5 = int((1-compress_rate_exter[4])*mid_ch)

        if mid_ch1<1:
            mid_ch1=1
        if mid_ch2<1:
            mid_ch2=1
        if mid_ch3<1:
            mid_ch3=1
        if mid_ch4<1:
            mid_ch4=1
        if mid_ch5<1:
            mid_ch5=1

        self.stage1 = RSU7(compress_rate_stage1_inter,in_ch,16,mid_ch1)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(compress_rate_stage2_inter,mid_ch1,16,mid_ch2)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(compress_rate_stage3_inter,mid_ch2,16,mid_ch3)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(compress_rate_stage4_inter,mid_ch3,16,mid_ch4)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(compress_rate_stage5_inter,mid_ch4,16,mid_ch5)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(compress_rate_stage6_inter,mid_ch5,16,mid_ch5)

        # decoder
        self.stage5d = RSU4F(compress_rate_stage5d_inter,mid_ch5*2,16,mid_ch4)
        self.stage4d = RSU4(compress_rate_stage4d_inter,mid_ch4*2,16,mid_ch3)
        self.stage3d = RSU5(compress_rate_stage3d_inter,mid_ch3*2,16,mid_ch2)
        self.stage2d = RSU6(compress_rate_stage2d_inter,mid_ch2*2,16,mid_ch1)
        self.stage1d = RSU7(compress_rate_stage1d_inter,mid_ch1*2,16,mid_ch1)

        self.side1 = nn.Conv2d(mid_ch1,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(mid_ch1,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(mid_ch2,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(mid_ch3,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(mid_ch4,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(mid_ch5,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6,out_ch,1)

    def forward(self,x):

        hx = x#1,3,288,288

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
