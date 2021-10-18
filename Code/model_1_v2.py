#-*- coding = utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pickle
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
import pickle
from torch.utils.data import DataLoader
import numpy as np
import resnet_model
import sub_attention

class AudioNet_v2_pooling(nn.Module):
    def __init__(self,dim):
        super(AudioNet_v2_pooling, self).__init__()
        sub_attention.init_gobal_variable(a=512, b=512, c=64, d=8)
        print("AudioNet_v2_pooling")
        self.encoder1 = sub_attention.Encoder(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 7), stride=(3, 4), padding=(0,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 5), stride=(3, 3), padding=(0,0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 5), stride=(1, 3), padding=(0,0)),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, dim, kernel_size=(1, 5), stride=(1, 3), padding=(0,0)),
            nn.BatchNorm2d(dim),
            nn.ReLU())

    def forward(self, x):
        # print(x.size())
        # [3, 96, 512]
        x = self.encoder1(x)
        # [3, 96, 512]
        x = x.unsqueeze(1)
        # [3, 1, 96, 512]
        x = self.conv1(x)
        # [3, 64, 32, 170]
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.mean(x,dim=-1).transpose(1,2)
        return x

class VideoNet(nn.Module):

    def __init__(self,dim=1024):
        super(VideoNet, self).__init__()
        self.dim = dim
        self.fc = nn.Sequential(
                    nn.Linear(2048,dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(dim)
                )
        self.resnet = resnet_model.resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
    def forward(self,x):
        batch = x.size(0)
        x = self.resnet(x)
        x = self.avgpool(x)+self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(int(batch / 10), 10, self.dim)
        return x


class Mixed_model1(nn.Module):

    def __init__(self,dim=1024):
        super(Mixed_model1, self).__init__()

        self.video = VideoNet(dim=dim)
        self.audio_net = AudioNet_v2_pooling(dim=dim)

        # para = sum([np.prod(list(p.size())) for p in self.video.parameters()])
        # print('video num of paras:', para)
        # print('Model {} : params: {}'.format(self.video._get_name(), para))
        #
        # para = sum([np.prod(list(p.size())) for p in self.audio_net.parameters()])
        # print('audio_net num of paras:', para)
        # print('Model {} : params: {}'.format(self.audio_net._get_name(), para))

        # video num of paras: 26082088
        # Model VideoNet : params: 26082088
        # audio_net num of paras: 2113024
        # Model AudioNet_v2_pooling : params: 2113024


        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dim , 10)
        )
    def forward(self,img,audio):
        batch = img.size(0)
        audio = self.audio_net(audio)
        img = img.view(10*batch,3,224,224)
        img = self.video(img)
        mixed = torch.cat([audio,img],dim=-1)
        result = self.classifier(mixed)
        result = torch.mean(result, dim=1)+torch.max(result,dim=1)[0]
        return result


if __name__ == '__main__':
    # model = AudioNet()
    # input = torch.rand(3,96,512)
    # model = VideoNet()
    # input = torch.rand(30,3,224,224)


    # img = model(input)
    # print(img.size())

    model = Mixed_model1(1024)
    x = torch.rand(3,10, 3, 224, 224)
    y = torch.rand(3,96,512)
    img = model(x,y)
    print(img.size())