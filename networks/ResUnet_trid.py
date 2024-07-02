from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils import prune
from networks.unet import UnetBlock
import torch.nn.functional as F
import torch
from torch import nn
from networks.fetureweight import extract_features, ACBBlock, DepthwiseSeparableConv
from networks.MBpool import MBPOOL
from networks.resnest import resnest50
import torch.nn.functional as F
from networks.vit import VisionTransformer
import random
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from networks.Rmcsam import RMCSAM, RMCSAM_CBAM
from networks.mixstyle_kernel import TriD,MixStyle,EFDMix
from networks.ChannelAttention import ChannelAttention, SpatialAttention, AFF,iAFF,FAFF,MultiModalFusionModule
import torch.optim as optim
class PolicyNetwork(nn.Module):
    def __init__(self, num_actions, num_states):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(num_states, num_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        return self.softmax(self.fc(state))

# 创建策略网络和优化器
num_actions = 2  # 假设有两个数据增强方式
num_states = 10  # 假设有10个状态特征
policy_network = PolicyNetwork(num_actions, num_states)
optimizer = optim.Adam(policy_network.parameters(), lr=0.001)


class ResnetModel(nn.Module):
    """ This is a subclass from nn.Module that creates a pre-trained resnet50 model.

    Attributes:
        cnn: The convolutional network, a pretrained resnet50.
        fc_meta_data: A fully connected network.
        classifier: The last layer in the network.
    """

    def __init__(self, resnet='resnest50',num_classes=2, pretrained=False, mixstyle_layers=[],random_type=None, p=0.5):
        """ The __init__ function.
        Args:
            n_columns: the number of columns in data (the meta data).
        """
        super(ResnetModel, self).__init__()
        
        self.mixstyle_layers = mixstyle_layers
       
        self.num_classes = num_classes

        self.p = p

        if mixstyle_layers:
            if random_type == 'TriD':#ACC:0.8525 AUC:0.8773
                self.random = TriD(p=p)
            elif random_type == 'MixStyle':
                self.random = MixStyle(p=p, mix='random')
            elif random_type == 'EFDMixStyle':
                self.random = EFDMix(p=p, mix='random')
            else:
                raise ValueError('The random method type is wrong!')
            print(random_type)
            print('Insert Random Style after the following layers: {}'.format(mixstyle_layers))
        # res2net预训练好的模型
        # self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # resnext预训练好的模型
        image = 3
        self.weight = extract_features(image)
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv_feature1 = nn.Conv2d(18, 1, kernel_size=1, stride=1, padding=1,
                                       bias=False)
        self.conv_feature2 = nn.Conv2d(18, 1, kernel_size=3, stride=1, padding=2,
                                       bias=False)
        self.conv_feature3 = nn.Conv2d(18, 1, kernel_size=5, stride=1, padding=3,
                                       bias=False)
        self.resnet = resnest50(pretrained=True)

        # cv2.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None)
        # self.resize = transforms.Resize([224,224])
        self.conv4_1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=1,
                               bias=False)
        self.conv4_2 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=2,
                                 bias=False)
        self.conv4_3 = nn.Conv2d(256, 64, kernel_size=5, stride=1, padding=3,
                                 bias=False)
        self.acb1 = ACBBlock(128, 64)
        self.bn4 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.conv5_1 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=1,
                               bias=False)
        self.conv5_2 = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=2,
                               bias=False)
        self.conv5_3 = nn.Conv2d(512, 64, kernel_size=5, stride=1, padding=3,
                               bias=False)
        self.acb2 = ACBBlock(64, 64)
        self.bn5 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv6_1 = nn.Conv2d(1024, 64, kernel_size=1, stride=2, padding=1,
                               bias=False)
        self.conv6_2 = nn.Conv2d(1024, 64, kernel_size=3, stride=2, padding=2,
                               bias=False)
        self.conv6_3 = nn.Conv2d(1024, 64, kernel_size=5, stride=2, padding=3,
                               bias=False)
        
        self.bn6 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.avg6 = nn.AvgPool2d(3, stride=2)
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = norm_layer(1024)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3 = norm_layer(512)
        self.conv3_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=1,
                                 bias=False)
        self.bn3_1 = norm_layer(64)
        self.conv3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        self.bn3_2 = norm_layer(128)
        self.conv3_3 = nn.Conv2d(128, 16, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        self.bn3_3 = norm_layer(16)
        self.relu = nn.ReLU(inplace=True)
        self.trans1 = VisionTransformer()
        self.conv1_1= nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=1,  # kernel_size=7, stride=2, padding=3,

                             bias=False)
        self.conv1_2= nn.Conv2d(2048, 64, kernel_size=3, stride=1, padding=2,  # kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1_3 = nn.Conv2d(2048, 64, kernel_size=5, stride=1, padding=3,  # kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.acb4 = ACBBlock(1024, 64)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(2048, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn7 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn8 = norm_layer(3)
        # self.relu = nn.ReLU(inplace=True)
        self.inplanes = 64
        self.k_size = 3

        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()
        # self.inplanes2 = 256
        # self.k_size = 3
        #
        # self.ca2 = ChannelAttention(self.inplanes2)
        # self.sa2 = SpatialAttention()
        # self.inplanes3 = 512
        # self.ca3 = ChannelAttention(self.inplanes3)
        # self.sa3 = SpatialAttention()
        # self.inplanes4 = 1024
        # self.ca4 = ChannelAttention(self.inplanes4)
        # self.sa4 = SpatialAttention()
        # self.inplanes1 = 2048
        # self.ca1 = ChannelAttention(self.inplanes1)
        # self.sa1 = SpatialAttention()
        self.aff = AFF()
        self.iaff = iAFF()
        self.faff = FAFF()

        self.feature_dim_1 = 64
        self.RMCSAM_1 = RMCSAM(self.feature_dim_1)
        self.feature_dim_2 = 128
        self.RMCSAM_2 = RMCSAM(self.feature_dim_2)
        self.feature_dim_3 = 256
        self.RMCSAM_3 = RMCSAM(self.feature_dim_3)
        self.feature_dim_4 = 512
        self.RMCSAM_4 = RMCSAM(self.feature_dim_4)
        self.feature_dim = 64
        self.RMCSAM = RMCSAM(self.feature_dim)  # (先点这一行最前面输入#)
        # self.RMCSAM = RMCSAM_CBAM(self.feature_dim)  #(调整一下这一行，删除这一行最前面的#)
        self.fc1 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(p=0.4)  # dropout训练
        self.fc2 = nn.Linear(4288, 4096)
        num_channels = 64
        self.MB1 = MBPOOL(num_channels)
        self.MC = MultiModalFusionModule(num_channels)

        self.depth = DepthwiseSeparableConv(192, 64, 3, 1, 1)
        self.depth1 = DepthwiseSeparableConv(192,64,3,1,1)
        self.depth2 = DepthwiseSeparableConv(256, 64, 3, 1, 1)
        self.depth3 = DepthwiseSeparableConv(512, 64, 3, 1, 1)
        self.depth4 = DepthwiseSeparableConv(1024, 64, 3, 1, 1)
        # self.MB2 = MBPOOL(num_channels)
        # self.MB3 = MBPOOL(num_channels)
        # self.MB4 = MBPOOL(num_channels)
        # self.fc3 = nn.Linear(1000, 512)

        # self.classifier = nn.Sequential(nn.Linear(1000 + 250, 1))
        # self.fc = nn.Linear(16384, 256)
        # self.reduce_dim = nn.Linear(335872, 1024)
        self.classifier = nn.Sequential(nn.Linear(335872, 1), nn.Dropout(0.5))#221952,3504384,avg:861184
        # self.z = nn.BatchNorm1d(num_features=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

        # self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        #
        # self.trans_conv = nn.ConvTranspose2d(192, 64, kernel_size=4, stride=4, padding=0)

    # def forward(self, image, data):
    def forward(self, x,is_train):
        # 把torch.size(4,4,3,240,240)变成(4,3,240,240)
        # x = x[:, 0, 0, :, :, :]


        # print(x.size())
        # x = torch.tensor(x)
        # x = self.weight(x)
        # x = torch.tensor(x)
        # x = x.to(device)
        # xf1 = self.conv_feature1(x)
        # xf2 = self.conv_feature2(x)
        # xf3 = self.conv_feature3(x)  # x.size[4, 64, 320, 320]
        # print(xf1.size())
        # print(xf2.size())
        # print(xf3.size())
        # x= torch.cat((xf1,xf2,xf3), dim=1)
        # x = torch.add(x, xf3)
        # print('x')
        # print('x.size()75',x.size())
        sfs = []
        if random.random() > 0.5:
            data_aug = True
        else:
            data_aug = False
        # data_aug = True
        # # max_x1 = torch.max(x)
        # # print(max_x1)
        # mix1 = None
        if is_train:
            if 'layer0' in self.mixstyle_layers and data_aug ==True:
                mix1 = self.random(x)
                # print("mix1", mix1)
                sfs.append(mix1)
                # z=self.conv(x)
                x1 = self.resnet.conv1(mix1)
            else:
                x1 = self.resnet.conv1(x)
        #
        #
        else:
            sfs=[]
            mix1=None
            x1 = self.resnet.conv1(x)
        # x1 = self.resnet.conv1(x)
        x = self.resnet.bn1(x1)  # x.size()78 torch.Size([4, 64, 320, 320])
        x = self.resnet.relu(x)

        x1 = self.resnet.maxpool(x)  # x1.size()83 torch.Size([4, 64, 160, 160])

        if is_train:
            if 'layer0' in self.mixstyle_layers and data_aug == False:
                mix2 = self.random(x1)
                # print("mix2",mix2)
                sfs.append(mix2)
                x2 = self.resnet.layer1(mix2)
            else:
                x2 = self.resnet.layer1(x1)
        else:
            sfs=[]
            mix2=None
            x2 = self.resnet.layer1(x1)

        x3 = self.resnet.layer2(x2)  # x3.size()88 torch.Size([4, 512, 80, 80])

        x4 = self.resnet.layer3(x3)  # x4.size()90 torch.Size([4, 1024, 40, 40])

        x5 = self.resnet.layer4(x4)  # x5.size()92 torch.Size([4, 2048, 20, 20])


        # 多尺度拼接特征
        # y2_1 = self.conv4_1(x2)
        y2_2 = self.conv4_2(x2)
        # y2_3 = self.conv4_3(x2)
        #
        y2 = self.relu(y2_2)  # size(4,64,60,60)
        y2 = self.bn4(y2)


        y2 = self.ca(y2) * y2
        y2 = self.sa(y2) * y2
        y2 = self.avg6(y2)
        print('y2conv shape:', y2.size())
        y2 = torch.flatten(y2, 1)

        y2 = self.dropout(y2)



        y3_2 = self.conv5_2(x3)

        y3 = self.relu(y3_2)  # size(4,64,30,30)
        y3 = self.bn5(y3)

        y3 = self.ca(y3) * y3
        y3 = self.sa(y3) * y3
        y3 = self.avg6(y3)
        print('y3conv shape:', y3.size())
        y3 = torch.flatten(y3, 1)

        y3 = self.dropout(y3)

        y4_2 = self.conv6_2(x4)


        y4 = self.relu(y4_2)  # size(4,64,8,8)
        y4 = self.bn6(y4)

        y4 = self.ca(y4) * y4
        y4 = self.sa(y4) * y4
        y4 = self.avg6(y4)
        print('y4conv shape:', y4.size())
        y4 = torch.flatten(y4, 1)

        y4 = self.dropout(y4)



        image_2 = self.conv1_2(x5)
        # image_3 = self.conv1_3(x5)
        #
        #
        image = self.relu(image_2)
        image = self.bn1(image)

        image = self.ca(image) * image
        image = self.sa(image) * image
        image = self.avg6(image)
        print('imageconv shape:', image.size())
        image = torch.flatten(image, 1)

        image = self.dropout(image)

#按照插值操作来拼接
        # y2 = F.interpolate(y2, size=(8, 8), mode='bilinear', align_corners=False)
        # #
        # image = torch.nn.functional.interpolate(image, size=(28, 28), mode='bilinear',align_corners=False)
        # y3 = torch.nn.functional.interpolate(y3, size=(28, 28), mode='bilinear', align_corners=False)
        # y4 = torch.nn.functional.interpolate(y4, size=(28, 28), mode='bilinear',align_corners=False)
        #
        # y2 = y2[:, :4096]
        # y3 = y3[:, :4096]
        # y4 = y4[:, :4096]

        # print("查看特征尺寸",y2.size(),y3.size(),y4.size(),image.size())
        classtoken1 = torch.cat([y2, image], dim=1)
        classtoken2 = torch.cat([classtoken1, y3], 1)
        classtoken3 = torch.cat([classtoken2, y4], 1)
        # x = self.classifier(F.adaptive_avg_pool2d(classtoken3, (1, 1)).squeeze())

        classtoken3 = torch.flatten(classtoken3, 1)
        print('classtoken3 size',classtoken3.size())
        # y_concat = torch.cat((y2, y3, y4, image), dim=1)
        # y_concat = self.fc(y_concat)
        # y_concat=self.reduce_dim(y_concat)
        x = self.classifier(classtoken3)
        # prune.l1_unstructured(linear, name='weight', amount=0.5)
        # if data_aug == True:
        #     output= mix1
        # else:
        output= classtoken3
        return x,output
