import fusion_config
from layers import DenseBlock, TransitionDown, Bottleneck, TransitionUp
from utils import *
from torchvision import models


# VGG+Densenet -ConvLSTM混合网络模型

def generate_model(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    assert args.model in ['Densenet-ConvLSTM']
    model = Vgg_Dense_Convlstm_Net67(fusion_config.class_num, fusion_config.growth_rate).to(device)
    return model
'''
args:
    in_channels: 输入的通道数
    down_blocks: encoder过程中每个denseblock的层数
    up_blocks:   decoder过程中每个denseblock的层数
    bottleneck_layers: encoder与decoder中间一个denseblock的层数
    growth_rate: 增长率
    out_chans_first_conv: 第一层卷积的卷积核个数
    n_classes: 类别数
'''

# 定义Dense_net网络


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12, fusion_channels_count=960):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.fusion_channels_count = fusion_channels_count
        cur_channels_count = 0
        skip_connection_channel_counts = []

        # 第一层卷积：3x3 48个卷积核
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_chans_first_conv, kernel_size=3,
                                               stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        # encoder

        # dense Block
        self.denseBlocksDown = nn.ModuleList([])
        # Transition Down
        self.transDownBlocks = nn.ModuleList([])

        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate * down_blocks[i])

            # 每一个denseblock后面都有一个skip connection
            skip_connection_channel_counts.insert(0, cur_channels_count)

            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        # Bottleneck

        self.add_module('bottleneck', Bottleneck(self.fusion_channels_count,
                                                 growth_rate, bottleneck_layers))

        # transition up 只包含上一个denseblock的输出，不包含上一个denseblock的输入（即x）
        prev_block_channels = growth_rate * bottleneck_layers

        cur_channels_count += prev_block_channels

        # decoder

        # transition up
        self.transUpBlocks = nn.ModuleList([])
        # denseblock
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)):
            # transition up
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            # dense block up, 此block的输入为transition up的输出+对应的encoder过程的skip connection
            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            # cur_channels_count += prev_block_channels

        cur_channels_count = growth_rate * up_blocks[-1]
        self.convlstm = ConvLSTM(input_size=(3, 12),
                                 input_dim=80,
                                 hidden_dim=[80, 80],
                                 kernel_size=(3, 3),
                                 num_layers=2,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)
        ## Softmax
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
    #  Vgg16 编码器 处理点云数据
        vgg16 = models.vgg16_bn(pretrained=False)
        pthfile = './pretrained/vgg16_bn-6c64b313.pth'
        vgg16.load_state_dict(torch.load(pthfile))
        self.vgg16_bn = vgg16.features

        self.relu = nn.ReLU(inplace=True)
        self.index_MaxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.index_UnPool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # net struct
        self.conv1_block = nn.Sequential(self.vgg16_bn[0],  # conv2d(3,64,(3,3))
                                         self.vgg16_bn[1],  # bn(64,eps=1e-05,momentum=0.1,affine=True)
                                         self.vgg16_bn[2],  # relu(in_place)
                                         self.vgg16_bn[3],  # conv2d(3,64,(3,3))
                                         self.vgg16_bn[4],  # bn(64,eps=1e-05,momentum=0.1,affine=True)
                                         self.vgg16_bn[5]  # relu(in_place)
                                         )
        self.conv2_block = nn.Sequential(self.vgg16_bn[7],
                                         self.vgg16_bn[8],
                                         self.vgg16_bn[9],
                                         self.vgg16_bn[10],
                                         self.vgg16_bn[11],
                                         self.vgg16_bn[12]
                                         )
        self.conv3_block = nn.Sequential(self.vgg16_bn[14],
                                         self.vgg16_bn[15],
                                         self.vgg16_bn[16],
                                         self.vgg16_bn[17],
                                         self.vgg16_bn[18],
                                         self.vgg16_bn[19],
                                         self.vgg16_bn[20],
                                         self.vgg16_bn[21],
                                         self.vgg16_bn[22]
                                         )
        self.conv4_block = nn.Sequential(self.vgg16_bn[24],
                                         self.vgg16_bn[25],
                                         self.vgg16_bn[26],
                                         self.vgg16_bn[27],
                                         self.vgg16_bn[28],
                                         self.vgg16_bn[29],
                                         self.vgg16_bn[30],
                                         self.vgg16_bn[31],
                                         self.vgg16_bn[32]
                                         )
        self.conv5_block = nn.Sequential(self.vgg16_bn[34],
                                         self.vgg16_bn[35],
                                         self.vgg16_bn[36],
                                         self.vgg16_bn[37],
                                         self.vgg16_bn[38],
                                         self.vgg16_bn[39],
                                         self.vgg16_bn[40],
                                         self.vgg16_bn[41],
                                         self.vgg16_bn[42]
                                         )

    # 前向传播
    def forward(self, x):
        x_image = x['image']
        x_cloud = x['point_cloud_image']
        x_image = torch.unbind(x_image, dim=1)
        x_cloud = torch.unbind(x_cloud, dim=1)


        # test 点云融合效果

        for item in range(len(x_image)):
            out_image = self.firstconv(x_image[item])
            skip_connections = []
            # print(x_cloud[item].shape)
            # exit()
            f1, idx1 = self.index_MaxPool(self.conv1_block(x_cloud[item]))
            f2, idx2 = self.index_MaxPool(self.conv2_block(f1))
            f3, idx3 = self.index_MaxPool(self.conv3_block(f2))
            f4, idx4 = self.index_MaxPool(self.conv4_block(f3))
            f5, idx5 = self.index_MaxPool(self.conv5_block(f4))
            idx1 = torch.as_tensor(idx1, dtype=torch.float32).cuda()
            idx2 = torch.as_tensor(idx2, dtype=torch.float32).cuda()
            idx3 = torch.as_tensor(idx3, dtype=torch.float32).cuda()
            idx4 = torch.as_tensor(idx4, dtype=torch.float32).cuda()
            idx5 = torch.as_tensor(idx5, dtype=torch.float32).cuda()
            out_cloud_fusion = idx1, idx2, idx3, idx4, idx5
            for i in range(len(self.down_blocks)):
                out_image = self.denseBlocksDown[i](out_image)
                # 现在传递给解码器的是图像数据浅层特征 如有需要也可也传点云
                skip_connections.append(out_image)

                # 特征图缩小一半
                out_image = self.transDownBlocks[i](out_image)

                # 融合点云和图像数据通道上
                out = torch.cat([out_image, out_cloud_fusion[i]], dim=1)


            out = self.bottleneck(out)

        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = self.softmax(out)
        return out


# FCDenseNet67

def Vgg_Dense_Convlstm_Net67(n_classes, growth_rate=16):
    return FCDenseNet(
        in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=growth_rate, out_chans_first_conv=48, n_classes=n_classes)




