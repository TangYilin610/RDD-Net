# coding:utf-8
import os

import cv2
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
# from sklearn.metrics import accuracy_score
from networks.ResUnet_trid import ResnetModel
from utils.metrics import calculate_metrics
import sys
sys.path.append('./torch-cam-main')
from config import *
from torchnet import meter

class Test:
    def __init__(self, config, test_loader):
        # 数据加载
        self.test_loader = test_loader

        # 模型
        self.model = None
        self.backbone = config.backbone
        self.model_type = config.model_type

        # 路径设置
        self.target = config.Target_Dataset
        self.result_path = config.result_path
        self.model_path = config.model_path
        self.seg_cost = Seg_loss()
        # 其他
        self.out_ch = config.out_ch
        self.image_size = config.image_size
        self.mode = config.mode
        self.device = config.device

        self.build_model()
        self.print_network(self.model)

    def build_model(self):
        if self.model_type == 'Res_Unet':
            self.model = ResnetModel(resnet=self.backbone, num_classes=self.out_ch, pretrained=True,
                                 mixstyle_layers=[])
        else:
            raise ValueError('The model type is wrong!')
        #
        # checkpoint = torch.load('./models/20240520_005459_490529/best-Res_Unet.pth')
        checkpoint = torch.load(self.model_path + '/' + 'best' + '-' + self.model_type + '.pth',
                                map_location=lambda storage, loc: storage.cuda(0))
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        print("The number of parameters: {}".format(num_params))

    def test(self):
        print("Testing and Saving the results... Domain Generalization Phase")
        print("--" * 15)
        metrics_y = [[], []]
        y_all_train, y_pred_all_train = [], []
        metric_dict = ['ACC', 'AUC']
        loss_meter = meter.AverageValueMeter()
        with torch.no_grad():
            # y_all_test = []
            # y_pred_all_test = []
            # activation_maps = []
            loss_meter.reset()
            for batch, data in enumerate(self.test_loader):
                x, y, path = data['data'], data['mask'], data['name']
                x = torch.from_numpy(x).to(dtype=torch.float32)
                y = torch.from_numpy(y).to(dtype=torch.float32)

                x, y = x.to(self.device), y.to(self.device)
                # seg_logit = self.model(x, is_train=False)
                seg_logit,feature = self.model(x, is_train=False)
                # print("y shape:",y.shape)
                # loss = self.seg_cost(seg_logit, y.float()) #使用AVGPOOL
                loss = self.seg_cost(seg_logit, y.view(-1, 1).float()) #使用flatten
                loss_meter.add(loss.sum().item())
                # net = self.model(x,is_train=False)
                # pthfile = r'F:/tangyilin/TriD-main/TriD-master/OPTIC/models/20231226_122508_326560/best-Res_Unet.pth'
                # self.model.load_state_dict(torch.load(pthfile))
                # finalconv_name = 'features'


                # features_blobs = []  # 后面用于存放特征图
                #
                # def hook_feature(module, input, output):
                #     features_blobs.append(output.data.cpu().numpy())
                #
                # # 获取 features 模块的输出
                # self.model._modules.get(feature).register_forward_hook(hook_feature)

                # net_name = []
                # params = []
                # for name, param in net.named_parameters():
                #     net_name.append(name)
                #     params.append(param)
                # print(net_name[-1], net_name[-2])  # classifier.1.bias classifier.1.weight
                # print(len(params))  # 52
                # weight_softmax = np.squeeze(params[-2].data.numpy())
                #
                # CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
                # 融合类激活图和原始图片
                # img = cv2.imread(x)
                # height, width, _ = img.shape
                # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
                # result = heatmap * 0.3 + img * 0.7
                # cv2.imwrite('CAM0.jpg', result)
                #
                #
                # def returnCAM(feature_conv, weight_softmax, class_idx):
                #     # 类激活图上采样到 256 x 256
                #     size_upsample = (256, 256)
                #     bz, nc, h, w = feature_conv.shape
                #     output_cam = []
                #     # 将权重赋给卷积层：这里的weigh_softmax.shape为(1000, 512)
                #     # 				feature_conv.shape为(1, 512, 13, 13)
                #     # weight_softmax[class_idx]由于只选择了一个类别的权重，所以为(1, 512)
                #     # feature_conv.reshape((nc, h * w))后feature_conv.shape为(512, 169)
                #     cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
                #     print(cam.shape)  # 矩阵乘法之后，为各个特征通道赋值。输出shape为（1，169）
                #     cam = cam.reshape(h, w)  # 得到单张特征图
                #     # 特征图上所有元素归一化到 0-1
                #     cam_img = (cam - cam.min()) / (cam.max() - cam.min())
                #     # 再将元素更改到　0-255
                #     cam_img = np.uint8(255 * cam_img)
                #     output_cam.append(cv2.resize(cam_img, size_upsample))
                #     return output_cam
                #
                #
                #
                # # Replace the SmoothGradCAMpp logic with custom SmoothGradCAMpp function
                # activation_map = smooth_grad_campp(self.model, x)
                # activation_maps.append(activation_map)

                seg_output = torch.sigmoid(seg_logit)
                #
                y_all_train.append(y.cpu().detach().numpy())
                y_pred_all_train.append(seg_output.detach().cpu().numpy())

                #
                # cam_np = activation_map.squeeze().cpu().numpy()

                # Visualize CAM on the input image
                # input_image_np = x.squeeze().cpu().numpy().transpose((1, 2, 0))  # Assuming channels-last format
                # plt.imshow(input_image_np)
                # plt.imshow(cam_np, alpha=0.6, cmap='jet')
                # plt.axis('off')
                #
                # # Save the visualization as an image file
                # plt.savefig(f'cam_visualization_{batch}.png')  # Modify the filename as needed
                # plt.close()  # Close the plot to release memory
        # print('输入y_all_train：',y_all_train  )
        # print('输入y_pred_all_train：', y_pred_all_train)



        print("Test ———— Total_Loss:{:.8f}".format(loss_meter.value()[0]))
        # print("Test ———— Total_Loss:",loss)
        acc = calculate_classification_metrics1(y_pred_all_train,y_all_train )
            # print('y_pred_all_train:', y_pred_all_train)
            # print('y_all_train:', y_all_train)

        auc = calculate_classification_metrics2(y_pred_all_train,y_all_train)
        # print('精度auc,acc', auc,acc)

        metrics_y[0].append(acc)
        metrics_y[1].append(auc)

                # for i in range(len(metrics)):
                #     metrics_y[i].append(metrics[i])

                # draw_output = (seg_output.detach().cpu().numpy() * 255).astype(np.uint8)
                #
                # img_name = path[0].split('/')[-2] + '-' + path[0].split('/')[-1].split('.')[0]
                # Optic Cup
                # cv2.imwrite(self.result_path + '/' + img_name + '_OC.png', draw_output[0][1])
                # Optic Disc
                # cv2.imwrite(self.result_path + '/' + img_name + '_OD.png', draw_output[0][0])
        metrics_y_array = np.array(metrics_y)
        # print('metrics_y_array:',metrics_y_array)
        # non_empty_lists = [sub_list for sub_list in metrics_y_array if sub_list]
        # if non_empty_lists:
        #     non_empty_lists_np = np.array(non_empty_lists)
        #     test_metrics_y = np.mean(non_empty_lists_np)
        #     print(f"非空子列表的平均值为: {test_metrics_y}")
        # else:
        #     print("没有非空子列表")
        test_metrics_y = np.mean(metrics_y_array, axis=1)
        # print(f"test_metrics_y : {test_metrics_y}")


        print_test_metric = {}
        # if isinstance(test_metrics_y, (list, tuple, str, dict, set)):
        #     length = len(test_metrics_y)
        #     print(f"对象的长度为: {length}")
        # else:
        #     print("对象不是可迭代对象，无法获取长度")


        for i in range(len(test_metrics_y)):
            # print(f"test_metrics_y 列表长度: {len(test_metrics_y)}")
            # print(f"metric_dict 列表长度: {len(metric_dict)}")
            print_test_metric[metric_dict[i]] = test_metrics_y[i]

        # with open('test_'+self.model_path.split('/')[-2]+'.txt', 'w', encoding='utf-8') as f:
        #     f.write('Disc Dice\n')
        #     f.write(str(metrics_y[0])+'\n')  # Disc Dice
        #     f.write('Cup Dice\n')
        #     f.write(str(metrics_y[2])+'\n')  # Cup Dice
        with open('test_' + self.model_path.split('/')[-2] + '.txt', 'w', encoding='utf-8') as f:
            f.write('ACC\n')
            f.write('AUC\n')
            for i in range(len(metrics_y[0])):
                f.write(f'{metrics_y[0][i]}, {metrics_y[1][i]}\n')
        #     best_test_auc=0.0
        #     if metrics_y[0][0] > best_test_auc:
        #         best_test_acc = metrics_y[0][0]
        #         best_test_auc = metrics_y[0][1]
        # print(f" Best AUC = {best_test_auc},Best ACC = {best_test_acc}")
        print("Test Metrics: ", print_test_metric)
        return print_test_metric  # 确保正确设置了返回值

        # with open('test_' + self.model_path.split('/')[-2] + '.txt', 'w', encoding='utf-8') as f:
        #     f.write('ACC\n AUC\n')
        #     for i in range(len(metrics_y[0])):
        #         f.write(f'{metrics_y[0][i]}, {metrics_y[1][i]}\n')
        # print("Test Metrics: ", print_test_metric)
        # return print_test_metric
def calculate_classification_metrics1(output, target):
    # Assuming 'output' is a list of arrays
    outputs = [pred[0] for pred in output]  # Extracting the first element from each array
    outputs = np.array(outputs)  # Convert to NumPy array if it's not already

    # Apply threshold and convert to binary predictions
    # print(outputs)
    import argparse
    # args = parser.parse_args()
    # taget_dataset = args.Target_Dataset
    p=0.4
    pred = (outputs > p).astype(np.int)  # Apply threshold for binary classification

    # Calculate accuracy
    acc = accuracy_score(target, pred)
    return acc

# def calculate_classification_metrics1(output, target):
#     outputs = [pred[0] for pred in output]
#     pred = (outputs > 0.5).astype(np.int)  # 根据阈值将输出转换为类别预测
#     acc = accuracy_score(target, pred)
#     return acc
from sklearn.metrics import roc_auc_score

def calculate_classification_metrics2(output, target):
    try:
        # 尝试计算 ROC AUC
        output = [pred[0] for pred in output]
        auc = roc_auc_score(target, output)
        return auc
    except ValueError as e:
        # 当出现 ValueError 时，输出提示信息并返回一个标记值（比如 -1）
        print("ValueError occurred:", e)
        return -1
# def calculate_classification_metrics2(output, target):
#
#     outputs = [pred[0] for pred in output]  # Extracting the first element from each array
#     outputs = np.array(outputs)
#
#     # Calculate AUC using extracted 'outputs'
#     auc = roc_auc_score(target, outputs)
#
#     return auc

# def calculate_classification_metrics2(output, target):
#     outputs = [pred[0] for pred in output]  # Extracting the first element from each array
#     outputs = np.array(outputs)
#     auc = roc_auc_score(target, output)
#
#     return auc

def smooth_grad_campp(model, x):
    # Forward pass through the model
    feature, logits = model(x, is_train=False)
    # logits = model(x)
    # Get the class prediction
    pred_class = logits.argmax(dim=1, keepdim=True)
    # Calculate gradients w.r.t. logits
    logits.backward(gradient=torch.ones_like(logits), retain_graph=True)
    # logits.backward(gradient=torch.ones_like(logits))
    logits[0, pred_class].backward()

    # Get the gradients of the last convolutional layer
    conv_outputs = model.get_last_conv()  # Replace this with your model's last conv layer

    # Calculate gradients w.r.t. feature maps
    grads = model.get_last_conv_gradients()

    # Global average pooling (GAP) of gradients
    weights = F.adaptive_avg_pool2d(grads, 1)

    # Create CAM
    cam = torch.mul(conv_outputs, weights).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    # Upsample CAM to input size
    cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
    cam = cam - cam.min()
    cam = cam / cam.max()

    return cam