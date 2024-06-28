# coding:utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys, traceback
import datetime
import random
import numpy as np
import torch
# print("banben:",torch.version.cuda)
import argparse
# F:/tangyilin/TriD-main/TriD-master/OPTIC/torch-cam-main/torch-1.8.0+cu101-cp36-cp36m-win_amd64.whl

from torch.utils.data import DataLoader
from train_DG import TrainDG
from test import Test
from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.transform import collate_fn_wo_transform, collate_fn_w_transform
torch.set_num_threads(1)
import torch
# print(torch.__version__)

# print(torch.cuda.device_count())
# print(torch.cuda.is_available())
# if torch.cuda.is_available():
#     print(f"GPU is available. CUDA version: {torch.version.cuda}")
# else:
#     print("No GPU available.")
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.log = open(filename, 'w')
        self.hook = sys.excepthook
        sys.excepthook = self.kill

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def kill(self, ttype, tvalue, ttraceback):
        for trace in traceback.format_exception(ttype, tvalue, ttraceback):
            print(trace)
        os.remove(self.filename)

    def flush(self):
        pass


def print_information(config):
    print('GPUs: ', torch.cuda.device_count())
    print('time: ', config.time_now)
    print('mode: ', config.mode)
    print('source domain: ', config.Source_Dataset)
    print('target domain: ', config.Target_Dataset)
    print('model: ' + str(config.model_type))

    print('mixstyle_layers: ', config.mixstyle_layers)
    print('random_type: ', config.random_type)
    print('random_prob: ', config.random_prob)

    print('input size: ', config.image_size)
    print('batch size: ', config.batch_size)

    print('optimizer: ', config.optimizer)
    print('lr_scheduler: ', config.lr_scheduler)
    print('lr: ', config.lr)
    print('momentum: ', config.momentum)
    print('weight_decay: ', config.weight_decay)
    print('***' * 10)


def main(config):
    config.time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")

    if config.load_time is not None:
        config.model_path = os.path.join(config.path_save_model, config.load_time)
    else:
        config.model_path = os.path.join(config.path_save_model, config.time_now)

    config.result_path = os.path.join(config.path_save_result, config.time_now, config.mode)
    config.log_path = os.path.join(config.path_save_log, config.mode)
    config.savefig = config.model_type+config.time_now

    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)

    config.log_path = os.path.join(config.log_path, config.time_now+'.log')
    sys.stdout = Logger(config.log_path, sys.stdout)

    source_name = config.Source_Dataset
    source_csv = []
    for s_n in source_name:
        source_csv.append(s_n + '_test.csv')
        source_csv.append(s_n + '_train.csv')
    print(source_csv)
    sr_img_list, sr_label_list = convert_labeled_list(config.dataset_root, source_csv)

    if config.mode == 'train_DG':
        print('Training Phase')
        print_information(config)
        # print('sr_label_list:',sr_label_list)
        # print('sr_img_list:', sr_img_list)
        # print('sr_img_list:',sr_img_list)
        source_dataset = OPTIC_dataset(config.dataset_root, sr_img_list, sr_label_list,
                                       config.image_size, img_normalize=False)
        source_dataloader = DataLoader(dataset=source_dataset,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       collate_fn=collate_fn_w_transform,
                                       num_workers=config.num_workers,
                                       drop_last=True)

        target_test_csv = [config.Target_Dataset + '_test.csv', config.Target_Dataset + '_train.csv']
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)

        target_valid_dataset = OPTIC_dataset(config.dataset_root, ts_img_list, ts_label_list,
                                             config.image_size, img_normalize=True)
        test_dataloader = DataLoader(dataset=target_valid_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     pin_memory=True,
                                     collate_fn=collate_fn_wo_transform,
                                     num_workers=config.num_workers,
                                     drop_last=True)

        train_DG = TrainDG(config, source_dataloader, test_dataloader)
        train_DG.run()

    elif config.mode == 'single_test':
        print('Single_test for multi-source domain generalization...')
        print('Test Target: ' + config.Target_Dataset)
        print('Loading model: ' + str(config.load_time) + '/' + 'best' + '-' + str(config.model_type) + '.pth')

        # target_test_csv = [config.Target_Dataset + '_test.csv']
        target_test_csv = [config.Target_Dataset + '_test.csv', config.Target_Dataset + '_train.csv']
        print(target_test_csv)
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)

        target_valid_dataset = OPTIC_dataset(config.dataset_root, ts_img_list, ts_label_list,
                                             config.image_size, img_normalize=True)
        test_dataloader = DataLoader(dataset=target_valid_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     pin_memory=True,
                                     collate_fn=collate_fn_wo_transform,
                                     num_workers=config.num_workers,
                                     drop_last=True)

        test = Test(config, test_dataloader)
        test.test()

    elif config.mode == 'multi_test':
        print('Multi_test for single-source domain generalization...')
        print('Train Source: ' + config.Target_Dataset)
        print('Loading model: ' + str(config.load_time) + '/' + 'best' + '-' + str(config.model_type) + '.pth')
        # Disc_Dice, Disc_ASD, Cup_Dice, Cup_ASD = [], [], [], []
        ACC, AUC, SPE, SEN = [],[],[],[]
        test_datasets = ['refuge2tr','refuge2te','refuge2va']
        test_datasets.remove(config.Target_Dataset)

        for target in test_datasets:
            target_test_csv = [target + '_test.csv', target + '_train.csv']
            print(target_test_csv)
            ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)

            target_valid_dataset = OPTIC_dataset(config.dataset_root, ts_img_list, ts_label_list,
                                                 config.image_size, img_normalize=True)
            test_dataloader = DataLoader(dataset=target_valid_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True,
                                         collate_fn=collate_fn_wo_transform,
                                         num_workers=config.num_workers,
                                         drop_last=True)

            test = Test(config, test_dataloader)
            result_dict = test.test()
            ACC.append(result_dict['ACC']), AUC.append(result_dict['AUC'])
            # SPE.append(result_dict['SPE']), SEN.append(result_dict['SEN'])
        print('Mean ACC:{:.8f} Mean AUC:{:.8f} '.format(
            np.mean(ACC), np.mean(AUC)))
        print('Mean ACC:{:.8f}  '.format(
            np.mean(ACC)))
        print('***'*10)
        print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train_DG',
                        help='train_DG/single_test/multi_test')   # choose the mode

    parser.add_argument('--load_time', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='Res_Unet', help='Res_Unet')  # choose the model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')

    parser.add_argument('--mixstyle_layers', nargs='+', type=str, default=['layer0'], help='layer0-4')
    parser.add_argument('--random_type', type=str, default='TriD', help='TriD/MixStyle/EFDMix')
    parser.add_argument('--random_prob', type=float, default=0.5)

    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=2)

    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)#读取的线程个数，之前是8

    parser.add_argument('--optimizer', type=str, default='SGD', help='SGD/Adam/AdamW')
    parser.add_argument('--lr_scheduler', type=str, default='Epoch',
                        help='Cosine/Step/Epoch')   # choose the decrease strategy of lr
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0005)  # weight_decay in SGD
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)  # beta1 in Adam/AdamW
    parser.add_argument('--beta2', type=float, default=0.99)  # beta2 in Adam/AdamW
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10)

    parser.add_argument('--Source_Dataset', nargs='+', type=str, default=['Refuge2','Harvard','ORIGA'],
                        help='Refuge2/ RIMONE/Harvard/ORIGA')
    parser.add_argument('--Target_Dataset', type=str, default='RIMONE',
                        help='Refuge2/ RIMONE/Harvard/ORIGA')

    parser.add_argument('--path_save_result', type=str, default='./results/')
    parser.add_argument('--path_save_model', type=str, default='./models/')
    parser.add_argument('--path_save_log', type=str, default='./logs/')
    parser.add_argument('--dataset_root', type=str, default='F:/tangyilin/TriD-main/dataset')

    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda:0')
    else:
        parser.add_argument('--device', type=str, default='cpu')

    config = parser.parse_args()

    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method("spawn")
    main(config)

