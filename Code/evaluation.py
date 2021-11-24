from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dataloader
import os, sys
import imp
import model
import model_1_v2
import model_3_v2
import model_4_v2
import math
import time
from tqdm import tqdm
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import log_loss
from torchvision import transforms
import pandas
import random


def create_folder(feature_dir):
    """ 如果目录有多级，则创建最后一级。如果最后一级目录的上级目录有不存在的，则会抛出一个OSError。   """
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)


config = imp.load_source("config", "config/Resnet50.py").config
device_ids = config["device_ids"]
data_train_opt = config['data_train_opt']
print("======================================")
print("Device: {}".format(device_ids))


class ImageClassify(object):
    def __init__(self):
        self.model = model.Mixed_model(data_train_opt["dim"])
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model = self.model.cuda(device=device_ids[0])

        self.transforms = transforms.Compose([
                transforms.Normalize(mean=[0.432, 0.413, 0.417],
                                     std=[0.231, 0.230, 0.231])])

    def obtain_modellist(self, model_path, ranking_metric, top_N_scope):
        for top_id in range(top_N_scope):
            top_N = top_id + 1

            model_list = []
            metric_list = []

            for modelfile in os.listdir(model_path):
                if modelfile.endswith('.pth'):  # Epoch_21_acc_0.9025710419485792_loss_0.001110262327169749.pth
                    model_list.append(os.path.join(model_path, modelfile))
                    # print('model_path: ', model_path, 'top N: ', top_N, 'modelfile: ', modelfile)

                    part = modelfile.split('.pth')[0].split('_')
                    if ranking_metric == 'val_acc':
                        metric_list.append(float(part[3]))
                    elif ranking_metric == 'training_loss':
                        metric_list.append(float(part[5]))
                    elif ranking_metric == 'val_logloss':
                        metric_list.append(float(part[7]))

            ascending = sorted(range(len(metric_list)), key=lambda k: metric_list[k])
            if ranking_metric == 'val_acc':
                indices = ascending[::-1][:top_N]
            elif ranking_metric == 'training_loss':
                indices = ascending[:top_N]
            elif ranking_metric == 'val_logloss':
                indices = ascending[:top_N]

        return [model_list[i] for i in indices]

    def fusion_multimodel_topN(self):
        ############## 修改这里，选择用哪个指标排序 #################################
        # ls -l|grep "^-"| wc -l

        ranking_metric1 = 'val_acc'
        # ranking_metric1 = 'val_logloss'
        top_N_scope1 = 1
        model_path1 = '/home/share/tyz/experiments/final_model_1_v2/dim_256_lr0.0001_batch16_with_logloss'
        modellist1 = self.obtain_modellist(model_path1, ranking_metric1, top_N_scope1)
        print('model1: ', len(modellist1), modellist1)

        # ranking_metric2 = 'training_loss'
        # top_N_scope2 = 11
        # model_path2 = '/home/share/tyz/experiments/final_model_2_v2/dim_256_lr0.0001_batch16_with_logloss'
        # modellist2 = self.obtain_modellist(model_path2, ranking_metric2, top_N_scope2)
        # # print('model2: ', len(modellist2), modellist2)

        # final_modellist = list(set(modellist1 + modellist2))
        # print('model1 final: ', len(final_modellist), final_modellist)

        if 'model_1_v2' in model_path1:
            self.model = model_1_v2.Mixed_model1(data_train_opt["dim"])
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
            self.model = self.model.cuda(device=device_ids[0])
        elif 'model_3_v2' in model_path1:
            self.model = model_3_v2.Mixed_model3(data_train_opt["dim"])
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
            self.model = self.model.cuda(device=device_ids[0])
        elif 'model_4_v2' in model_path1:
            self.model = model_4_v2.Mixed_model4(data_train_opt["dim"])
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
            self.model = self.model.cuda(device=device_ids[0])


        final_modellist = modellist1

        label_list = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                      'street_pedestrian', 'street_traffic', 'tram']

        submission = os.path.join(os.getcwd(), 'submission')
        create_folder(submission)

        ############## 以下的不要动 #################################
        evaluation_audio_dir = '/home/tyz/evaluation/audio_emb'
        evaluation_image_dir = '/home/share/tyz/dataset/eval/image'

        all_num = 72000
        ids_list = [i for i in range(all_num)]
        # random.shuffle(ids_list)

        # step = int(all_num / 6)
        step = 0
        if step:
            list_chunk = [ids_list[i:i + step] for i in range(0, all_num, step)]
        else:
            list_chunk = [ids_list]

        target_chunk = 0

        columns = [i for i in label_list]
        columns.insert(0, 'scene_label')
        columns.insert(0, 'filename')
        df = pandas.DataFrame(columns=columns)

        for fileid in list_chunk[target_chunk]:
            audiofile_path = os.path.join(evaluation_audio_dir, str(fileid) + '.npy')
            audiodata = np.load(audiofile_path)

            imagefile_path = os.path.join(evaluation_image_dir, str(fileid) + '.npy')
            imagedata = np.load(imagefile_path)
            # print(id, audiodata.shape, imagedata.shape)
            # # (6, 512) (10, 224, 224, 3)

            audiodata = np.tile(audiodata, (16, 1))
            imagedata = imagedata/255
            imagedata = np.transpose(imagedata, (0, 3, 1, 2))
            # print(audiodata.shape, imagedata.shape)
            audiodata = np.expand_dims(audiodata, axis=0)
            imagedata = np.expand_dims(imagedata, axis=0)

            audiodata = torch.Tensor(audiodata).float().to(device_ids[0])
            imagedata = torch.Tensor(imagedata).float().to(device_ids[0])

            # print(id, audiodata.size(), imagedata.size())
            # # 0 torch.Size([6, 512]) torch.Size([10, 3, 224, 224])
            imagedata = self.transforms(imagedata)

            # print(id, audiodata.size(), imagedata.size())

            top_y_possible = []
            for each_model in final_modellist:
                # print('current model: ', each_model)
                state = torch.load(each_model)
                self.model.load_state_dict(state)
                self.model.eval()
                with torch.no_grad():
                    predict = self.model(imagedata, audiodata)
                    possible = torch.nn.functional.softmax(predict, dim=1)
                    possible = possible.cpu().detach().numpy()
                    # print(fileid, possible)
                    # print(possible.shape) # (1, 10)
                    top_y_possible.append(possible)
            # print(np.array(top_y_possible).shape) # (2, 1, 10)

            final_average_y_possible = np.array(top_y_possible).mean(axis=0)
            # (1, 10)
            # print(final_average_y_possible.shape)

            prob = final_average_y_possible.tolist()[0]
            m = max(prob)  # 挑选出最高概率值的场景
            es_class = prob.index(m)

            print(fileid, label_list[es_class])

            data = prob  # 将一组10个概率值放入列表
            data.insert(0, label_list[es_class])  # 插入在scene_label处
            data.insert(0, str(fileid) + '.wav')  # 插入filename
            data = dict(zip(columns, data))
            df = df.append([data], ignore_index=True)

        filename = 'task1b_1.output.csv'
        df.to_csv(os.path.join(submission, filename), sep='\t', index=False)





def main():
    ImgCla = ImageClassify()

    ImgCla.fusion_multimodel_topN()


if __name__ == '__main__':
    main()