from torch.utils.data import Dataset
import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
from matplotlib import pyplot as plt
import torchvision.models as models
import torchvision.transforms.functional as TF
import json
import csv
import pickle
import h5py

class Load_Data(Dataset):
    def __init__(self,data_root="/home/share/tyz/dataset",type="train"):
        classes_dic = {'airport': 0,
                       'bus': 1,
                       'metro': 2,
                       'metro_station': 3,
                       'park': 4,
                       'public_square': 5,
                       'shopping_mall': 6,
                       'street_pedestrian': 7,
                       'street_traffic': 8,
                       'tram': 9}
        id2class = ['airport',
                       'bus',
                       'metro',
                       'metro_station',
                       'park',
                       'public_square',
                       'shopping_mall',
                       'street_pedestrian',
                       'street_traffic',
                       'tram']
        three_dic = {'airport': 0,
                       'bus': 1,
                       'metro': 1,
                       'metro_station': 0,
                       'park': 2,
                       'public_square': 2,
                       'shopping_mall': 0,
                       'street_pedestrian': 2,
                       'street_traffic': 2,
                       'tram': 1}
        self.type = type
        fragments = []
        self.transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.432,0.413,0.417],
                          std=[0.231, 0.230, 0.231])

            ])


        if  self.type == 'train':
            self.path_input = os.path.join(data_root,'features_data/audio_features_data/tr.hdf5')

        if self.type == 'val':
            self.path_input =  os.path.join(data_root,'features_data/audio_features_data/val.hdf5')

        global_mean_std_path = os.path.join(data_root,'features_data/audio_features_data/global_mean_std.npz')
        mean_std = np.load(global_mean_std_path)
        self.mean = mean_std['global_mean']
        self.std = mean_std['global_std']

        self.group = []
        def func(name, obj):
            if isinstance(obj, h5py.Group):
                sp =name.split("/")
                if len(name.split("/"))== 3:
                    class_id = sp[0]
                    fragment_id = sp[2]
                    audio_name = name
                    self.group.append([fragment_id,class_id,audio_name])
        self.hf = h5py.File(self.path_input, 'r')
        self.hf.visititems(func)
        self.hf.close()




        # with open(os.path.join(data_root,type)+".csv", 'r') as f:
        #     reader = csv.reader(f)
        #     x = 0
        #     for i in reader:
        #         if x ==0:
        #             x = 1
        #             continue
        #         file_name = os.path.split(i[1])[-1]
        #         fragment_id = os.path.splitext(file_name)[0]
        #         class_id = i[2]
        #         location_id = i[3]
        #         fragments.append([fragment_id,class_id,location_id])

        self.data = []
        for i in self.group:
            fragment_id, class_id, audio_name = i
            class_id = int(class_id)
            path = os.path.join(data_root,"video_5fps",id2class[class_id],fragment_id)
            img = []
            for i in range(2,51,5):
                img.append(os.path.join(path,fragment_id+"_{}.jpg".format(i)))
            self.data.append([img,audio_name,class_id])


        self.length = len(self.data)


        
    def __len__(self):

        return  self.length

    def __getitem__(self, idx):
        img_adds,audio_name, class_id= self.data[idx]
        imgs = []
        for img_add in img_adds:
            img = Image.open(img_add)
            img = img.resize((224, 224))
            print(img.size)
            # (224, 224)

            img = img.convert("RGB")
            print(img.size)
            # (224, 224)
            print(np.max(img), np.min(img))

            img = self.transforms(img)
            print(img.size())
            # torch.Size([3, 224, 224])
            print(torch.max(img), torch.min(img))

            imgs.append(img.unsqueeze(0))
        imgs = torch.cat(imgs,dim=0)
        print(imgs.size())
        print(torch.max(imgs), torch.min(imgs))

        # 255 9
        # torch.Size([3, 224, 224])
        # tensor(2.5522) tensor(-1.6664)
        # (224, 224)
        # (224, 224)
        # 255 0
        # torch.Size([3, 224, 224])
        # tensor(2.5522) tensor(-1.8701)
        # torch.Size([10, 3, 224, 224])
        # tensor(2.5522) tensor(-1.8701)


        hhhhhh


        audios = []
        for i in range(96):
            hf = h5py.File(self.path_input, 'r')
            emb_audio = torch.from_numpy(np.array(hf[audio_name+"/{}".format(i)])).unsqueeze(0).float()
            audios.append(emb_audio)
        audios = torch.cat(audios,dim=0)

        return imgs,audios, class_id




if __name__ == '__main__':

    train_dataset = Load_Data()
    model = models.resnet18()
    trainloader_s = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for i, (x,y, class_id) in enumerate(trainloader_s):
        print(x.size())
        print(y.size())
        break
    inv_transform = transforms.Compose([
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8),
        ])
    for i in range(x.size(0)):
        plt.subplot(2,4,i+1)
        fig=plt.imshow(inv_transform(x[i]))
        plt.subplot(2,4,4+i+1)
        fig=plt.imshow(inv_transform(x[i]))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.show()