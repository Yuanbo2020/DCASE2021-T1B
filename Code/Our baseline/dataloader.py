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
class Video_Data(Dataset):
    def __init__(self,data_root="/home/tyz/TAU_dataset/dataset",type="train"):
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
        self.type = type
        fragments = []
        self.transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),

            ])
        with open(os.path.join(data_root,type)+".csv", 'r') as f:
            reader = csv.reader(f)
            x = 0
            for i in reader:
                if x ==0:
                    x = 1
                    continue
                file_name = os.path.split(i[1])[-1]
                fragment_id = os.path.splitext(file_name)[0]
                class_id = i[2]
                location_id = i[3]
                fragments.append([fragment_id,class_id,location_id])

        self.data = []
        for i in fragments:
            fragment_id, class_id, location_id = i
            path = os.path.join(data_root,"video",class_id,fragment_id)
            photo_names = os.listdir(path)
            for img in photo_names:
                img_add = os.path.join(path,img)
                self.data.append([img_add,classes_dic[class_id],fragment_id,location_id])


        self.length = len(self.data)




        
    def __len__(self):

        return  self.length

    def __getitem__(self, idx):
        img_add, class_id, fragment_id, location_id = self.data[idx]

        img = Image.open(img_add)
        img = img.resize((224, 224))
        img = img.convert("RGB")
        img = self.transforms(img)
        return img, class_id, fragment_id, location_id




if __name__ == '__main__':

    train_dataset = Video_Data()
    model = models.resnet18()
    trainloader_s = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for i, (x, class_id, fragment_id, location_id) in enumerate(trainloader_s):

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