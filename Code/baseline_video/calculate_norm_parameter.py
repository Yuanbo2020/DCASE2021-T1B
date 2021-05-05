from torch.utils.data import DataLoader
import torch
import numpy as np
import dataloader
def get_mean_std():

    train_dataset = dataloader.Video_Data("/home/tyz/TAU_dataset/dataset", "train")
    trainloader = DataLoader(train_dataset, batch_size=32,
                                  num_workers=16, shuffle=True, drop_last=False)
    all_data = []
    for i, (img, class_id, fragment_id, location_id) in enumerate(trainloader):
        all_data.append(img)
        if (i+1)% 10 == 0:
            print(i+1)
        if i+1 == 500:
            break
    all_data = torch.cat(all_data,dim=0)

    mean = np.mean(all_data.numpy(), axis=(0,2,3))
    std = np.std(all_data.numpy(), axis=(0,2,3))
    print(mean, std)
    return mean, std

get_mean_std()