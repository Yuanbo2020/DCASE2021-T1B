from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dataloader
import os
import imp
import model
import math
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

config = imp.load_source("config","config/Resnet50.py").config
device_ids = config["device_ids"]
data_train_opt = config['data_train_opt']
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print("======================================")
print("Device: {}".format(device_ids))


def fix_bn(m):
   classname = m.__class__.__name__
   if classname.find('BatchNorm') != -1:
       m.eval()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k]
            correct_k = torch.sum(correct_k).float()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ImageClassify(object):
    def __init__(self):
        self.name_list = []
        self.model = model.VideoNet()
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model = self.model.cuda()
        self.save = data_train_opt["final_model_file"]
        self.training_save = data_train_opt["feat_training_file"]
        self.training_log = data_train_opt["training_log"]

        self.best = 0
        self.train_dataset = dataloader.Video_Data(config["data_dir"],"train")
        self.trainloader = DataLoader(self.train_dataset, batch_size=data_train_opt['batch_size']*len(device_ids),num_workers=16,shuffle=True,drop_last=False)


        self.valid_dataset = dataloader.Video_Data(config["data_dir"],"val")
        self.validloader = DataLoader(self.valid_dataset,batch_size=data_train_opt['batch_size']*len(device_ids),num_workers=16,shuffle=True)
        self.LossFun()
        print("Trainloader: {}".format(len(self.trainloader)))
        print("Validloader: {}".format(len(self.validloader)))
    def LossFun(self):
        print("lossing...")
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=data_train_opt['lr'])
        # state = torch.load(
        #     os.path.join(data_train_opt["feat_training_file"], 'checkpoint_0001.pth'))
        # self.model.load_state_dict(state["state_dict"])
        # self.optimizer.load_state_dict(state["optimizer"])

    def TrainingData(self):
        self.model.train()
        log = []
        for epoch in range(data_train_opt['epoch']):
            if (epoch+1) % data_train_opt["decay_epoch"] == 0 :
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*data_train_opt["decay_rate"]

            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            progress = ProgressMeter(
                len(self.trainloader),
                [batch_time, data_time, losses, top1],
                prefix="Epoch: [{}]".format(epoch+1))

            # switch to train mode
            self.model.train()
            end = time.time()
            for i, (img, class_id, fragment_id, location_id) in enumerate(self.trainloader):
                # measure data loading time
                data_time.update(time.time() - end)
                img, class_id = img.cuda(device=device_ids[0]), class_id.cuda(device=device_ids[0])
                # compute output
                predict,feature = self.model(img)

                loss = self.criterion(predict, class_id)

                # acc1/acc5 are (K+1)-way contrast classifier accuracy
                # measure accuracy and record loss
                acc1= accuracy(predict, class_id, topk=(1,))
                losses.update(loss.item(), img.size(0))
                top1.update(acc1[0], img.size(0))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if (i+1) % data_train_opt["log_step"] == 0:
                    loss_avg = losses.avg
                    acc_avg = top1.avg
                    log.append([epoch, i + 1, loss.item(), acc1[0], loss_avg, acc_avg])
                    progress.display(i+1)

            if (epoch+1) % data_train_opt["save_epoch"] == 0:
                acc, a = self.ValidingData(epoch+1)
                np.save(data_train_opt["training_log"], log)
                if a == 1:
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'acc':acc
                    # }, filename=os.path.join(data_train_opt["feat_training_file"],'checkpoint_{:04d}.pth'.format(epoch+1)))
                    }, filename=os.path.join(data_train_opt["feat_training_file"],'best.pth'))

    def save_checkpoint(self,state,filename='checkpoint.pth.tar'):
        torch.save(state, filename)
    def ValidingData(self,epoch):

        self.model.eval()
        a = 0
        with torch.no_grad():
            y_pre = []
            y_true = []
            with tqdm(total=len(self.validloader), desc='Example', leave=True, ncols=100, unit='batch', unit_scale=True) as pbar:
                for i, (img, class_id, fragment_id, location_id) in enumerate(self.validloader):
                    img, class_id = img.cuda(device=device_ids[0]), class_id.cuda(device=device_ids[0])
                    predict, feature = self.model(img)
                    _, pre = torch.max(predict,dim=1)
                    y_pre.append(pre.cpu())
                    y_true.append(class_id.cpu())
                    pbar.update(1)
            y_pre = torch.cat(y_pre).cpu().detach().numpy()
            y_true = torch.cat(y_true).cpu().detach().numpy()
            report = classification_report(y_true, y_pre, target_names=
                        ['airport',
                       'bus',
                       'metro',
                       'metro_station',
                       'park',
                       'public_square',
                       'shopping_mall',
                       'street_pedestrian',
                       'street_traffic',
                       'tram'], digits=4)
            acc = accuracy_score(y_true, y_pre)
            if acc>self.best:
                a = 1
            print(report)

            print("==================")
            with open(data_train_opt["txt"],"a") as f:
                f.write("========= {} =======\n".format(epoch))
                f.write("classification_report".format(epoch))
                f.write(report)
                f.write("\n")
        self.model.train()

        if a ==1:
            with open(data_train_opt["best"], "a") as f:
                f.write("========= {} =======\n".format(epoch))
                f.write("classification_report".format(epoch))
                f.write(report)
                f.write("================\n")

        return acc,a


def main():

    ImgCla = ImageClassify()
    # ImgCla.TrainingData()
    acc, a = ImgCla.ValidingData(epoch=0)
if __name__ == '__main__':
    main()