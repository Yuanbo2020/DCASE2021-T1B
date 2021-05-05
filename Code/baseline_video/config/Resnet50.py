import os



config = {}
data_train_opt = {}
config["data_dir"] = "/home/tyz/TAU_dataset/dataset"
config["device_ids"] = [1]

data_train_opt['batch_size'] = 64
data_train_opt['epoch'] = 50
data_train_opt['split'] = 'train'
data_train_opt['lr'] = 0.0001
data_train_opt["decay_epoch"] = 20
data_train_opt["decay_rate"] = 0.5
data_train_opt["save_epoch"] = 1
data_train_opt["log_step"] = 50
data_train_opt["continue_model"] = ""

feat_training_file = '/home/share/tyz/experiments/Our_baseline_video/lr{}_batch{}'.format(data_train_opt['lr'],data_train_opt['batch_size'])
final_model_file = os.path.join(feat_training_file,"Final_model.pth")

data_train_opt["training_log"] = os.path.join(feat_training_file,"training_log.npy")
data_train_opt["txt"] = os.path.join(feat_training_file,"acc.txt")
if not os.path.exists(feat_training_file):
    os.makedirs(feat_training_file)
data_train_opt["best"] = os.path.join(feat_training_file,"acc_best.txt")


data_train_opt["feat_training_file"] = feat_training_file
data_train_opt["final_model_file"] = final_model_file
config["data_train_opt"] = data_train_opt


