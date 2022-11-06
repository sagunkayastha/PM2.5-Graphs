from icecream import ic
import torch
import os
import sys
from graph import Graph
from dataset import HazeData

from PM25_GNN import PM25_GNN

import arrow
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pickle
import glob
import shutil
torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


g = Graph()
city_num = g.node_num
batch_size = 8
epochs = 50
hist_len = 1
pred_len = 24
weight_decay = 0.00005
early_stop = 10
lr = 0.00005
results_dir = '/project/ychoi/rdimri/results'
dataset_num = 1
# exp_model = config['experiments']['model']
exp_repeat = 1
save_npy = True

def IOA_Loss(o,p): 
    ioa = 1 -(torch.sum((o-p)**2))/(torch.sum((torch.abs(p-torch.mean(o))+torch.abs(o-torch.mean(o)))**2))
    return (-ioa)

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

criterion1 = RMSELoss
criterion2 = IOA_Loss
criterion = nn.MSELoss()

train_data = HazeData(g, hist_len, pred_len, dataset_num, flag='Train')
val_data = HazeData(g, hist_len, pred_len, dataset_num, flag='Val')
test_data = HazeData(g, hist_len, pred_len, dataset_num, flag='Test')
in_dim = train_data.feature.shape[-1] + 1
wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
wind_max, wind_min = train_data.wind_max, train_data.wind_min
pm25_mean, pm25_std = test_data.pm25_mean, test_data.pm25_std
pm25_max, pm25_min = test_data.pm25_max, test_data.pm25_min

def get_metric(predict_epoch, label_epoch):
    haze_threshold = 75
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi = hit / (hit + falsealarm + miss)
    pod = hit / (hit + miss)
    far = falsealarm / (hit + falsealarm)
    predict = predict_epoch[:,:,:,0].transpose((0,2,1))
    label = label_epoch[:,:,:,0].transpose((0,2,1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae = np.mean(np.mean(np.abs(predict - label), axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    return rmse, mae, csi, pod, far

#def IOA_Loss(o,p): 
 #   ioa = 1 -(torch.sum((o-p)**2))/(torch.sum((torch.abs(p-torch.mean(o))+torch.abs(o-torch.mean(o)))**2))
  #  return (ioa)

#def index_agreement(s,o):
 #   ia = 1 -(np.sum((o-s)**2))/(np.sum((np.abs(s-np.mean(o))+np.abs(o-np.mean(o)))**2))
  #  return ia

def get_exp_info():
    exp_info =  '============== Train Info ==============\n' + \
                'City number: %s\n' % city_num + \
                'batch_size: %s\n' % batch_size + \
                'epochs: %s\n' % epochs + \
                'hist_len: %s\n' % hist_len + \
                'pred_len: %s\n' % pred_len + \
                'weight_decay: %s\n' % weight_decay + \
                'early_stop: %s\n' % early_stop + \
                'lr: %s\n' % lr + \
                '========================================\n'
    return exp_info


def get_model():
    return PM25_GNN(hist_len, pred_len, in_dim, city_num, batch_size, device, g.edge_index, g.edge_attr, wind_mean, wind_std, wind_max, wind_min)



def train(train_loader, model, optimizer):
    model.train()
    train_loss = 0
    i = 0
    for batch_idx, data in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        pm25, feature= data
        pm25 = pm25[:,:,:,np.newaxis].to(device)
        # ic(pm25.sharpe)
        # ic(feature.shape)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        #ic(pm25_label.shape)
        pm25_hist = pm25[:, :hist_len]
        #ic(pm25_hist.shape)
        pm25_pred = model(pm25_hist, feature)
        loss1 = criterion1(pm25_pred, pm25_label)
        loss2 = criterion2(pm25_pred, pm25_label)
        loss = loss1 + loss2
        if ((i % 1) == 0):
            ic(loss)
            ic(-IOA_Loss(pm25_label, pm25_pred))
        #ic(index_agreement(pm25_pred, pm25_label))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        i = i + 1
    train_loss /= batch_idx + 1
    return train_loss


def val(val_loader, model):
    model.eval()
    val_loss = 0
    i = 0
    for batch_idx, data in tqdm(enumerate(val_loader)):
        pm25, feature = data
        pm25 = pm25.to(device)
        pm25 = pm25[:,:,:,np.newaxis].to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        lossv1 = criterion1(pm25_pred, pm25_label)
        lossv2 = criterion2(pm25_pred, pm25_label)
        lossv = lossv1 + lossv2
        if ((i % 1) == 0):
            ic(lossv)
            ic(-IOA_Loss(pm25_label, pm25_pred))
        val_loss += lossv.item()
        i = i + 1

    val_loss /= batch_idx + 1
    return val_loss


def test(test_loader, model):
    model.eval()
    predict_list = []
    label_list = []
    time_list = []
    test_loss = 0
    i = 0
    for batch_idx, data in enumerate(test_loader):
        pm25, feature = data
        pm25 = pm25.to(device)
        pm25 = pm25[:,:,:,np.newaxis].to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss1 = criterion1(pm25_pred, pm25_label)
        loss2 = criterion2(pm25_pred, pm25_label)
        loss = loss1 + loss2
        if ((i % 1) == 0):
            ic(loss)
            ic(-IOA_Loss(pm25_label, pm25_pred))
        test_loss += loss.item()
        i = i + 1

        pm25_pred_val = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1) * (pm25_std) + pm25_mean
        pm25_label_val = pm25.cpu().detach().numpy() * (pm25_std) + pm25_mean
        predict_list.append(pm25_pred_val)
        label_list.append(pm25_label_val)
        #time_list.append(time_arr.cpu().detach().numpy())

    test_loss /= batch_idx + 1

    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    #time_epoch = np.concatenate(time_list, axis=0)
    predict_epoch[predict_epoch < 0] = 0

    return test_loss, predict_epoch, label_epoch


def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()


def main():                
    exp_info = get_exp_info()
    print(exp_info)

    exp_time = arrow.now().format('YYYYMMDDHHmmss')

    train_loss_list, val_loss_list, test_loss_list, rmse_list, mae_list, csi_list, pod_list, far_list = [], [], [], [], [], [], [], []

    for exp_idx in range(exp_repeat):
        print('\nNo.%2d experiment ~~~' % exp_idx)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle = False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle = False, drop_last=True)

        model = get_model()
        model = model.to(device)
        model_name = type(model).__name__

        print(str(model))
	
        try:
            model.load_dict()
            print('this')
	#model.load_dict()
        except:
            print('wrong')
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

        exp_model_dir = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), model_name, str(exp_time), '%02d' % exp_idx)
        if not os.path.exists(exp_model_dir):
            os.makedirs(exp_model_dir)
        model_fp = os.path.join(exp_model_dir, 'model.pth')

        val_loss_min = 100000
        best_epoch = 0

        train_loss_, val_loss_ = 0, 0

        for epoch in range(epochs):
            print('\nTrain epoch %s:' % (epoch))

            train_loss = train(train_loader, model, optimizer)
            val_loss = val(val_loader, model)

            print('train_loss: %.4f' % train_loss)
            print('val_loss: %.4f' % val_loss)

            if epoch - best_epoch > early_stop:
                break

            if val_loss < val_loss_min:
                val_loss_min = val_loss
                best_epoch = epoch
                print('Minimum val loss!!!')
                torch.save(model.state_dict(), model_fp)
                print('Save model: %s' % model_fp)

                test_loss, predict_epoch, label_epoch = test(test_loader, model)
                train_loss_, val_loss_ = train_loss, val_loss
                rmse, mae, csi, pod, far = get_metric(predict_epoch, label_epoch)
                print('Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (train_loss_, val_loss_, test_loss, rmse, mae, csi, pod, far))

                if save_npy:
                    np.save(os.path.join(exp_model_dir, 'predict.npy'), predict_epoch)
                    np.save(os.path.join(exp_model_dir, 'label.npy'), label_epoch)

        train_loss_list.append(train_loss_)
        val_loss_list.append(val_loss_)
        test_loss_list.append(test_loss)
        rmse_list.append(rmse)
        mae_list.append(mae)
        csi_list.append(csi)
        pod_list.append(pod)
        far_list.append(far)

        print('\nNo.%2d experiment results:' % exp_idx)
        print(
            'Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (
            train_loss_, val_loss_, test_loss, rmse, mae, csi, pod, far))

    exp_metric_str = '---------------------------------------\n' + \
                     'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(train_loss_list)) + \
                     'val_loss   | mean: %0.4f std: %0.4f\n' % (get_mean_std(val_loss_list)) + \
                     'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(test_loss_list)) + \
                     'RMSE       | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list)) + \
                     'MAE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list)) + \
                     'CSI        | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_list)) + \
                     'POD        | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_list)) + \
                     'FAR        | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_list))

    metric_fp = os.path.join(os.path.dirname(exp_model_dir), 'metric.txt')
    with open(metric_fp, 'w') as f:
        f.write(exp_info)
        f.write(str(model))
        f.write(exp_metric_str)

    print('=========================\n')
    print(exp_info)
    print(exp_metric_str)
    print(str(model))
    print(metric_fp)


if __name__ == '__main__':
    main()
