# Copyright (c) 2019-present, Royal Bank of Canada.
# Copyright (c) 2021 THUML @ Tsinghua University
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the Autoformer (https://arxiv.org/pdf/2106.13008.pdf) implementation
# from https://github.com/thuml/Autoformer by THUML @ Tsinghua University
####################################################################################

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, Reformer, \
        FEDformer, Performer,  NHits, FiLM, EVT, Seq2Seq
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from robust_loss_pytorch import AdaptiveLossFunction
from dilate_loss import DilateLoss

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os
import time
import warnings

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'FEDformer': FEDformer,
            'Performer': Performer,
            'NHits': NHits,
            'FiLM': FiLM,
            'EVT': EVT,
            'Seq2Seq': Seq2Seq,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        print(f"NUMBER OF PARAMETERS: {self.args.model}: {sum(p.numel() for p in model.parameters())}")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, additional_params=None):
        if additional_params is not None:
            model_optim = optim.AdamW(list(self.model.parameters())+additional_params, lr=self.args.learning_rate)
        else:
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, save=False):
        total_loss = []
        X = []
        yt = []
        yp = []
        means = []
        scales = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mean, scale) in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # model call
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if self.args.output_attention:
                    outputs = outputs[0]
                
                if self.args.loss=='evl':
                    outputs, us = outputs
                    vs = self.model.extreme_values(batch_y)
                    loss = criterion(outputs, batch_y, us, vs)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if save:
                    X.append(batch_x.detach().cpu())
                    yt.append(true)
                    yp.append(pred)
                    means.append(mean)
                    scales.append(scale)

                assert pred.shape == true.shape
                if self.args.loss != 'evl':
                    loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        if save:
            X = np.concatenate(X, axis=0)
            yt = np.concatenate(yt, axis=0)
            yp = np.concatenate(yp, axis=0)
            means = np.concatenate(means, axis=0)
            scales = np.concatenate(scales, axis=0)
            np.savez(
                os.path.join(self.args.checkpoints, 'preds.npz'),
                X=X, yt=yt, yp=yp, means=means, scales=scales
            )
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()


        criterion = self._select_criterion()
        if self.args.loss == 'mse':
            criterion_tmp = torch.nn.MSELoss(reduction='none')
        elif self.args.loss == 'huber':
            criterion_tmp = torch.nn.HuberLoss(reduction='none', delta=0.5)
        elif self.args.loss == 'l1':
            criterion_tmp = torch.nn.L1Loss(reduction='none')
        elif self.args.loss == 'adaptive':
            adaptive = AdaptiveLossFunction(1, torch.float32, self.device, alpha_hi=3.0)
            criterion_tmp = adaptive.lossfun 
            adaptive_optim = optim.AdamW(list(adaptive.parameters()), lr=0.001)
        elif self.args.loss == 'dilate':
            criterion_tmp = DilateLoss(alpha=0.5, gamma=1e-2, device=self.device)

        for epoch in range(self.args.train_epochs):
            if self.args.loss == 'adaptive':
                adaptive.print()
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            pbar = tqdm(enumerate(train_loader), position=0, leave=True, total=len(train_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mean, scale) in pbar:
                iter_count += 1
                model_optim.zero_grad()
                if self.args.loss == 'adaptive':
                    adaptive_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # model call
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if self.args.output_attention:
                    outputs = outputs[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                assert outputs.shape == batch_y.shape, f"{outputs.shape}, {batch_y.shape}"

                if self.args.loss == 'adaptive':
                    loss =  criterion_tmp((outputs-batch_y).flatten().unsqueeze(-1))
                else:
                    loss =  criterion_tmp(outputs, batch_y)
                loss = loss.mean()
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                if self.args.loss == 'adaptive':
                    adaptive_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print("Vali 1")
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("Vali 2")
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        X = []
        preds = []
        trues = []
        means = []
        scales = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        running_times = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mean, scale) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                start_time = time.time()

                # model call
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if self.args.output_attention:
                    outputs = outputs[0]
                
                if self.args.loss == 'evl':
                    outputs, us = outputs
                    vs = self.model.extreme_values(batch_y)
                    loss = self.model.l1(outputs, batch_y, us, vs)
                
                running_times.append(time.time()-start_time)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()

                pred = outputs.numpy()
                true = batch_y.numpy()

                preds.append(pred)
                trues.append(true)
                if self.args.save_preds:
                    X.append(batch_x.detach().cpu().numpy())
                    means.append(mean.detach().cpu().numpy())
                    scales.append(scale.detach().cpu().numpy())

                save_every = len(test_loader)//20 if len(test_loader)>20 else 1
                if i % save_every == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.svg'))

        # result save
        result_file_name = 'result.txt'
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        if self.args.save_preds:
            X = np.concatenate(X, axis=0)
            means = np.concatenate(means, axis=0)
            scales = np.concatenate(scales, axis=0)
            if self.args.features != 'SA':
                means = means[0]
                scales = scales[0]
            np.savez(
                folder_path + 'preds.npz',
                X=X, preds=preds, trues=trues, means=means, scales=scales
            )
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print(f'running time: {np.array(running_times).sum()}')
        with open(result_file_name, 'a') as f:
            f.write(f'{setting}  \n mse:{mse}, mae:{mae} \n \n')
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='test')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        inputs = []
        trues = []
        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mean, scale) in tqdm(enumerate(pred_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # model call
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if self.args.output_attention:
                    outputs = outputs[0]
                if self.args.loss=='evl':
                    outputs, us = outputs
                    vs = self.model.extreme_values(batch_y)
                    loss = self.model.l1(outputs, batch_y, us, vs)
                
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                inputs.append(batch_x.detach().cpu().numpy())
                trues.append(batch_y[:, -self.args.pred_len:].detach().cpu().numpy())
                preds.append(pred)

        inputs = np.concatenate(inputs, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.savez(folder_path + 'real_prediction.npz', inputs=inputs, preds=preds, trues=trues)
        return
