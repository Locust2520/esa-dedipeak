from argparse import Namespace
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.exp_main import Exp_Main
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from robust_loss_pytorch import AdaptiveLossFunction
from models import InverseHeatDissipation

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from random import random
import os
import time
import warnings

warnings.filterwarnings('ignore')


class Exp_Diffusion(Exp_Basic):
    def __init__(self, args):
        super(Exp_Diffusion, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'InverseHeatDissipation': InverseHeatDissipation,
        }
        model = model_dict[self.args.model].Model(self.args, self.device).float()

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

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mean, scale) in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:].float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:].float().to(self.device)

                t = self.model.get_t().to(self.device)
                dec_inp, eps = self.model.q_xt_x0(batch_y, batch_y_mark, batch_x, batch_x_mark, t)

                eps_hat = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, t)

                pred = eps_hat.detach().cpu()
                true = eps.detach().cpu()

                assert pred.shape == true.shape, f"pred shape: {pred.shape}, true shape: {true.shape}"
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
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

        for epoch in range(self.args.train_epochs):
            if self.args.loss=='adaptive':
                adaptive.print()
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mean, scale) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                if self.args.loss=='adaptive':
                    adaptive_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:].float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:].float().to(self.device)

                t = self.model.get_t().to(self.device)
                dec_inp, eps = self.model.q_xt_x0(batch_y, batch_y_mark, batch_x, batch_x_mark, t)

                if random() < self.args.condition_prob:
                    eps_hat = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, t)
                else:
                    eps_hat = self.model(None, None, dec_inp, batch_y_mark, t)

                assert eps_hat.shape == eps.shape, f"{eps_hat.shape=}, {eps.shape=}"

                if self.args.loss=='adaptive':
                    loss =  criterion_tmp((eps_hat - eps).flatten().unsqueeze(-1))
                else:
                    loss =  criterion_tmp(eps_hat, eps)
                loss = loss.mean()
                if self.args.loss == 'mse':
                    loss = loss * 100
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    
                loss.backward()
                model_optim.step()
                if self.args.loss=='adaptive':
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

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        running_times = []
        preds = []
        trues = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mean, scale) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:].float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:].float().to(self.device)

                start_time = time.time()

                t = self.model.get_t().to(self.device)
                dec_inp, eps = self.model.q_xt_x0(batch_y, batch_y_mark, batch_x, batch_x_mark, t)

                eps_hat = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, t)

                running_times.append(time.time()-start_time)
                
                eps = eps.detach().cpu()
                eps_hat = eps_hat.detach().cpu()
                batch_x = batch_x.detach().cpu()
                batch_y = batch_y.detach().cpu()

                pred = eps_hat.numpy()
                true = eps.numpy()

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
        
        transformer_based = bool(self.args.transformer_model)
        if transformer_based:
            targs = Namespace(**self.args.__dict__)
            targs.model = self.args.transformer_model
            targs.features = "M"
            texp = Exp_Main(targs)
            texp.model.load_state_dict(torch.load(self.args.transformer_checkpoint))
            texp.model.eval()

        inputs = []
        trues = []
        tpreds = []
        preds = []

        self.model.eval()
        with torch.no_grad():
            for j, (batch_x, batch_y, batch_x_mark, batch_y_mark, mean, scale) in enumerate(pred_loader):
                if j != 5: continue
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if transformer_based: # and self.args.transformer_model != "FiLM":
                    x_dec = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    x_dec = torch.cat([batch_y[:, :self.args.label_len, :], x_dec], dim=1).float().to(self.device)
                    print(batch_x[:, -self.args.seq_len:, :].shape)
                    outputs = texp.model(
                        batch_x[:, -self.args.seq_len:, :],
                        batch_x_mark[:, -self.args.seq_len:, :],
                        x_dec,
                        batch_y_mark
                    )
                    tpreds.append(outputs.detach().cpu().numpy())
                    self.model.batch_size = self.args.batch_size
                    t = self.model.get_t(self.args.transformer_blurring).to(self.device)
                    dec_inp, _ = self.model.q_xt_x0(outputs, batch_y_mark, batch_x, batch_x_mark, t)
                
                batch_y = batch_y[:, -self.args.pred_len:]
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:]
                
                if not transformer_based:
                    # we blur batch_x as a starting point
                    t = self.model.get_t(self.model.num_steps - 1).to(self.device)
                    dec_inp, _ = self.model.q_xt_x0(batch_x, batch_y_mark, batch_x, batch_x_mark, t)

                pred = []
                channels = 1
                for i in range(0, self.args.c_out, channels):
                    subset = slice(i, i+channels)
                    b = batch_x[..., subset].shape[-1]
                    self.model.batch_size = self.args.batch_size * b
                    outputs = self.model.inference(
                        batch_x[..., subset].transpose(1, 2).reshape(-1, self.args.seq_len, 1),
                        batch_x_mark.repeat(b, 1, 1),
                        dec_inp[..., subset].transpose(1, 2).reshape(-1, self.args.pred_len, 1),
                        batch_y_mark.repeat(b, 1, 1),
                        steps=self.args.transformer_deblurring if transformer_based else None
                    )
                        
                    pred.append(outputs.detach().reshape(-1, b, self.args.pred_len).transpose(1, 2).cpu().numpy())
                pred = np.concatenate(pred, axis=-1)

                inputs.append(batch_x.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
                preds.append(pred)

                if j % 100 == 0:
                    print('predicting {}/{}'.format(i, len(pred_loader)))

        inputs = np.concatenate(inputs, axis=0)
        trues = np.concatenate(trues, axis=0)
        tpreds = np.concatenate(tpreds, axis=0)
        preds = np.concatenate(preds, axis=0)

        # mse
        mse = np.mean((preds - trues) ** 2)
        # mae
        mae = np.mean(np.abs(preds - trues))
        print('mse:{}, mae:{}'.format(mse, mae))

        # result save
        folder_path = './results/' + self.args.transformer_setting \
            + f"blur{self.args.transformer_blurring}_deblur{self.args.transformer_deblurring}" + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.savez(folder_path + 'real_prediction.npz', inputs=inputs, preds=preds, trues=trues)
        # np.savez(folder_path + 'real_prediction.npz', inputs=inputs, tpreds=tpreds, preds=preds, trues=trues)
        return
