import torch
import torch.nn as nn
import torch.nn.functional as F


# implementation of the model described in the paper
# https://hexiangnan.github.io/papers/kdd19-timeseries.pdf


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.gru = nn.GRU(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            batch_first=True
        )
        self.wgru = nn.GRU(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            batch_first=True
        )
        self.dense = nn.Linear(configs.d_model, configs.c_out)
        self.pproj = nn.Linear(configs.d_model, 1)
        self.corr = nn.Linear(1, configs.c_out)

        self.window_size = configs.window_size
        self.num_windows = configs.num_windows
        self.pred_len = configs.pred_len
        self.l2loss_ = None

    def extreme_values(self, x):
        # we consider the top 5% values as being extreme
        std = torch.std(x, dim=1, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        threshold = mean + 1.48 * std
        ev = x > threshold
        return ev.float()
    
    def l1(self, o, y, u, v, include_l2=True):
        mse = F.mse_loss(o, y)
        evl = 2*EVL(u, v)
        if include_l2 and self.l2loss_ is not None:
            return mse + evl + self.l2loss_
        return mse + evl
    
    def l2(self, ps, qs):
        # ps: list of tensors of shape (batch_size, num_windows, 1)
        # qs: list of tensors of shape (batch_size, num_windows, 1)
        ps = torch.cat(ps, dim=0)
        qs = torch.cat(qs, dim=0)
        loss = torch.sum(torch.vstack([EVL(p, q) for p, q in zip(ps, qs)]))
        return loss

    def onestep(self, x, initial_state=None):
        # x shape should be (batch_size, window_size, input_size=1)
        if initial_state is None:
            _, H = self.gru(x)
        else:
            _, H = self.gru(x[:, :-1], initial_state)
        H = H.squeeze(0)
        o = self.dense(H).unsqueeze(1)

        # sample `num_windows` windows from x:
        batch_size = x.shape[0]
        window_size = self.window_size
        num_windows = self.num_windows
        window_starts = torch.randint(0, x.shape[1] - window_size, (batch_size, num_windows), device=x.device)
        window_indices = window_starts[:, :, None] + torch.arange(0, window_size, device=x.device)[None, None, :]
        windows = torch.vstack([x[i, window_indices[i]] for i in range(batch_size)])

        _, S = self.wgru(windows)
        S = S.reshape(batch_size, num_windows, -1)
        P = torch.sigmoid(self.pproj(S))[..., 0]

        A = torch.softmax(torch.einsum("bh,bmh->bm", H, S), dim=-1)
        ev = self.extreme_values(x).squeeze(-1)
        Q = torch.gather(ev, dim=1, index=window_starts + window_size)
        u = torch.sigmoid(self.corr(Q[:, None, :] @ A[:, :, None]))

        y = o + u
        return y, H.unsqueeze(0), P, Q, u
    
    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        # x shape should be (batch_size, seq_len, input_size=1)
        x = batch_x
        state = None
        ps = []
        qs = []
        us = []
        self.l2loss_ = None
        for i in range(self.pred_len):
            y, state, p, q, u = self.onestep(x, state)
            ps.append(p)
            qs.append(q)
            us.append(u)
            x = torch.cat([x, y], dim=1)
        if self.training:
            self.l2loss_ = self.l2(ps, qs)
        us = torch.cat(us, dim=1)
        return x[:, -self.pred_len:, :], us


def EVL(ut, vt, beta=0.05, gamma=2.0):
    # Here we assume ut is the predicted probability of an extreme event
    # vt is the true label (0 for normal, 1 for extreme)
    # beta0 and beta1 are respectively the proportions for the normal and extreme cases
    beta0 = 1 - beta
    beta1 = beta
    loss = - beta0 * ((1 - ut/gamma) ** gamma) * (ut + 1e-6).log() * vt \
           - beta1 * ((1 - (1 - ut)/gamma) ** gamma) * (1 - ut + 1e-6).log() * (1 - vt)
    return loss.mean()


if __name__ == "__main__":
    x = torch.randn(16, 168, 1)
    model = Model(1, 8, 24, 7)
    y = model(x)
    print(y.shape)