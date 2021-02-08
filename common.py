import torch
import numpy as np
from torch import nn

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from tqdm import tqdm

import torch.nn.functional as F
import random
import os

class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature = 0.5, use_cosine_similarity = True):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
    def forward(self, reps): #assumes that we have two different "augmentations" after each other each other in the batch dim. So real dim is batch_dim/2
        if self.use_cosine_similarity:
            reps = F.normalize(reps, dim = -1)
        sim_mat = (reps @ reps.T) / self.temperature
        sim_mat.fill_diagonal_(-np.inf) #we cannot predict oursleves.
        batch_size = reps.shape[0]//2
        labels = torch.cat([torch.arange(batch_size)+batch_size, torch.arange(batch_size)]) # positive samples are one batch away 
        labels = labels.to(reps.device)
        return F.cross_entropy(sim_mat, labels)

class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

class ContrastiveNetwork(nn.Module):
    def __init__(self, input_size, projection_head_out_size = 128, emb_size = 256, dim_hidden = 128):
        super(ContrastiveNetwork, self).__init__()
        self.model_emb = model = nn.Sequential(DeepSet(input_size, 1, emb_size, dim_hidden = dim_hidden), Rearrange("a b c -> a (b c)"))
        self.projection_head = nn.Sequential(
            nn.Linear(emb_size, projection_head_out_size)
        )
        
    def forward(self, x):
        x = self.model_emb(x)
        p = self.projection_head(F.relu(x))
        return x, p

def train_contrastive_network(net, data_set, batch_size = 256, epochs = 10, num_workers = 0, lr = 0.5e-4):
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95**epoch)

    val_size = 1028
    train_set, val_set = torch.utils.data.random_split(data_set, [len(data_set)-val_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, drop_last = True, shuffle = True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, drop_last = True, shuffle = True, pin_memory=True)

    criterion = NTXentLoss(temperature = 0.1)

    losses = []
    val_losses = []

    for epoch in range(epochs):  
        for i, data in enumerate(train_loader, 0):

            net.train()
            data = data.cuda()
            optimizer.zero_grad()

            _, projs = net(rearrange(data, "a b ... -> (b a) ..."))
            loss = criterion(projs)
            loss.backward()
            optimizer.step()

            losses += [loss.item()]
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, losses[-1]))

        val_loss = 0
        for i, data in enumerate(val_loader, 0):
            net.eval()
            data = data.cuda()
            _, projs = net(rearrange(data, "a b ... -> (b a) ..."))
            val_loss += criterion(projs).item() / len(val_loader)

        print("val_loss:", val_loss)
        val_losses+=[val_loss]
        lr_schedule.step()



def get_embs(net, data_set, indices = None):
    net.eval()
    embs = []

    if indices is None:
        indices = np.arange(len(data_set))
    
    for idx in tqdm(indices):
        e, _ = net(torch.tensor(data_set[idx]).cuda())
        embs += [e.detach().cpu().numpy()]

    return np.array(embs)


import lightgbm as lgb 
import sklearn
from scipy.stats import kendalltau

from sklearn.ensemble import RandomForestRegressor

def fit_surrogate(embs, vals, num_augs = 4, norm_embs = True, method = 'bo'):
    import GPy
    
    embs = rearrange(embs[:, :num_augs], "a b ... -> (a b) ...")
    accs = repeat(vals, "a -> (a b)", b=num_augs)
    
    if method == 'bo':
        kernel = GPy.kern.Matern52(input_dim=128, lengthscale = 1)
        m = GPy.models.gp_regression.GPRegression(embs,accs.reshape(-1,1), noise_var = 0.05, kernel = kernel)
        return m
    if method == 'rf':
        rf = RandomForestRegressor()
        rf.fit(embs, accs)
        return rf
    if method == 'xgb':
        xg_reg = xgb.XGBRegressor(objective ='reg:squarederror')
        xg_reg.fit(embs,accs)
        return xg_reg
    if method == 'lgb':
        return lgb.train({'objective': 'regression', 'verbosity':-1}, lgb.Dataset(embs, label=accs))
    if method == 'rank_nn':
        return fit_rank_network(embs, accs)


def predict_surrogate(surrogate, embs, num_augs=4, norm_embs = True, method = 'rf'):
    
    embs = rearrange(embs[:,:num_augs], "a b ... -> (a b) ...")
    
    if method == 'bo':
        predicted = surrogate.predict(embs)[0].T[0]
    if method == 'rf':
        predicted = surrogate.predict(embs)
    if method == 'xgb':
        predicted = surrogate.predict(embs)
    if method == 'lgb':
        predicted = surrogate.predict(embs)
    if method == 'rank_nn':
        predicted = predict_rank_network(surrogate, embs)

    return reduce(predicted,  "(b augs)-> b", 'mean', augs = num_augs)

def embs_and_accs_function(embs, vals):
    def f(idx, num_augs, embs = embs, vals = vals):
        assert(num_augs <= len(embs[0]) and "num_augs must be smaller or equal to the number of augmentations in the underlying data")
        embs_ret = rearrange(embs[idx, :num_augs], "a b ... -> (a b) ...")
        accs_ret = repeat(vals[idx], "a -> (a b)", b=num_augs)
        return accs_ret, embs_ret
    f.len = len(embs)
    return f

        
from scipy.stats import norm
n = norm()
def EI(mean, std, best):
    z = (mean-best)/std
    return (mean-best)*n.cdf(z) + std * n.pdf(z)

def sim_one_run(data_func, num_trials, num_randoms = 5, num_augs = 4, lengthscale=1, noise = 0.05, num_candidates = 20, ei_offset = 0.2):
    import GPy
    archs = []
    num_archs = data_func.len
    embs = []
    accs = []
    useGPU = False
    for i in range(num_trials):
        if i < num_randoms:
            archs += [np.random.randint(num_archs)]
        else:
            if not useGPU:
                kernel = GPy.kern.Matern52(input_dim=len(embs[0]), lengthscale = lengthscale)
            else:
                kernel = GPy.kern.RBF(input_dim=len(embs[0]), lengthscale = lengthscale, useGPU=True)


            m = GPy.models.gp_regression.GPRegression(np.array(embs),np.array(accs).reshape(-1,1), noise_var = noise, kernel = kernel)
            candidates = np.random.randint(num_archs, size = num_candidates)
            actual_acc, cand_embs = data_func(candidates, num_augs)

            m_pred, m_var = m.predict(cand_embs)
            predicted     = reduce(m_pred.T[0],  "(b augs)-> b", 'mean', augs = num_augs)
            predicted_var = reduce(m_var.T[0],   "(b augs)-> b", 'mean', augs = num_augs)
            ei = EI(predicted, np.sqrt(predicted_var), np.max(accs)+ei_offset)
            archs += [candidates[np.argmax(ei)]]

        
        acc, emb = data_func([archs[-1]], num_augs)
        embs += [*emb]
        accs += [*acc]
    return np.array(accs[::num_augs])

def seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    

def sns_lineplot(x, y, *args, **kwargs): #sns doesn't broadcast 1D x to match 2D y, so we do that here.
    l = len(x) 
    assert(y.shape[-1] == l)
    y.reshape(-1, l)
    n = len(y)
    x = repeat(np.array(x), "x -> (b x)", b = n)
    sns.lineplot(x, y.flatten(), *args, **kwargs)
