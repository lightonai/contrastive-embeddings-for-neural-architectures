from common import ContrastiveNetwork,train_contrastive_network, get_embs, seed
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import kendalltau
from tqdm import tqdm
import torch
from einops import rearrange, reduce
import sklearn
import lightgbm as lgb 
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import argparse
import SearchSpaces

parser = argparse.ArgumentParser(description='Generate transfer learning plots between two search spaces.')
parser.add_argument('path_EPDJM_a', type=str)
parser.add_argument('path_EPDJM_b', type=str)
parser.add_argument('task_a', choices=['CIFAR10', 'CIFAR100', 'ImageNet16'])
parser.add_argument('task_b', choices=['CIFAR10', 'CIFAR100', 'ImageNet16'])
parser.add_argument('search_space_a', choices=['201', 'nats_ss'])
parser.add_argument('search_space_b', choices=['201', 'nats_ss'])
args = parser.parse_args()

epdjms_search_space_a = args.path_EPDJM_a
epdjms_search_space_b = args.path_EPDJM_b

accuracies_a = SearchSpaces.get_metrics(args.search_space_a, args.task_a)
accuracies_b = SearchSpaces.get_metrics(args.search_space_b, args.task_b)

seed(seed = 42)
sns.set()

from torch.utils.data import Dataset
class CachedProjectedJacobians(Dataset):

    def __init__(self, num_augs, emb_path, val, num_jacs = -1):

        self.data = np.load(emb_path, mmap_mode='r')
        self.data_augs = self.data[0].shape[0]
        self.proj_size = self.data[0].shape[2]
        self.num_jacs = num_jacs if num_jacs != -1 else self.data[0].shape[1]
        
        self.num_augs = num_augs
        self.val = val
        

        assert(self.num_augs <= self.data_augs)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        e = self.data[index][np.random.choice(self.data_augs, self.num_augs, replace=False), :self.num_jacs]
        if np.any(np.isnan(e)):
            e.fill(0)

        if self.val is not None:
            return e, [self.val[index]]*self.num_augs
        else:
            return e


class CombinedData():
    def __init__(self, num_augs, emb_path_a, val_a , emb_path_b, val_b):
        self.a = CachedProjectedJacobians(num_augs, emb_path_a , val_a)
        self.b = CachedProjectedJacobians(num_augs, emb_path_b , val_b, num_jacs = self.a.num_jacs)

        
    def __len__(self):
        return len(self.a)+ len(self.b)

    def __getitem__(self, index):
        if index < len(self.a):
            return self.a[index]
        else:
            return self.b[index-len(self.a)]


data_set = CombinedData(num_augs = 2, 
    emb_path_a = epdjms_search_space_a, val_a = None, 
    emb_path_b = epdjms_search_space_b, val_b = None)


net = ContrastiveNetwork(data_set[0][0].shape[-1],  emb_size = 256, projection_head_out_size=256, dim_hidden = 256)
net.cuda()

train_contrastive_network(net, data_set, batch_size=256, epochs=1)

def get_embs_():
    net.eval()
    data_set = CombinedData(num_augs = 4, 
        emb_path_a = epdjms_search_space_a, val_a = accuracies_a, 
        emb_path_b = epdjms_search_space_b, val_b = accuracies_b)

    embs = []
    vals = []
    for c in tqdm(range(len(data_set))):
        emb, val = data_set.__getitem__(c)
        a, _ = net(torch.tensor(emb).cuda())
        embs += [a.detach().cpu().numpy()]
        vals += [val]

    embs = np.array(embs)
    vals = np.array(vals)
    return embs, vals

all_embs, all_vals = get_embs_()


def get_embs(indices, num_augs):
    accs = rearrange(all_vals[indices, :num_augs], "b augs ... -> (b augs) ...")
    embs = rearrange(all_embs[indices, :num_augs], "b augs ... -> (b augs) ...")
    return accs, embs

a_len = len(data_set.a)
b_len = len(data_set.b)
a_start = 0
b_start = a_len

num_train = 10000
num_val = 3000
indices = np.random.choice(a_len, num_train+num_val, replace = False)
a_train = indices[:num_train]+a_start
a_val = indices[num_train:]+a_start


num_train = 10000
num_val = 3000
indices = np.random.choice(b_len, num_train+num_val, replace = False)
b_train = indices[:num_train]+b_start
b_val = indices[num_train:]+b_start

names = ["a", "b"]
train = [a_train, b_train]
val  = [a_val, b_val]

def fit_surrogate(indices, num_augs = 4, method = 'bo'):
    accs, embs = get_embs(indices, num_augs = num_augs)
        
    if method == 'rf':
        rf = RandomForestRegressor(max_features = 8) # not a tunder hyper paramete, using many is just slow and I don't like waiting. That said this is fine.
        rf.fit(embs, accs)
        return rf
    elif method == 'lgb':
        return lgb.train({'objective': 'regression', 'verbosity':-1}, lgb.Dataset(embs, label=accs))
    
    assert(False)

def predict_surrogate(surrogate, indices, num_augs = 4, method = 'bo'):
    accs, embs = get_embs(indices, num_augs = num_augs)
        
    if method == 'rf':
        predicted = surrogate.predict(embs)
    elif method == 'lgb':
        predicted = surrogate.predict(embs)
    else: 
        assert(False)
    
    return reduce(predicted,  "(b augs)-> b", 'mean', augs = num_augs), reduce(accs,  "(b augs)-> b", 'mean', augs = num_augs)


print(all_embs.shape)
print(all_vals.shape)


num_runs = 10
corrs = np.zeros((2, 2, 10))
taus  = np.zeros((2, 2, 10))

for run in range(10):

    na = 4
    for i in range(2):
        for j in range(2):
            train_space =  names[i]
            val_space   = names[j]
            m = fit_surrogate(train[i], method = 'rf')
            predicted, accs = predict_surrogate(m, val[j], method = 'rf')
            tau, corr = kendalltau(predicted, accs)[0], np.corrcoef(predicted,accs)[0,1]
            print(train_space + "--" + val_space," ", corr, tau)
            corrs[i,j, run] = corr
            taus [i,j, run] = tau


            if run == 0:
                plt.figure(figsize=(5,5))
                plt.scatter(accs,predicted , s=1)
                plt.xlabel("Accuracy", fontsize=14)
                plt.ylabel("Predicted Accuracy", fontsize=14)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.tight_layout()
                plt.savefig("transfer_learning_" + train_space + "--" + val_space + ".pdf")

    print("\n")

np.save("transfer_learning_metrics.npy", [corrs, taus])