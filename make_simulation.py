from common import ContrastiveNetwork,train_contrastive_network, get_embs, fit_surrogate, predict_surrogate, sim_one_run, embs_and_accs_function
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from einops import rearrange, reduce, repeat
from tqdm import tqdm
import torch.nn as nn
import argparse
import SearchSpaces
sns.set()

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
parser = argparse.ArgumentParser(description='Generate prediction plot based on subset of samples')
parser.add_argument('path_EPDJMs', type=str)
parser.add_argument('task', choices=['CIFAR10', 'CIFAR100', 'ImageNet16'])
parser.add_argument('search_space', choices=['201', 'nats_ss'])
parser.add_argument('--num_simulations', type=int, default = 500)
parser.add_argument('--evaluation_budget', type=int, default = 200)
parser.add_argument('--lengthscale', type = float, default =1)
parser.add_argument('--noise', type = float, default =0.1)
parser.add_argument('--ei_offset', type = float, default =0.15)
args = parser.parse_args()

class CachedProjectedJacobians(Dataset):

    def __init__(self, path, num_augs):
        self.data = np.load(path, mmap_mode='r')
        self.data_augs = self.data[0].shape[0]
        self.proj_size = self.data[0].shape[2]
        self.num_augs = num_augs
        assert(num_augs <= self.data_augs)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, np.random.choice(self.data_augs, self.num_augs, replace=False)]

def sns_lineplot(x, y, *args, **kwargs): #sns doesn't broadcast 1D x to match 2D y, so we do that here.
    l = len(x) 
    assert(y.shape[-1] == l)
    y.reshape(-1, l)
    n = len(y)
    x = repeat(np.array(x), "x -> (b x)", b = n)
    sns.lineplot(x, y.flatten(), *args, **kwargs)


data_set = CachedProjectedJacobians(path = args.path_EPDJMs, num_augs =2)
net = ContrastiveNetwork(data_set[0].shape[-1],  emb_size = 256, projection_head_out_size=256, dim_hidden = 256)
net.cuda()

train_contrastive_network(net, data_set, batch_size=1024, epochs=30)

data_set = CachedProjectedJacobians(path = args.path_EPDJMs, num_augs = 4)
embs = get_embs(net,data_set)

validation = SearchSpaces.get_metrics(args.search_space, args.task)

plt.figure(figsize=(6,5))

data_func = embs_and_accs_function(embs, validation)

runs  = [sim_one_run(data_func, args.evaluation_budget, num_randoms = 5, num_augs = 4,  lengthscale = args.lengthscale, noise = args.noise, ei_offset = args.ei_offset) for _ in tqdm(range(args.num_simulations))]
runs = np.maximum.accumulate(runs, axis = 1)

random_runs  = [sim_one_run(data_func, args.evaluation_budget, num_randoms = args.evaluation_budget, num_augs = 1) for _ in tqdm(range(args.num_simulations))]
random_runs = np.maximum.accumulate(random_runs, axis = 1)

sns_lineplot(np.arange(args.evaluation_budget), runs, label = 'Contrasive Embeddings + GP')
sns_lineplot(np.arange(args.evaluation_budget), random_runs, label = 'Random Search')

plt.ylim(bottom = random_runs[:, 5].mean())

plt.xlabel("Number of Evaluated Architectures")
plt.ylabel("Accuracy of Best Architecture")

plt.savefig("simulations"+args.task + ".pdf")
np.save("simulations" + args.task + ".npy", (runs, random_runs))
