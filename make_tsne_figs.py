
import argparse
parser = argparse.ArgumentParser(description='Generate prediction plot based on subset of samples')
parser.add_argument('path_EPDJMs', type=str)
parser.add_argument('task', choices=['CIFAR10', 'CIFAR100', 'ImageNet16'])
parser.add_argument('search_space', choices=['201', 'nats_ss'])
args = parser.parse_args()

import SearchSpaces

from common import ContrastiveNetwork, train_contrastive_network, get_embs, fit_surrogate, predict_surrogate
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()

from torch.utils.data import Dataset
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



data_set = CachedProjectedJacobians(path = args.path_EPDJMs, num_augs = 2)
net = ContrastiveNetwork(data_set[0].shape[-1],  emb_size = 256, projection_head_out_size=256, dim_hidden = 256)
net.cuda()

indices = np.random.choice(len(data_set), 1500, False)
jacs = data_set.data[indices, 0].reshape((len(indices), -1))
pre_embs = get_embs(net, data_set, indices)[:,0]

train_contrastive_network(net, data_set, batch_size=1024, epochs=30)

data_set = CachedProjectedJacobians(path = args.path_EPDJMs, num_augs = 1)
post_embs = get_embs(net, data_set, indices)[:,0]
validation = SearchSpaces.get_metrics(args.search_space, args.task)[indices]

for name, high_dim in [("jacs", jacs), ("pre_embs", pre_embs), ("post_embs", post_embs)]:
    plt.figure(figsize=(6,5))
    res = TSNE(n_components=2).fit_transform(high_dim).T
    plt.scatter(*res, s=1, c = validation)
    cbar = plt.colorbar(label = 'test accuracy', pad = 0.02)
    cbar.set_label('test accuracy', labelpad=10)
    plt.tight_layout()
    plt.savefig("tsne_"+name+".pdf")
    np.save("tsne_"+name+".npy", res)