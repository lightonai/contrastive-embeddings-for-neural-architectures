import argparse
parser = argparse.ArgumentParser(description='Generate prediction plot based on subset of samples')
parser.add_argument('path_EPDJMs', type=str)
parser.add_argument('task', choices=['CIFAR10', 'CIFAR100', 'ImageNet16'])
parser.add_argument('search_space', choices=['201', 'nats_ss'])

args = parser.parse_args()



from common import ContrastiveNetwork,train_contrastive_network, get_embs, fit_surrogate, predict_surrogate
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import SearchSpaces
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

def eval_embs(embs, validation, num_train, num_vals=2000,  num_augs = 4):
    all = np.random.choice(len(embs),num_train+num_vals,replace=False)
    train = all[:num_train]
    val = all[num_train:]
    surrogate = fit_surrogate(embs[train], validation[train], num_augs = num_augs, method = 'lgb')
    validation_pred = predict_surrogate(surrogate, embs[val], num_augs = num_augs, method = 'lgb')
    return (validation[val], validation_pred)


data_set = CachedProjectedJacobians(path = args.path_EPDJMs, num_augs =2)
net = ContrastiveNetwork(data_set[0].shape[-1],  emb_size = 256, projection_head_out_size=256, dim_hidden = 256)
net.cuda()

train_contrastive_network(net, data_set, batch_size=1024, epochs=30)

data_set = CachedProjectedJacobians(path = args.path_EPDJMs, num_augs = 4)
embs = get_embs(net,data_set)

validation = SearchSpaces.get_metrics(args.search_space, args.task)

x,y = eval_embs(embs, validation, 500)
plt.figure(figsize=(6,5))
plt.scatter(x,y, s=1)
plt.xlabel('Accuracy')
plt.ylabel("Predicted Accuracy")
plt.savefig("correlation_plot.pdf")
np.save("correlation_plot.npy", (x,y))
