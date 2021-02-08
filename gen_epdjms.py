import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import argparse
from einops import rearrange
from torch.utils.data import Dataset
import SearchSpaces
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate the Extend Projected Data Jacobian Matrix (EPDJM) for a given search space')
parser.add_argument('dataset', choices=['CIFAR10', 'CIFAR100', 'ImageNet16'])
parser.add_argument('benchmark', choices=['201', 'nats_ss'])
parser.add_argument('--proj_dim', type=int,  default=128)
parser.add_argument('--num_jacobians', type=int, default=256)
parser.add_argument('--num_augs', type=int,  default = 4)
args = parser.parse_args()

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if args.dataset == 'CIFAR10':
    cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_loader  = torch.utils.data.DataLoader(cifar10_dataset, batch_size=args.num_jacobians, shuffle=True)
    batch = next(iter(cifar10_loader))[0] 
    input_size = 3*32*32
elif args.dataset == 'CIFAR100':
    cifar10_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    cifar10_loader  = torch.utils.data.DataLoader(cifar10_dataset, batch_size=args.num_jacobians, shuffle=True)
    batch = next(iter(cifar10_loader))[0] 
    input_size = 3*32*32
else:
    assert(False)

class EDJM(Dataset):
    def __init__(self, dataset_batch, search_space, num_augs):
        self.dataset_batch = dataset_batch.cuda()
        self.dataset_batch.requires_grad = True
        self.num_augs = num_augs
        self.search_space = search_space

    def __len__(self):
        return SearchSpaces.get_num_networks(self.search_space)

    def get_jacobians(self, net):
        if self.dataset_batch.grad != None:
            self.dataset_batch.grad.data.zero_()
        _, output = net(self.dataset_batch)
        output.sum().backward()
        jacs = self.dataset_batch.grad.clone()
        return rearrange(jacs, 'batch ... -> batch (...)')
        

    def __getitem__(self, index):

        ret = []
        for _ in range(self.num_augs):
            net = SearchSpaces.get_network(self.search_space, index)
            net.cuda()
            j = self.get_jacobians(net)
            ret += [j]
        return ret

jacobian_dataset = EDJM(batch, args.benchmark, num_augs = args.num_augs)

all_samples = []
for i in tqdm(range(len(jacobian_dataset))):
    try:
        data = jacobian_dataset[i]

        if args.proj_dim != 0:
            u,s,_ = torch.svd(torch.stack(data))      
            sample = (u[:, :, :args.proj_dim] @ torch.diag_embed(s[:, :args.proj_dim])).cpu().numpy()
        else:
            sample = np.array([x.cpu().numpy() for x in data])

        all_samples+= [sample]
    except RuntimeError as e:
        all_samples+= [None]
        print("exception:", e)

np.save("./data/"+args.benchmark,np.array(all_samples))