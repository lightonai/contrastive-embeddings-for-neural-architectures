
# This file contains all the code for generating networks in the NATS / bench201 search spaces.
# The full code, subject to bug fixes etc is available from the original authors at: https://github.com/D-X-Y/AutoDL-Projects


import numpy as np
import importlib.resources as resources
import SearchSpaces.Bench201
import SearchSpaces.BenchNatsss


with resources.open_binary(SearchSpaces.BenchNatsss, "cifar10_accs.npy") as f:
    nats_cifar10 = np.load(f)

with resources.open_binary(SearchSpaces.BenchNatsss, "cifar100_accs.npy") as f:
    nats_cifar100 = np.load(f)

with resources.open_binary(SearchSpaces.BenchNatsss, "ImageNet16-120.npy") as f:
    nats_imgnet = np.load(f)


with resources.open_binary(SearchSpaces.Bench201, "cifar10_test_accs.npy") as f:
    b201_cifar10 = np.load(f)

with resources.open_binary(SearchSpaces.Bench201, "cifar100_test_accs.npy") as f:
    b201_cifar100 = np.load(f)

with resources.open_binary(SearchSpaces.Bench201, "ImageNet16-120_test_accs.npy") as f:
    b201_imgnet = np.load(f)


all_metrics = \
    dict({'nats_ss': dict({'CIFAR10': nats_cifar10, 'CIFAR100': nats_cifar100, 'ImageNet16': nats_imgnet}),
          '201':  dict({'CIFAR10': b201_cifar10, 'CIFAR100': b201_cifar100, 'ImageNet16': b201_imgnet})})

def get_metrics(search_space, task, index = None):
    if index is not None:
        return all_metrics[search_space][task][index]
    else:
        return all_metrics[search_space][task] # return numpy array with all architectures if no index is supplied.

def get_metric_names(search_space):
    if search_space == 'nats_ss' or search_space == '201':
        return ['CIFAR10', 'CIFAR100', 'ImageNet16']
    else:
        assert(False)



