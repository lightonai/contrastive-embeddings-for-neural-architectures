
# This file contains all the code for generating networks in the NATS / bench201 search spaces.
# The full code, subject to bug fixes etc is available from the original authors at: https://github.com/D-X-Y/AutoDL-Projects


import numpy as np
import importlib.resources as resources
import SearchSpaces
import SearchSpaces.Bench201
import SearchSpaces.BenchNatsss

with resources.open_text(SearchSpaces.Bench201, "configs.txt") as f:
    b201_configs = f.read().split('\n')

with resources.open_binary(SearchSpaces.BenchNatsss, "arch_strings.npy") as f:
    nats_configs = np.load(f)

def get_network(search_space, idx, num_classes = 10):
    if search_space == '201':
        genotype = Structure.str2structure(b201_configs[idx])
        return TinyNetwork(16, 5, genotype, num_classes)
    elif search_space == 'nats_ss':
        genotype = Structure.str2structure('|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|')
        config = nats_configs[idx]
        channels = tuple([int(x) for x in config.split(':')])
        return DynamicShapeTinyNet(channels, genotype, num_classes)
    else:
        assert(False)

def get_num_networks(search_space):
    if search_space == '201':
        return len(b201_configs)
    elif search_space == 'nats_ss':
        return len(nats_configs)
    else:
        print('got search space:', search_space, 'expected either "201" or "nats_ss"\n')
        assert(False)

def get_search_space_names(search_space):
    return ['201', 'nats_ss']

# yank out all the stuff to be able to construct networks form the nas201 benchmark. 

from copy import deepcopy

__all__ = ["TinyNetwork"]


def get_combination(space, num):
  combs = []
  for i in range(num):
    if i == 0:
      for func in space:
        combs.append( [(func, i)] )
    else:
      new_combs = []
      for string in combs:
        for func in space:
          xstring = string + [(func, i)]
          new_combs.append( xstring )
      combs = new_combs
  return combs


class Structure:

  def __init__(self, genotype):
    assert isinstance(genotype, list) or isinstance(genotype, tuple), 'invalid class of genotype : {:}'.format(type(genotype))
    self.node_num = len(genotype) + 1
    self.nodes    = []
    self.node_N   = []
    for idx, node_info in enumerate(genotype):
      assert isinstance(node_info, list) or isinstance(node_info, tuple), 'invalid class of node_info : {:}'.format(type(node_info))
      assert len(node_info) >= 1, 'invalid length : {:}'.format(len(node_info))
      for node_in in node_info:
        assert isinstance(node_in, list) or isinstance(node_in, tuple), 'invalid class of in-node : {:}'.format(type(node_in))
        assert len(node_in) == 2 and node_in[1] <= idx, 'invalid in-node : {:}'.format(node_in)
      self.node_N.append( len(node_info) )
      self.nodes.append( tuple(deepcopy(node_info)) )

  def tolist(self, remove_str):
    # convert this class to the list, if remove_str is 'none', then remove the 'none' operation.
    # note that we re-order the input node in this function
    # return the-genotype-list and success [if unsuccess, it is not a connectivity]
    genotypes = []
    for node_info in self.nodes:
      node_info = list( node_info )
      node_info = sorted(node_info, key=lambda x: (x[1], x[0]))
      node_info = tuple(filter(lambda x: x[0] != remove_str, node_info))
      if len(node_info) == 0: return None, False
      genotypes.append( node_info )
    return genotypes, True

  def node(self, index):
    assert index > 0 and index <= len(self), 'invalid index={:} < {:}'.format(index, len(self))
    return self.nodes[index]

  def tostr(self):
    strings = []
    for node_info in self.nodes:
      string = '|'.join([x[0]+'~{:}'.format(x[1]) for x in node_info])
      string = '|{:}|'.format(string)
      strings.append( string )
    return '+'.join(strings)

  def check_valid(self):
    nodes = {0: True}
    for i, node_info in enumerate(self.nodes):
      sums = []
      for op, xin in node_info:
        if op == 'none' or nodes[xin] is False: x = False
        else: x = True
        sums.append( x )
      nodes[i+1] = sum(sums) > 0
    return nodes[len(self.nodes)]

  def to_unique_str(self, consider_zero=False):
    # this is used to identify the isomorphic cell, which rerquires the prior knowledge of operation
    # two operations are special, i.e., none and skip_connect
    nodes = {0: '0'}
    for i_node, node_info in enumerate(self.nodes):
      cur_node = []
      for op, xin in node_info:
        if consider_zero is None:
          x = '('+nodes[xin]+')' + '@{:}'.format(op)
        elif consider_zero:
          if op == 'none' or nodes[xin] == '#': x = '#' # zero
          elif op == 'skip_connect': x = nodes[xin]
          else: x = '('+nodes[xin]+')' + '@{:}'.format(op)
        else:
          if op == 'skip_connect': x = nodes[xin]
          else: x = '('+nodes[xin]+')' + '@{:}'.format(op)
        cur_node.append(x)
      nodes[i_node+1] = '+'.join( sorted(cur_node) )
    return nodes[ len(self.nodes) ]

  def check_valid_op(self, op_names):
    for node_info in self.nodes:
      for inode_edge in node_info:
        #assert inode_edge[0] in op_names, 'invalid op-name : {:}'.format(inode_edge[0])
        if inode_edge[0] not in op_names: return False
    return True

  def __repr__(self):
    return ('{name}({node_num} nodes with {node_info})'.format(name=self.__class__.__name__, node_info=self.tostr(), **self.__dict__))

  def __len__(self):
    return len(self.nodes) + 1

  def __getitem__(self, index):
    return self.nodes[index]

  @staticmethod
  def str2structure(xstr):
    if isinstance(xstr, Structure): return xstr
    assert isinstance(xstr, str), 'must take string (not {:}) as input'.format(type(xstr))
    nodestrs = xstr.split('+')
    genotypes = []
    for i, node_str in enumerate(nodestrs):
      inputs = list(filter(lambda x: x != '', node_str.split('|')))
      for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
      inputs = ( xi.split('~') for xi in inputs )
      input_infos = tuple( (op, int(IDX)) for (op, IDX) in inputs)
      genotypes.append( input_infos )
    return Structure( genotypes )

  @staticmethod
  def str2fullstructure(xstr, default_name='none'):
    assert isinstance(xstr, str), 'must take string (not {:}) as input'.format(type(xstr))
    nodestrs = xstr.split('+')
    genotypes = []
    for i, node_str in enumerate(nodestrs):
      inputs = list(filter(lambda x: x != '', node_str.split('|')))
      for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
      inputs = ( xi.split('~') for xi in inputs )
      input_infos = list( (op, int(IDX)) for (op, IDX) in inputs)
      all_in_nodes= list(x[1] for x in input_infos)
      for j in range(i):
        if j not in all_in_nodes: input_infos.append((default_name, j))
      node_info = sorted(input_infos, key=lambda x: (x[1], x[0]))
      genotypes.append( tuple(node_info) )
    return Structure( genotypes )

  @staticmethod
  def gen_all(search_space, num, return_ori):
    assert isinstance(search_space, list) or isinstance(search_space, tuple), 'invalid class of search-space : {:}'.format(type(search_space))
    assert num >= 2, 'There should be at least two nodes in a neural cell instead of {:}'.format(num)
    all_archs = get_combination(search_space, 1)
    for i, arch in enumerate(all_archs):
      all_archs[i] = [ tuple(arch) ]
  
    for inode in range(2, num):
      cur_nodes = get_combination(search_space, inode)
      new_all_archs = []
      for previous_arch in all_archs:
        for cur_node in cur_nodes:
          new_all_archs.append( previous_arch + [tuple(cur_node)] )
      all_archs = new_all_archs
    if return_ori:
      return all_archs
    else:
      return [Structure(x) for x in all_archs]



ResNet_CODE = Structure(
  [(('nor_conv_3x3', 0), ), # node-1 
   (('nor_conv_3x3', 1), ), # node-2
   (('skip_connect', 0), ('skip_connect', 2))] # node-3
  )

AllConv3x3_CODE = Structure(
  [(('nor_conv_3x3', 0), ), # node-1 
   (('nor_conv_3x3', 0), ('nor_conv_3x3', 1)), # node-2
   (('nor_conv_3x3', 0), ('nor_conv_3x3', 1), ('nor_conv_3x3', 2))] # node-3
  )

AllFull_CODE = Structure(
  [(('skip_connect', 0), ('nor_conv_1x1', 0), ('nor_conv_3x3', 0), ('avg_pool_3x3', 0)), # node-1 
   (('skip_connect', 0), ('nor_conv_1x1', 0), ('nor_conv_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 1), ('nor_conv_1x1', 1), ('nor_conv_3x3', 1), ('avg_pool_3x3', 1)), # node-2
   (('skip_connect', 0), ('nor_conv_1x1', 0), ('nor_conv_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 1), ('nor_conv_1x1', 1), ('nor_conv_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('nor_conv_1x1', 2), ('nor_conv_3x3', 2), ('avg_pool_3x3', 2))] # node-3
  )

AllConv1x1_CODE = Structure(
  [(('nor_conv_1x1', 0), ), # node-1 
   (('nor_conv_1x1', 0), ('nor_conv_1x1', 1)), # node-2
   (('nor_conv_1x1', 0), ('nor_conv_1x1', 1), ('nor_conv_1x1', 2))] # node-3
  )

AllIdentity_CODE = Structure(
  [(('skip_connect', 0), ), # node-1 
   (('skip_connect', 0), ('skip_connect', 1)), # node-2
   (('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2))] # node-3
  )

architectures = {'resnet'  : ResNet_CODE,
                 'all_c3x3': AllConv3x3_CODE,
                 'all_c1x1': AllConv1x1_CODE,
                 'all_idnt': AllIdentity_CODE,
                 'all_full': AllFull_CODE}


# The macro structure for architectures in NAS-Bench-201
import torch.nn as nn

class ResNetBasicblock(nn.Module):

  def __init__(self, inplanes, planes, stride, affine=True, track_running_stats=True):
    super(ResNetBasicblock, self).__init__()
    assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
    self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine, track_running_stats)
    self.conv_b = ReLUConvBN(  planes, planes, 3,      1, 1, 1, affine, track_running_stats)
    if stride == 2:
      self.downsample = nn.Sequential(
                           nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                           nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
    elif inplanes != planes:
      self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, affine, track_running_stats)
    else:
      self.downsample = None
    self.in_dim  = inplanes
    self.out_dim = planes
    self.stride  = stride
    self.num_conv = 2

  def extra_repr(self):
    string = '{name}(inC={in_dim}, outC={out_dim}, stride={stride})'.format(name=self.__class__.__name__, **self.__dict__)
    return string

  def forward(self, inputs):

    basicblock = self.conv_a(inputs)
    basicblock = self.conv_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(inputs)
    else:
      residual = inputs
    return residual + basicblock


# Cell for NAS-Bench-201
class InferCell(nn.Module):

  def __init__(self, genotype, C_in, C_out, stride, affine=True, track_running_stats=True):
    super(InferCell, self).__init__()

    self.layers  = nn.ModuleList()
    self.node_IN = []
    self.node_IX = []
    self.genotype = deepcopy(genotype)
    for i in range(1, len(genotype)):
      node_info = genotype[i-1]
      cur_index = []
      cur_innod = []
      for (op_name, op_in) in node_info:
        if op_in == 0:
          layer = OPS[op_name](C_in , C_out, stride, affine, track_running_stats)
        else:
          layer = OPS[op_name](C_out, C_out,      1, affine, track_running_stats)
        cur_index.append( len(self.layers) )
        cur_innod.append( op_in )
        self.layers.append( layer )
      self.node_IX.append( cur_index )
      self.node_IN.append( cur_innod )
    self.nodes   = len(genotype)
    self.in_dim  = C_in
    self.out_dim = C_out

  def extra_repr(self):
    string = 'info :: nodes={nodes}, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
    laystr = []
    for i, (node_layers, node_innods) in enumerate(zip(self.node_IX,self.node_IN)):
      y = ['I{:}-L{:}'.format(_ii, _il) for _il, _ii in zip(node_layers, node_innods)]
      x = '{:}<-({:})'.format(i+1, ','.join(y))
      laystr.append( x )
    return string + ', [{:}]'.format( ' | '.join(laystr) ) + ', {:}'.format(self.genotype.tostr())

  def forward(self, inputs):
    nodes = [inputs]
    for i, (node_layers, node_innods) in enumerate(zip(self.node_IX,self.node_IN)):
      node_feature = sum( self.layers[_il](nodes[_ii]) for _il, _ii in zip(node_layers, node_innods) )
      nodes.append( node_feature )
    return nodes[-1]


class TinyNetwork(nn.Module):

  def __init__(self, C, N, genotype, num_classes):
    super(TinyNetwork, self).__init__()
    self._C               = C
    self._layerN          = N

    self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C))
  
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev = C
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2, True)
      else:
        cell = InferCell(genotype, C_prev, C_curr, 1)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self._Layer= len(self.cells)

    self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def forward(self, inputs):
    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      feature = cell(feature)

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits

class ProbedTinyNetwork(nn.Module):

  def __init__(self, C, N, genotype, num_classes, probes = []):
    super(ProbedTinyNetwork, self).__init__()
    self._C               = C
    self._layerN          = N

    self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C))
  
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev = C
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2, True)
      else:
        cell = InferCell(genotype, C_prev, C_curr, 1)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self._Layer= len(self.cells)

    self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.probes = probes

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def forward(self, inputs):
    feature = self.stem(inputs)
    probe_features = []
    for i, cell in enumerate(self.cells):
      feature = cell(feature)
      if i in self.probes:
        probe_features += [feature.clone()]

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits, probe_features



  
##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch
import torch.nn as nn

__all__ = ['OPS', 'RAW_OP_CLASSES', 'ResNetBasicblock', 'SearchSpaceNames']

OPS = {
  'none'         : lambda C_in, C_out, stride, affine, track_running_stats: Zero(C_in, C_out, stride),
  'avg_pool_3x3' : lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, stride, 'avg', affine, track_running_stats),
  'max_pool_3x3' : lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, stride, 'max', affine, track_running_stats),
  'nor_conv_7x7' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (7,7), (stride,stride), (3,3), (1,1), affine, track_running_stats),
  'nor_conv_3x3' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (3,3), (stride,stride), (1,1), (1,1), affine, track_running_stats),
  'nor_conv_1x1' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (1,1), (stride,stride), (0,0), (1,1), affine, track_running_stats),
  'dua_sepc_3x3' : lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(C_in, C_out, (3,3), (stride,stride), (1,1), (1,1), affine, track_running_stats),
  'dua_sepc_5x5' : lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(C_in, C_out, (5,5), (stride,stride), (2,2), (1,1), affine, track_running_stats),
  'dil_sepc_3x3' : lambda C_in, C_out, stride, affine, track_running_stats: SepConv(C_in, C_out, (3,3), (stride,stride), (2,2), (2,2), affine, track_running_stats),
  'dil_sepc_5x5' : lambda C_in, C_out, stride, affine, track_running_stats: SepConv(C_in, C_out, (5,5), (stride,stride), (4,4), (2,2), affine, track_running_stats),
  'skip_connect' : lambda C_in, C_out, stride, affine, track_running_stats: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
}

CONNECT_NAS_BENCHMARK = ['none', 'skip_connect', 'nor_conv_3x3']
NAS_BENCH_201         = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
DARTS_SPACE           = ['none', 'skip_connect', 'dua_sepc_3x3', 'dua_sepc_5x5', 'dil_sepc_3x3', 'dil_sepc_5x5', 'avg_pool_3x3', 'max_pool_3x3']

SearchSpaceNames = {'connect-nas'  : CONNECT_NAS_BENCHMARK,
                    'nats-bench'   : NAS_BENCH_201,
                    'nas-bench-201': NAS_BENCH_201,
                    'darts'        : DARTS_SPACE}


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=not affine),
      nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)
    )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=not affine),
      nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
      )

  def forward(self, x):
    return self.op(x)


class DualSepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
    super(DualSepConv, self).__init__()
    self.op_a = SepConv(C_in, C_in , kernel_size, stride, padding, dilation, affine, track_running_stats)
    self.op_b = SepConv(C_in, C_out, kernel_size, 1, padding, dilation, affine, track_running_stats)

  def forward(self, x):
    x = self.op_a(x)
    x = self.op_b(x)
    return x


class ResNetBasicblock(nn.Module):

  def __init__(self, inplanes, planes, stride, affine=True, track_running_stats=True):
    super(ResNetBasicblock, self).__init__()
    assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
    self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine, track_running_stats)
    self.conv_b = ReLUConvBN(  planes, planes, 3,      1, 1, 1, affine, track_running_stats)
    if stride == 2:
      self.downsample = nn.Sequential(
                           nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                           nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
    elif inplanes != planes:
      self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, affine, track_running_stats)
    else:
      self.downsample = None
    self.in_dim  = inplanes
    self.out_dim = planes
    self.stride  = stride
    self.num_conv = 2

  def extra_repr(self):
    string = '{name}(inC={in_dim}, outC={out_dim}, stride={stride})'.format(name=self.__class__.__name__, **self.__dict__)
    return string

  def forward(self, inputs):

    basicblock = self.conv_a(inputs)
    basicblock = self.conv_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(inputs)
    else:
      residual = inputs
    return residual + basicblock


class POOLING(nn.Module):

  def __init__(self, C_in, C_out, stride, mode, affine=True, track_running_stats=True):
    super(POOLING, self).__init__()
    if C_in == C_out:
      self.preprocess = None
    else:
      self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine, track_running_stats)
    if mode == 'avg'  : self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    elif mode == 'max': self.op = nn.MaxPool2d(3, stride=stride, padding=1)
    else              : raise ValueError('Invalid mode={:} in POOLING'.format(mode))

  def forward(self, inputs):
    if self.preprocess: x = self.preprocess(inputs)
    else              : x = inputs
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, C_in, C_out, stride):
    super(Zero, self).__init__()
    self.C_in   = C_in
    self.C_out  = C_out
    self.stride = stride
    self.is_zero = True

  def forward(self, x):
    if self.C_in == self.C_out:
      if self.stride == 1: return x.mul(0.)
      else               : return x[:,:,::self.stride,::self.stride].mul(0.)
    else:
      shape = list(x.shape)
      shape[1] = self.C_out
      zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
      return zeros

  def extra_repr(self):
    return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, stride, affine, track_running_stats):
    super(FactorizedReduce, self).__init__()
    self.stride = stride
    self.C_in   = C_in  
    self.C_out  = C_out  
    self.relu   = nn.ReLU(inplace=False)
    if stride == 2:
      #assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
      C_outs = [C_out // 2, C_out - C_out // 2]
      self.convs = nn.ModuleList()
      for i in range(2):
        self.convs.append(nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=not affine))
      self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
    elif stride == 1:
      self.conv = nn.Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=not affine)
    else:
      raise ValueError('Invalid stride : {:}'.format(stride))
    self.bn = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)

  def forward(self, x):
    if self.stride == 2:
      x = self.relu(x)
      y = self.pad(x)
      out = torch.cat([self.convs[0](x), self.convs[1](y[:,:,1:,1:])], dim=1)
    else:
      out = self.conv(x)
    out = self.bn(out)
    return out

  def extra_repr(self):
    return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


# Auto-ReID: Searching for a Part-Aware ConvNet for Person Re-Identification, ICCV 2019
class PartAwareOp(nn.Module):

  def __init__(self, C_in, C_out, stride, part=4):
    super().__init__()
    self.part   = 4
    self.hidden = C_in // 3
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.local_conv_list = nn.ModuleList()
    for i in range(self.part):
      self.local_conv_list.append(
            nn.Sequential(nn.ReLU(), nn.Conv2d(C_in, self.hidden, 1), nn.BatchNorm2d(self.hidden, affine=True))
            )
    self.W_K = nn.Linear(self.hidden, self.hidden)
    self.W_Q = nn.Linear(self.hidden, self.hidden)

    if stride == 2  : self.last = FactorizedReduce(C_in + self.hidden, C_out, 2)
    elif stride == 1: self.last = FactorizedReduce(C_in + self.hidden, C_out, 1)
    else:             raise ValueError('Invalid Stride : {:}'.format(stride))

  def forward(self, x):
    batch, C, H, W = x.size()
    assert H >= self.part, 'input size too small : {:} vs {:}'.format(x.shape, self.part)
    IHs = [0]
    for i in range(self.part): IHs.append( min(H, int((i+1)*(float(H)/self.part))) )
    local_feat_list = []
    for i in range(self.part):
      feature = x[:, :, IHs[i]:IHs[i+1], :]
      xfeax   = self.avg_pool(feature)
      xfea    = self.local_conv_list[i]( xfeax )
      local_feat_list.append( xfea )
    part_feature = torch.cat(local_feat_list, dim=2).view(batch, -1, self.part)
    part_feature = part_feature.transpose(1,2).contiguous()
    part_K       = self.W_K(part_feature)
    part_Q       = self.W_Q(part_feature).transpose(1,2).contiguous()
    weight_att   = torch.bmm(part_K, part_Q)
    attention    = torch.softmax(weight_att, dim=2)
    aggreateF    = torch.bmm(attention, part_feature).transpose(1,2).contiguous()
    features = []
    for i in range(self.part):
      feature = aggreateF[:, :, i:i+1].expand(batch, self.hidden, IHs[i+1]-IHs[i])
      feature = feature.view(batch, self.hidden, IHs[i+1]-IHs[i], 1)
      features.append( feature )
    features  = torch.cat(features, dim=2).expand(batch, self.hidden, H, W)
    final_fea = torch.cat((x,features), dim=1)
    outputs   = self.last( final_fea )
    return outputs


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1. - drop_prob
    mask = x.new_zeros(x.size(0), 1, 1, 1)
    mask = mask.bernoulli_(keep_prob)
    x = torch.div(x, keep_prob)
    x.mul_(mask)
  return x


# Searching for A Robust Neural Architecture in Four GPU Hours
class GDAS_Reduction_Cell(nn.Module):

  def __init__(self, C_prev_prev, C_prev, C, reduction_prev, affine, track_running_stats):
    super(GDAS_Reduction_Cell, self).__init__()
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, 2, affine, track_running_stats)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, 1, affine, track_running_stats)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, 1, affine, track_running_stats)

    self.reduction = True
    self.ops1 = nn.ModuleList(
                  [nn.Sequential(
                      nn.ReLU(inplace=False),
                      nn.Conv2d(C, C, (1, 3), stride=(1, 2), padding=(0, 1), groups=8, bias=not affine),
                      nn.Conv2d(C, C, (3, 1), stride=(2, 1), padding=(1, 0), groups=8, bias=not affine),
                      nn.BatchNorm2d(C, affine=affine, track_running_stats=track_running_stats),
                      nn.ReLU(inplace=False),
                      nn.Conv2d(C, C, 1, stride=1, padding=0, bias=not affine),
                      nn.BatchNorm2d(C, affine=affine, track_running_stats=track_running_stats)),
                   nn.Sequential(
                      nn.ReLU(inplace=False),
                      nn.Conv2d(C, C, (1, 3), stride=(1, 2), padding=(0, 1), groups=8, bias=not affine),
                      nn.Conv2d(C, C, (3, 1), stride=(2, 1), padding=(1, 0), groups=8, bias=not affine),
                      nn.BatchNorm2d(C, affine=affine, track_running_stats=track_running_stats),
                      nn.ReLU(inplace=False),
                      nn.Conv2d(C, C, 1, stride=1, padding=0, bias=not affine),
                      nn.BatchNorm2d(C, affine=affine, track_running_stats=track_running_stats))])

    self.ops2 = nn.ModuleList(
                  [nn.Sequential(
                      nn.MaxPool2d(3, stride=2, padding=1),
                      nn.BatchNorm2d(C, affine=affine, track_running_stats=track_running_stats)),
                   nn.Sequential(
                      nn.MaxPool2d(3, stride=2, padding=1),
                      nn.BatchNorm2d(C, affine=affine, track_running_stats=track_running_stats))])

  @property
  def multiplier(self):
    return 4

  def forward(self, s0, s1, drop_prob = -1):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    X0 = self.ops1[0] (s0)
    X1 = self.ops1[1] (s1)
    if self.training and drop_prob > 0.:
      X0, X1 = drop_path(X0, drop_prob), drop_path(X1, drop_prob)

    #X2 = self.ops2[0] (X0+X1)
    X2 = self.ops2[0] (s0)
    X3 = self.ops2[1] (s1)
    if self.training and drop_prob > 0.:
      X2, X3 = drop_path(X2, drop_prob), drop_path(X3, drop_prob)
    return torch.cat([X0, X1, X2, X3], dim=1)


# To manage the useful classes in this file.
RAW_OP_CLASSES = {
  'gdas_reduction': GDAS_Reduction_Cell
}




from typing import List, Text, Any


class DynamicShapeTinyNet(nn.Module):

  def __init__(self, channels: List[int], genotype: Any, num_classes: int):
    super(DynamicShapeTinyNet, self).__init__()
    self._channels = channels
    if len(channels) % 3 != 2:
      raise ValueError('invalid number of layers : {:}'.format(len(channels)))
    self._num_stage = N = len(channels) // 3

    self.stem = nn.Sequential(
                    nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels[0]))

    # layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    c_prev = channels[0]
    self.cells = nn.ModuleList()
    for index, (c_curr, reduction) in enumerate(zip(channels, layer_reductions)):
      if reduction : cell = ResNetBasicblock(c_prev, c_curr, 2, True)
      else         : cell = InferCell(genotype, c_prev, c_curr, 1)
      self.cells.append( cell )
      c_prev = cell.out_dim
    self._num_layer = len(self.cells)

    self.lastact = nn.Sequential(nn.BatchNorm2d(c_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(c_prev, num_classes)

  def get_message(self) -> Text:
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_channels}, N={_num_stage}, L={_num_layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def forward(self, inputs):
    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      feature = cell(feature)

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits