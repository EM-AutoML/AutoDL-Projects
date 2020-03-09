##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import math, torch
import torch.nn as nn
from ..initialization import initialize_resnet
from ..SharedUtils    import additive_func
from .SoftSelect      import select2withP, ChannelWiseInter
from .SoftSelect      import linear_forward
from .SoftSelect      import get_width_choices as get_choices
from torch.nn.modules.utils import _pair as pair
import torch.nn.functional as F
from torch.nn import init


def get_gumbel_prob(xins, tau):
    while True:
        gumbels = -torch.empty_like(xins).exponential_().log()
        logits  = (xins.log_softmax(dim=1) + gumbels) / tau
        probs   = F.softmax(logits, dim=1)
        index   = probs.max(-1, keepdim=True)[1]
        one_h   = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        hardwts = one_h - probs.detach() + probs
        if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
            continue
        else: break
    return hardwts, index

def conv_forward(inputs, conv, choices):
  iC = conv.in_channels
  fill_size = list(inputs.size())
  fill_size[1] = iC - fill_size[1]
  filled  = torch.zeros(fill_size, device=inputs.device)
  xinputs = torch.cat((inputs, filled), dim=1)
  outputs = conv(xinputs)
  selecteds = [outputs[:,:oC] for oC in choices]
  return selecteds


class ConvBNReLU(nn.Module):
  num_conv  = 1
  def __init__(self, nIn, nOut, kernel, stride, padding, bias, has_avg, has_bn, has_relu, min_channel_ratio = 0.2, groups = 1):
    super(ConvBNReLU, self).__init__()
    self.InShape  = None
    self.OutShape = None
    self.kernel_size = pair(kernel)
    self.min_channel = int(nOut*min_channel_ratio)
    self.weights = nn.Parameter(torch.Tensor(nOut, nIn // groups, *self.kernel_size))
    self.register_parameter('layer_channel_attentions', nn.Parameter(1e-3*torch.randn(nOut-self.min_channel, 2)))
    if has_avg : self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    else       : self.avg = None
    self.has_bn = has_bn
    self.BN = nn.BatchNorm2d(nOut)
    # self.BN = nn.LayerNorm([nOut,100,100])
    if has_relu: self.relu = nn.ReLU(inplace=True)
    else       : self.relu = None
    self.in_dim   = nIn
    self.out_dim  = nOut
    self.search_mode = 'basic'
    self.stride = stride
    self.padding = padding
    self.dilation = 1
    self.groups = groups
    self.nOut = nOut
    self.nIn = nIn
    self.bias = False

    init.kaiming_normal(self.weights, mode='fan_in')


   
  def get_flops(self, channels, check_range=True, divide=1):
    iC, oC = channels
    if check_range: assert iC <= self.nIn and oC <= self.nOut, '{:} vs {:}  |  {:} vs {:}'.format(iC, self.nIn, oC, self.nOut)
    assert isinstance(self.InShape, tuple) and len(self.InShape) == 2, 'invalid in-shape : {:}'.format(self.InShape)
    assert isinstance(self.OutShape, tuple) and len(self.OutShape) == 2, 'invalid out-shape : {:}'.format(self.OutShape)
    conv_per_position_flops = (self.kernel_size[0] * self.kernel_size[1] * 1.0 / self.groups)
    all_positions = self.OutShape[0] * self.OutShape[1]
    flops = (conv_per_position_flops * all_positions / divide) * iC * oC
    if self.bias is not None: flops += all_positions / divide
    return flops

  def get_range(self):
    return [(self.min_channel, self.nOut)]

  def forward(self, inputs):
    if self.search_mode == 'basic':
      return self.basic_forward(inputs)
    elif self.search_mode == 'search':
      return self.search_forward(inputs)
    else:
      raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

  def search_forward(self, tuple_inputs):
    assert isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 3, 'invalid type input : {:}'.format( type(tuple_inputs) )
    inputs, expected_inC, tau = tuple_inputs

    #
    # import pdb;pdb.set_trace()
    # probs   = F.softmax(self.layer_channel_attentions, dim=1)
    # index   = probs.max(-1, keepdim=True)[1]
    # one_h   = torch.zeros_like(self.layer_channel_attentions).scatter_(-1, index, 1.0)

    self.gum_soft_C, _ = get_gumbel_prob(self.layer_channel_attentions, tau)
    # compute expected flop
    expected_outC = (self.min_channel + (F.softmax(self.layer_channel_attentions, dim=1)[:,1]).sum())
    expected_flop = self.get_flops([expected_inC, expected_outC], False, 1e6)
    if self.avg : out = self.avg( inputs )
    else        : out = inputs

    # convolutional layer
    weights = self.weights
    output = F.conv2d(out, weights, None, self.stride, self.padding, self.dilation, self.groups)
    out = self.BN(output)
    out =  torch.cat([torch.ones([self.min_channel,1]).cuda(), self.gum_soft_C[:,1].view(-1,1)], dim=0).view(1,self.nOut,  1, 1)*out
    if self.relu: out = self.relu( out )
    else        : out = out
    return out, expected_outC, expected_flop

  def basic_forward(self, inputs):
    if self.avg : out = self.avg( inputs )
    else        : out = inputs
    # conv = self.conv( out )
    probs   = F.softmax(self.layer_channel_attentions, dim=1)
    index   = probs.max(-1, keepdim=True)[1]
    one_h   = torch.zeros_like(self.layer_channel_attentions).scatter_(-1, index, 1.0)
    
    weights = self.weights
    conv = F.conv2d(out, weights, None, self.stride, self.padding, self.dilation, self.groups)


    if self.InShape is None:
      self.InShape  = (inputs.size(-2), inputs.size(-1))
      self.OutShape = (conv.size(-2)   , conv.size(-1))
      # self.BN = nn.LayerNorm(tuple([self.nOut,self.OutShape[0],self.OutShape[1]]))

    if self.has_bn: out= self.BN(conv)
    else          : out = conv
    out = torch.cat([torch.ones([self.min_channel,1]).to(one_h.device), one_h[:,1].view(-1,1)], dim=0).view(1, self.nOut, 1, 1) * out

    if self.relu: out = self.relu( out )
    else        : out = out
    # if self.InShape is None:
    #   self.InShape  = (inputs.size(-2), inputs.size(-1))
    #   self.OutShape = (out.size(-2)   , out.size(-1))
    #   self.BN.normalized_shape = tuple([self.nOut,self.OutShape[0],self.OutShape[1]])
    return out


class ResNetBasicblock(nn.Module):
  expansion = 1
  num_conv  = 2
  def __init__(self, inplanes, planes, stride):
    super(ResNetBasicblock, self).__init__()
    assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
    self.conv_a = ConvBNReLU(inplanes, planes, 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
    self.conv_b = ConvBNReLU(  planes, planes, 3,      1, 1, False, has_avg=False, has_bn=True, has_relu=False)
    if stride == 2:
      self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=True, has_bn=False, has_relu=False)
    elif inplanes != planes:
      self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=False,has_bn=True , has_relu=False)
    else:
      self.downsample = None
    self.out_dim     = planes
    self.search_mode = 'basic'

  def get_range(self):
    return self.conv_a.get_range() + self.conv_b.get_range()

  def get_flops(self, channels):
    assert len(channels) == 3, 'invalid channels : {:}'.format(channels)
    flop_A = self.conv_a.get_flops([channels[0], channels[1]])
    flop_B = self.conv_b.get_flops([channels[1], channels[2]])
    if hasattr(self.downsample, 'get_flops'):
      flop_C = self.downsample.get_flops([channels[0], channels[-1]])
    else:
      flop_C = 0
    if channels[0] != channels[-1] and self.downsample is None: # this short-cut will be added during the infer-train
      flop_C = channels[0] * channels[-1] * self.conv_b.OutShape[0] * self.conv_b.OutShape[1]
    return flop_A + flop_B + flop_C

  def forward(self, inputs):
    if self.search_mode == 'basic'   : return self.basic_forward(inputs)
    elif self.search_mode == 'search': return self.search_forward(inputs)
    else: raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

  def search_forward(self, tuple_inputs):
    assert isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 3, 'invalid type input : {:}'.format( type(tuple_inputs) )
    inputs, expected_inC, tau = tuple_inputs
    # assert indexes.size(0) == 2 and probs.size(0) == 2 and probability.size(0) == 2
    out_a, expected_inC_a, expected_flop_a = self.conv_a( (inputs, expected_inC, tau) )
    out_b, expected_inC_b, expected_flop_b = self.conv_b( (out_a , expected_inC_a, tau) )
    if self.downsample is not None:
      residual, _, expected_flop_c = self.downsample( (inputs, expected_inC, tau) )
    else:
      residual, expected_flop_c = inputs, 0
    out = additive_func(residual, out_b)
    out = nn.functional.relu(out, inplace=True)
    return out, expected_inC_b, sum([expected_flop_a, expected_flop_b, expected_flop_c])

  def basic_forward(self, inputs):
    basicblock = self.conv_a(inputs)
    basicblock = self.conv_b(basicblock)
    if self.downsample is not None: residual = self.downsample(inputs)
    else                          : residual = inputs
    out = additive_func(residual, basicblock)
    # return out
    return nn.functional.relu(out, inplace=True)



class ResNetBottleneck(nn.Module):
  expansion = 4
  num_conv  = 3
  def __init__(self, inplanes, planes, stride):
    super(ResNetBottleneck, self).__init__()
    assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
    self.conv_1x1 = ConvBNReLU(inplanes, planes, 1,      1, 0, False, has_avg=False, has_bn=True, has_relu=True)
    self.conv_3x3 = ConvBNReLU(  planes, planes, 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
    self.conv_1x4 = ConvBNReLU(planes, planes*self.expansion, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
    if stride == 2:
      self.downsample = ConvBNReLU(inplanes, planes*self.expansion, 1, 1, 0, False, has_avg=True, has_bn=False, has_relu=False)
    elif inplanes != planes*self.expansion:
      self.downsample = ConvBNReLU(inplanes, planes*self.expansion, 1, 1, 0, False, has_avg=False,has_bn=True , has_relu=False)
    else:
      self.downsample = None
    self.out_dim     = planes * self.expansion
    self.search_mode = 'basic'

  def get_range(self):
    return self.conv_1x1.get_range() + self.conv_3x3.get_range() + self.conv_1x4.get_range()

  def get_flops(self, channels):
    assert len(channels) == 4, 'invalid channels : {:}'.format(channels)
    flop_A = self.conv_1x1.get_flops([channels[0], channels[1]])
    flop_B = self.conv_3x3.get_flops([channels[1], channels[2]])
    flop_C = self.conv_1x4.get_flops([channels[2], channels[3]])
    if hasattr(self.downsample, 'get_flops'):
      flop_D = self.downsample.get_flops([channels[0], channels[-1]])
    else:
      flop_D = 0
    if channels[0] != channels[-1] and self.downsample is None: # this short-cut will be added during the infer-train
      flop_D = channels[0] * channels[-1] * self.conv_1x4.OutShape[0] * self.conv_1x4.OutShape[1]
    return flop_A + flop_B + flop_C + flop_D

  def forward(self, inputs):
    if self.search_mode == 'basic'   : return self.basic_forward(inputs)
    elif self.search_mode == 'search': return self.search_forward(inputs)
    else: raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

  def basic_forward(self, inputs):
    bottleneck = self.conv_1x1(inputs)
    bottleneck = self.conv_3x3(bottleneck)
    bottleneck = self.conv_1x4(bottleneck)
    if self.downsample is not None: residual = self.downsample(inputs)
    else                          : residual = inputs
    out = additive_func(residual, bottleneck)
    return nn.functional.relu(out, inplace=True)

  def search_forward(self, tuple_inputs):
    assert isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 5, 'invalid type input : {:}'.format( type(tuple_inputs) )
    inputs, expected_inC, probability, indexes, probs = tuple_inputs
    assert indexes.size(0) == 3 and probs.size(0) == 3 and probability.size(0) == 3
    out_1x1, expected_inC_1x1, expected_flop_1x1 = self.conv_1x1( (inputs, expected_inC    , probability[0], indexes[0], probs[0]) )
    out_3x3, expected_inC_3x3, expected_flop_3x3 = self.conv_3x3( (out_1x1,expected_inC_1x1, probability[1], indexes[1], probs[1]) )
    out_1x4, expected_inC_1x4, expected_flop_1x4 = self.conv_1x4( (out_3x3,expected_inC_3x3, probability[2], indexes[2], probs[2]) )
    if self.downsample is not None:
      residual, _, expected_flop_c = self.downsample( (inputs, expected_inC  , probability[2], indexes[2], probs[2]) )
    else:
      residual, expected_flop_c = inputs, 0
    out = additive_func(residual, out_1x4)
    out = nn.functional.relu(out,inplace=True)
    return out, expected_inC_1x4, sum([expected_flop_1x1, expected_flop_3x3, expected_flop_1x4, expected_flop_c])



class SearchWidthCifarResNet(nn.Module):

  def __init__(self, block_name, depth, num_classes):
    super(SearchWidthCifarResNet, self).__init__()

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    if block_name == 'ResNetBasicblock':
      block = ResNetBasicblock
      assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
      layer_blocks = (depth - 2) // 6
    elif block_name == 'ResNetBottleneck':
      block = ResNetBottleneck
      assert (depth - 2) % 9 == 0, 'depth should be one of 164'
      layer_blocks = (depth - 2) // 9
    else:
      raise ValueError('invalid block : {:}'.format(block_name))

    self.message     = 'SearchWidthCifarResNet : Depth : {:} , Layers for each block : {:}'.format(depth, layer_blocks)
    self.num_classes = num_classes
    self.channels    = [16]
    self.layers      = nn.ModuleList( [ ConvBNReLU(3, 16, 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=True) ] )
    self.InShape     = None
    for stage in range(3):
      for iL in range(layer_blocks):
        iC     = self.channels[-1]
        planes = 16 * (2**stage)
        stride = 2 if stage > 0 and iL == 0 else 1
        module = block(iC, planes, stride)
        self.channels.append( module.out_dim )
        self.layers.append  ( module )
        self.message += "\nstage={:}, ilayer={:02d}/{:02d}, block={:03d}, iC={:3d}, oC={:3d}, stride={:}".format(stage, iL, layer_blocks, len(self.layers)-1, iC, module.out_dim, stride)
  
    self.avgpool     = nn.AvgPool2d(8)
    self.classifier  = nn.Linear(module.out_dim, num_classes)
    self.InShape     = None
    self.tau         = -1
    self.search_mode = 'basic'
    #assert sum(x.num_conv for x in self.layers) + 1 == depth, 'invalid depth check {:} vs {:}'.format(sum(x.num_conv for x in self.layers)+1, depth)
    
    # parameters for width/chanel
    self.Ranges = []
    self.layer2indexRange = []
    for i, layer in enumerate(self.layers):
      start_index = len(self.Ranges)
      self.Ranges += layer.get_range()
      self.layer2indexRange.append( (start_index, len(self.Ranges)) )
    assert len(self.Ranges) + 1 == depth, 'invalid depth check {:} vs {:}'.format(len(self.Ranges) + 1, depth)

    self.apply(initialize_resnet)

 
    

  def base_parameters(self):
    return list(self.layers.parameters()) + list(self.avgpool.parameters()) + list(self.classifier.parameters())

  def get_flop(self, mode, config_dict, extra_info):
    if config_dict is not None: config_dict = config_dict.copy()
    #weights = [F.softmax(x, dim=0) for x in self.width_attentions]
    channels = [3]
    for i, block in enumerate(self.layers):
          if isinstance(block, ConvBNReLU):
            temp = block.layer_channel_attentions
            channels.append((temp[:,0]<temp[:,1]).sum().item()+block.min_channel)
          elif isinstance(block, ResNetBasicblock):
            conv_a = block.conv_a.layer_channel_attentions
            conv_b = block.conv_b.layer_channel_attentions
            channels.append((conv_a[:,0]<conv_a[:,1]).sum().item()+block.conv_a.min_channel)
            channels.append((conv_b[:,0]<conv_b[:,1]).sum().item()+block.conv_b.min_channel)

          
    
    flop = 0
    for i, layer in enumerate(self.layers):
      s, e = self.layer2indexRange[i]
      xchl = tuple( channels[s:e+1] )
      flop+= layer.get_flops(xchl)
    # the last fc layer
    flop += channels[-1] * self.classifier.out_features
    if config_dict is None:
      return flop / 1e6
    else:
      config_dict['xchannels']  = channels
      config_dict['super_type'] = 'infer-width'
      config_dict['estimated_FLOP'] = flop/ 1e6
      return flop / 1e6, config_dict

  def get_arch_info(self):



    def com(block):
      probs   = F.softmax(block.layer_channel_attentions, dim=1)
      index   = probs.max(-1, keepdim=True)[1]
      one_h   = torch.zeros_like(block.layer_channel_attentions).scatter_(-1, index, 1.0)
      return block.min_channel + (one_h[:,1]).sum()
    channels_ori, channels_after_prune = [3], [3]
    for i, block in enumerate(self.layers):
          if isinstance(block, ConvBNReLU):
            channels_after_prune.append(com(block).item())
            channels_ori.append(block.nOut)
          elif isinstance(block, ResNetBasicblock):
            channels_after_prune.append(com(block.conv_a).item())
            channels_after_prune.append(com(block.conv_b).item())

            channels_ori.append(block.conv_a.nOut)
            channels_ori.append(block.conv_b.nOut)
        
    return channels_ori, channels_after_prune

  def set_tau(self, tau_max, tau_min, epoch_ratio):
    assert epoch_ratio >= 0 and epoch_ratio <= 1, 'invalid epoch-ratio : {:}'.format(epoch_ratio)
    tau = tau_min + (tau_max-tau_min) * (1 + math.cos(math.pi * epoch_ratio)) / 2
    self.tau = tau

  def get_message(self):
    return self.message

  def forward(self, inputs):
    if self.search_mode == 'basic':
      return self.basic_forward(inputs)
    elif self.search_mode == 'search':
      return self.search_forward(inputs, self.tau)
    else:
      raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

  def search_forward(self, inputs, tau):
    x = inputs
    last_channel_idx = 0
    expected_inC = 3
    flops = []
    for i, layer in enumerate(self.layers):
      x, expected_inC, expected_flop = layer( (x, expected_inC, tau) )
      last_channel_idx += layer.num_conv
      flops.append( expected_flop )
    flops.append( expected_inC * (self.classifier.out_features*1.0/1e6) )
    features = self.avgpool(x)
    features = features.view(features.size(0), -1)
    logits   = linear_forward(features, self.classifier)
    return logits, torch.stack( [sum(flops)] )

  def basic_forward(self, inputs):
    if self.InShape is None: self.InShape = (inputs.size(-2), inputs.size(-1))
    x = inputs
    for i, layer in enumerate(self.layers):
      x = layer( x )
    features = self.avgpool(x)
    features = features.view(features.size(0), -1)
    # logits   = self.classifier(features)
    logits   = linear_forward(features, self.classifier)
    return features, logits
