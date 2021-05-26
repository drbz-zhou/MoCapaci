import torch
from torch import nn

def init_weights(m):
	if type(m) == nn.LSTM:
		for name, param in m.named_parameters():
			if 'weight_ih' in name:
				torch.nn.init.orthogonal_(param.data)
			elif 'weight_hh' in name:
				torch.nn.init.orthogonal_(param.data)
			elif 'bias' in name:
				param.data.fill_(0)
	elif type(m) == nn.Conv1d or type(m) == nn.Linear:
		torch.nn.init.kaiming_uniform_(m.weight)
		m.bias.data.fill_(0)

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 