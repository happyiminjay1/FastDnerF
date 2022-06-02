import os
import imageio
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from positional_encodings import PositionalEncoding3D


from run_dnerf_helpers import *
import random

from load_blender import load_blender_data

try:
    from apex import amp
except ImportError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)



class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet,self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, inputs):
        x = inputs.view(inputs.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.tanh(self.fc6(x))
        return x

class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet,self).__init__()

        self.fc1 = nn.Linear(33, 16)
        self.fc2 = nn.Linear(16, 3)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, inputs):

        #inputs with time
        x = inputs.view(inputs.size(0),-1)

        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))

        return x

# positional encoding library

#pip install positional-encodings

# model load

encoder = EncoderNet().to(device)
decoder = DecoderNet().to(device)

encoder, decoder = torch.load('/home/minjay/FastDynamicNeRF/MLPDeform/encoder_decoder40.pkl')

#encoder ( input : batch_size, 10 -> output : batch_size , 32 )
#deocder ( input : batch_size, 32 + 1(time: 0~1) -> output : batch_size , 3)

#positional embeding
p_enc_3d = PositionalEncoding3D(10)
z = torch.ones((1,256,256,256,10))
z = p_enc_3d(z) 

# if sampling point coordinate is (x_idx, y_idx, z_idx)
# dim_10 = z[0,x_idx, y_idx, z_idx:10]

#network flow
outputs = encoder(dim_10) # dim_10 - positional encoded latentm, shape : batchsize, 10
t_seq = torch.Tensor([ 1/150 for _ in range(outputs.shape[0]) ]).to(device) 
t_seq = torch.unsqueeze(t_seq, 1) # t_seq - time feature adding , shape : batchsize, 1
outputs = torch.cat( (outputs,t_seq),1) # output - shape : batchsize, 33
outputs = decoder(outputs) #output - shape : batchszie, 3



