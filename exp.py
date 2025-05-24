import os
import torch
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.join(os.path.dirname('__file__'), '../'))

from engine.solver import Trainer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from engine.logger import Logger
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one


class get_dataset(Dataset):
    def __init__(self, data, seq_length,mode,train_split=0.8):
        super(get_dataset, self).__init__()

        self.samples = data
        
        self.seq_length=seq_length
        self.pred_length = 0
        self.features = data.shape[-1]
        
        self.data = self.get_data(data)
        train_num = int(train_split * len(self.data))
        if mode == 'train':
            self.data = self.data[:train_num, :, :]
        else:
            self.data = self.data[train_num:, :, :]
        
    def __getitem__(self, index):
        return self.data[index, :, :]
                

    def __len__(self):
        return len(self.data)
    
    def get_data(self,data):

        # data_max = np.max(data, axis=0)
        # data_min = np.min(data, axis=0)
 
        # data = (data - data_min) / (data_max - data_min)
        num_sample = len(data) - self.seq_length - self.pred_length + 1
        seq_data = torch.zeros(num_sample,
                               self.seq_length + self.pred_length,
                               self.features)
 
        #         print(data.iloc[0:0 + self.seq_length + 1, self.features].values)
 
        for i in range(num_sample):
            seq_data[i] = torch.tensor(data[i:i + self.seq_length + self.pred_length,
                                       :])
        #         print(data_max)
        #         print(data_min)
 
        return seq_data

p=20
t=1000
f=40
data_name=f'lorenz_p{p}_t{t}_f{f}'
data= np.load(f'/data/0shared/liubo/diffusion-gc/DiffuGC/my_exp/{data_name}/data.npy')

print(f'p:{p}, t:{t}, f:{f}')  

train_split=0.8
batch_size=64

train = data[:int(train_split*data.shape[0]), :]
test = data[int(train_split*data.shape[0]):,:].reshape(1, -1, data.shape[-1])


class Args_Example:
    def __init__(self) -> None:
        self.name=data_name
        self.config_path = '/data/0shared/liubo/diffusion-gc/DiffuGC/Config/'+data_name+'.yaml'
        self.save_dir = '/data/0shared/liubo/diffusion-gc/DiffuGC/my_exp/'+data_name
        self.gpu = 6
        os.makedirs(self.save_dir, exist_ok=True)

args =  Args_Example()
configs = load_yaml_config(args.config_path)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
train_dataset = get_dataset(train, seq_length=configs['model']['params']['seq_length'], mode='train', train_split=train_split)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

log_name= 'seq_length'+str(configs['model']['params']['seq_length'])+'_batch_size'+str(batch_size)+'_lr'+str(configs['solver']['base_lr'])+'_d_model'+str(configs['model']['params']['d_model'])+'_num_layers'+str(configs['model']['params']['n_layer_enc'])+'_num_heads'+str(configs['model']['params']['n_heads'])+'_dropout'+str(configs['model']['params']['attn_pd'])
feature_size=data.shape[1]
ts_target=0
print(f"ts_target: {ts_target}")
# for seq_length in [24,48,96]:    

    # print(f'-------{seq_length}-------')
    # configs['model']['params']['seq_length'] = seq_length
    # for d_model in [256,512]:
    #     configs['model']['params']['d_model'] = d_model
    #     print(f'-------{d_model}-------')
gc_logger = Logger(args=args,name=log_name,ts_target=ts_target)
model = instantiate_from_config(configs['model']).to(device)
trainer = Trainer(config=configs, args=args, model=model, dataloader=trainloader,weight_decay=configs['solver']['weight_decay'],logger=gc_logger)
trainer.train(ts_target)