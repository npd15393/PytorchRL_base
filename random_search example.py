import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import random 
def gen_normal(mu=0, sig=70):
    k=[]
    v=[]
    for i in range(200):
        x=i-100
        k.append(x)
        v.append(10*math.exp(-((x-mu)/sig)**2)/2)  
    return k,v      

# generate normal distro data
data_in,data_lbl=gen_normal()
np_k=np.array(data_in).reshape([len(data_in),1])
np_v=np.array(data_lbl).reshape([len(data_in),1])

def sample_hp():
    # sample hyperparameters and model params
    hp_lr=0.0001+0.0009*random.random() #[0.0001,0.0005,0.001,0.005]
    hp_batch_sz=np.random.randint(16,64)
    hp_hs=np.random.randint(50,150)
    hp_act=(torch.nn.Tanh(),torch.nn.Tanh())#[(torch.nn.Tanh(),torch.nn.Tanh()),(torch.nn.Tanh(),torch.nn.ReLU())]
    return [hp_lr,hp_batch_sz,hp_hs,hp_act]

param_set=[]
test_loss=100
opt_hp=None
for _ in range(25):
    param_set.append(sample_hp())

# gridsearch
for hp in param_set:
    lr=hp[0]
    batch_sz=hp[1]
    acts=hp[3]
    hidl=hp[2]

    # define model
    model=torch.nn.Sequential(torch.nn.Linear(1,hidl), acts[0],torch.nn.Linear(hidl,hidl//2),acts[1],torch.nn.Linear(hidl//2,1))
    model=model.float()

    minibatch_size=batch_sz
    opt=torch.optim.RMSprop(model.parameters(), lr=lr)
    loss=torch.nn.MSELoss(reduction='sum')

    # train
    for ep in range(10000):
        idx=np.random.randint(0,len(data_in),minibatch_size)
        ep_loss=loss(model(torch.tensor(np_k[idx],dtype=torch.float)),torch.tensor(list(np_v[idx]),dtype=torch.float))
        # if ep%50==0:
        #     print(ep_loss)

        opt.zero_grad()
        ep_loss.backward()
        opt.step()

    # x_ax=list(range(-100,100))
    # y1=data_lbl
    # y2=model(torch.tensor(np_k,dtype=torch.float)).tolist()

    if test_loss>loss(model(torch.tensor(np_k,dtype=torch.float)),torch.tensor(list(np_v),dtype=torch.float)).tolist():
        test_loss=loss(model(torch.tensor(np_k,dtype=torch.float)),torch.tensor(list(np_v),dtype=torch.float)).tolist()
        opt_hp=hp
    # plt.plot(x_ax,y1,x_ax,y2)
    # plt.show()
print(test_loss)
print(opt_hp)

x_ax=list(range(-100,100))
y1=data_lbl
y2=model(torch.tensor(np_k,dtype=torch.float)).tolist()
plt.plot(x_ax,y1,x_ax,y2)
plt.show()