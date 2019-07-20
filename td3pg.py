import torch 
import gym
import copy
import random as rand
import numpy as np
import numpy.random as nr
import itertools

dev=torch.device("cuda")
dfloat= torch.cuda.FloatTensor
env=gym.make("MountainCarContinuous-v0")
BATCH_SIZE=32
MEM_SIZE=100000
EPOCHS = 500000
GAMMA=0.95
TTAU=0.001
############################ Replay memory class #################################
class ReplayMemory:
    def __init__(self,n):
        self.size=n
        self.expBuffer=[]

    # Circular memory
    def push(self,exp):
        if len(self.expBuffer)<self.size:
            self.expBuffer=self.expBuffer+[exp]
        else:
            self.expBuffer.pop(0)
            self.expBuffer=self.expBuffer+[exp]

    # Check if buffer has sufficient experience
    def isReady(self):
        return len(self.expBuffer)>=BATCH_SIZE

    def sampleBatch(self,sz=None):
        if sz==None:
            sz=BATCH_SIZE
        idxs=[np.random.randint(0,len(self.expBuffer)) for _ in range(sz)]
        return [self.expBuffer[idx] for idx in idxs]

############################# OU noise ###########################

# class OUNoise:
#     def __init__(self,action_dimension,mu=None, theta=0.15, sigma=0.3):
#         self.action_dimension = action_dimension
#         self.mu = np.zeros(action_dimension) if mu==None else mu
#         self.sigma = sigma
#         self.state = np.ones(self.action_dimension) * self.mu
#         self.reset()

#     def reset(self):
#         self.state = np.ones(self.action_dimension) * self.mu

#     def noise(self):
#         x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
#         self.state = x + dx
#         return self.state

class GaussianNoise:
    def __init__(self,action_dimension,mu=None, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = np.zeros(action_dimension) if mu==None else mu
        self.sigma = sigma

    def noise(self):
        dx = self.mu + self.sigma * nr.randn(self.action_dimension)
        return dx

class UniformNoise:
    def __init__(self,action_dimension):
        self.action_dimension = action_dimension

    def noise(self):
        dx = 2*nr.rand(self.action_dimension)-1
        return dx
######################## Model Definition #####################

class Critic(torch.nn.Module):
    def __init__(self,n_f,n_a):
        super(Critic, self).__init__()
        self.st_fc=torch.nn.Sequential(torch.nn.Linear(n_f,64),torch.nn.ReLU())
        self.model = torch.nn.Sequential(torch.nn.Linear(64+n_a,32), \
            torch.nn.ReLU(),torch.nn.Linear(32,64), \
            torch.nn.ReLU(),torch.nn.Linear(64,1))

    def forward(self,xs):
        [x,a] = xs
        # state = torch.from_numpy(x).float()
        cat_dim=len(x.shape)-1
        branches=[a,self.st_fc(x)] #if targ else [self.get_action(st,targ),self.st_fc_c1(state)] 
        return self.model(torch.cat(branches,cat_dim))

    def grads(self,):
        print([p.grad.data.numpy() for p in list(self.model.parameters())])

class Actor(torch.nn.Module):
    def __init__(self,n_f,n_a):
        super(Actor, self).__init__()
        self.model=torch.nn.Sequential(torch.nn.Linear(n_f,64), \
            torch.nn.ReLU(),torch.nn.Linear(64,32), \
            torch.nn.ReLU(),torch.nn.Linear(32,64),torch.nn.ReLU(),torch.nn.Linear(64,n_a),torch.nn.Tanh())
            
    def forward(self,x):
        # if 
        # state = torch.from_numpy(x).float()
        cat_dim=len(x.shape)-1
        return self.model(x)

    def grads(self,):
        print(list(self.model.parameters()))
        print([p.grad.data.numpy() for p in list(self.model.parameters())])


###################### Agent Utils #################
class ModelFactory:
    def __init__(self,env):
        self.buff=ReplayMemory(MEM_SIZE)
        self.n_features= env.low_state.shape[0]
        self.n_actions=1

        self.alr=1e-4
        self.clr=1e-3
        self.noise_model=GaussianNoise(self.n_actions)
        self.uniform=UniformNoise(self.n_actions)
        self.epoch=0
        # define models
        # self.st_fc_c1=torch.nn.Sequential(torch.nn.Linear(self.n_features,128))
        # self.st_fc_tc1=torch.nn.Sequential(torch.nn.Linear(self.n_features,128))
        # self.st_fc_c2=torch.nn.Sequential(torch.nn.Linear(self.n_features,128))
        # self.st_fc_tc2=torch.nn.Sequential(torch.nn.Linear(self.n_features,128))
        
        self.critic1=Critic(self.n_features,self.n_actions) #torch.nn.Sequential(torch.nn.Linear(128+self.n_actions,64), \torch.nn.ReLU(),torch.nn.Linear(64,1))
        
        self.tcritic1=Critic(self.n_features,self.n_actions)#copy.deepcopy(self.critic1)

        self.critic2=Critic(self.n_features,self.n_actions)#torch.nn.Sequential(torch.nn.Linear(128+self.n_actions,64), \torch.nn.ReLU(),torch.nn.Linear(64,1))
        
        self.tcritic2=Critic(self.n_features,self.n_actions)#copy.deepcopy(self.critic2)

        self.actor=Actor(self.n_features,self.n_actions)#torch.nn.Sequential(torch.nn.Linear(self.n_features,128), \torch.nn.Tanh(),torch.nn.Linear(128,self.n_actions),torch.nn.Tanh())
        
        self.tactor=Actor(self.n_features,self.n_actions)#copy.deepcopy(self.actor)

        self.actor_opt=torch.optim.Adam(self.actor.parameters(), lr=self.alr)
        # self.critic1_opt=torch.optim.Adam(itertools.chain(self.st_fc_c1.parameters(),self.critic1.parameters()), lr=self.clr)
        # self.critic2_opt=torch.optim.Adam(itertools.chain(self.st_fc_c2.parameters(),self.critic2.parameters()), lr=self.clr)
        self.critic1_opt=torch.optim.Adam(self.critic1.parameters(), lr=self.clr)
        self.critic2_opt=torch.optim.Adam(self.critic2.parameters(), lr=self.clr)
        self.tcs=[self.tcritic1,self.tcritic2]
        self.cs=[self.critic1,self.critic2]
        # self.fcs=[self.st_fc_c1,self.st_fc_c2]
        # self.ftcs=[self.st_fc_tc1,self.st_fc_tc2]

        for param in self.tcritic1.parameters():
            param.requires_grad = False 

        for param in self.tcritic2.parameters():
            param.requires_grad = False    

        for param in self.tactor.parameters():
            param.requires_grad = False     

        self.hard_update(self.critic1,self.tcritic1)
        self.hard_update(self.critic2,self.tcritic2)
        self.hard_update(self.actor,self.tactor)

    def get_action_random(self,):
        return torch.tensor(self.uniform.noise(),dtype=torch.float) 

    def get_action(self,st,targ=False):
        state = torch.from_numpy(st).float()
        v=self.tactor(state) if targ else self.actor(state)
        return v.detach()

    def get_noisy_action(self,st,targ=False):
        state = torch.from_numpy(st).float()
        noise=torch.tensor(self.noise_model.noise(),dtype=torch.float)
        # print('noise: '+str(noise))
        act=self.tactor(state) if targ else self.actor(state)
        clipped_act=torch.clamp(act.add(noise),min=-1.0,max=1.0)
        return clipped_act.detach()

    def get_q1(self, st, targ=False, a=None):
        state = torch.from_numpy(st).float()
        cat_dim=len(state.size())-1
        if a is None:
            a=self.get_action(st,targ)
        # else:
        #     a=torch.from_numpy(a).float()
        # branches=[self.get_action(st,targ),self.st_fc_tc1(state)] if targ else [self.get_action(st,targ),self.st_fc_c1(state)] 
        # return self.tcritic1(torch.cat(branches,cat_dim)) if targ else self.critic1(torch.cat(branches,cat_dim))
        return self.tcritic1([state,a]) if targ else self.critic1([state,a])

    def get_q2(self, st, targ=False, a=None):
        state = torch.from_numpy(st).float()
        cat_dim=len(state.size())-1
        if a is None:
            a=self.get_action(st,targ)
        # else:
        #     a=torch.from_numpy(a).float()
        # branches=[self.get_action(st,targ),self.st_fc_tc1(state)] if targ else [self.get_action(st,targ),self.st_fc_c1(state)] 
        # return self.tcritic1(torch.cat(branches,cat_dim)) if targ else self.critic1(torch.cat(branches,cat_dim))
        return self.tcritic2([state,a]) if targ else self.critic2([state,a])

    def get_q_ue(self, st, targ=False):
        state = torch.from_numpy(st).float()
        cat_dim=len(state.size())-1
        # if targ:
            # branches1=[self.get_action(st,targ),self.st_fc_tc1(state)]
            # branches2=[self.get_action(st,targ),self.st_fc_tc2(state)]
            # a=self.tcritic1(torch.cat(branches1,cat_dim))
            # b=self.tcritic2(torch.cat(branches2,cat_dim))
        a=self.tcritic1([state,self.get_noisy_action(st,targ)])
        b=self.tcritic2([state,self.get_noisy_action(st,targ)])
        # else:
        #     branches1=[self.get_action(st,targ),self.st_fc_c1(state)]
        #     branches2=[self.get_action(st,targ),self.st_fc_c2(state)]
        #     a= self.critic1(torch.cat(branches1,cat_dim))
        #     b= self.critic2(torch.cat(branches2,cat_dim))

        t=torch.cat([a,b],cat_dim)
        ue=torch.min(t,cat_dim)[0]
        return ue

    # def update_tcritic(self,id=0):
    #     primary_weights = list(self.cs[id].parameters())
    #     psd=list(self.fcs[id].parameters())
    #     secondary_weights = list(self.tcs[id].parameters())
    #     ssd=list(self.ftcs[id].parameters())
        
    #     n = len(primary_weights)
    #     for i in range(n):
    #         secondary_weights[i].data[:] = primary_weights[i].data[:]

    #     n = len(psd)
    #     for i in range(n):
    #         ssd[i].data[:] = psd[i].data[:]

    #     self.tcs[id].load_state_dict(self.cs[id].state_dict())
    #     self.ftcs[id].load_state_dict(self.fcs[id].state_dict())

    # def update_tactor(self):
    #     primary_weights = list(self.actor.parameters())
    #     secondary_weights = list(self.tactor.parameters())
    #     n = len(primary_weights)
    #     for i in range(n):
    #         secondary_weights[i].data[:] = primary_weights[i].data[:]

    #     self.tactor.load_state_dict(self.actor.state_dict())

    def hard_update(self,model,tmodel):
        for target_param, param in zip(tmodel.parameters(), model.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self,model,tmodel):
        for target_param, param in zip(tmodel.parameters(), model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TTAU) + param.data * 
            TTAU)        

############################# TD3PG  Agent ############################ 
class TD3PGAgent:
    def __init__(self):
        self.buff=ReplayMemory(MEM_SIZE)
        self.n_features= env.low_state.shape[0]
        self.n_actions=1
        self.td3 = ModelFactory(env)
        s=env.reset()

        # self.critic_loss=torch.nn.MSELoss(reduction='sum')

    def update(self):
        """
        Update Q function and policy
        :param exp: Experience tuple from Env
        :return: void
        """
        max_del=0
        exp=self.buff.sampleBatch()
        q_s=[]
        q_targ=[]
        ss=[]
        acts=[]
        ls1=0
        ls2=0
        for i in range(len(exp)):
            s=exp[i][0]
            a=exp[i][1] #action to apply to learning critic
            s_=exp[i][2]
            r=exp[i][3]
            done=exp[i][4]

            # a_=self.td3.get_noisy_action(s_,True) # noisy action for next state

            # q update
            ss.append(s)
            acts.append(a.numpy())
            
            q_n=self.td3.get_q_ue(s_,True)
            q_targ=r+GAMMA*q_n if not done else r
            # print('prev q '+str((s,a))+' : '+str(self.get_Q(s,a,self.Q)))
            q_s1=self.td3.get_q1(s,False,a)
            q_s2=self.td3.get_q2(s,False,a)

            # add to losses
            ls1+=0.5*torch.pow(q_targ-q_s1,2)
            ls2+=0.5*torch.pow(q_targ-q_s2,2)
            # q_s[-1][a]=q_target#q_s[-1][a]+self.alpha(0)*(q_target-q_s[-1][a])

            # print('s:'+str(s)+' a:'+str(a))
            # print(alpha(ep)*(q_target))
            # max_del=max_del if max_del>abs(q_target-q_s[-1][a]) else abs(q_target-q_s[-1][a])
            # print('new q '+str((s,a))+' : '+str(self.get_Q(s,a,self.Q)))
            # keys=[(s,act) for act in self.actions]
            # tot_q=sum(np.exp(list(get_Q(s).values())))
            # for k in keys:
            # pi_s=get_pi(s)
            # pi_s[option_idxs(a)]= np.exp(get_Q(s,a))/tot_q #math.exp(self.get_Q(k[0],k[1]))/tot_q


        # define losses
        # print(len(q_targ))
        # print(len(q_s))
        ss=np.array(ss)
        acts=np.array(acts)
        # q_targ=np.array(q_targ)
        # q_s=np.array(q_s)

        # ls=0.5*torch.sum(torch.pow(torch.from_numpy(q_targ-q_s).float(),2))
        q1=self.td3.get_q1(ss,False,torch.from_numpy(acts).float())
        q2=self.td3.get_q2(ss,False,torch.from_numpy(acts).float())
        q=self.td3.get_q1(ss)
        als=-q.mean()
        # print('als:'+str(als))

        torch.nn.utils.clip_grad_norm(self.td3.critic1.parameters(), 1)
        torch.nn.utils.clip_grad_norm(self.td3.critic2.parameters(), 1)
        torch.nn.utils.clip_grad_norm(self.td3.actor.parameters(), 1)
        
        # actor update
        self.td3.actor_opt.zero_grad()
        als.backward()
        # self.td3.actor.grads()
        self.td3.actor_opt.step()

        # update target actor
        # nr=rand.random()
        # if nr<0.1: 
        self.td3.soft_update(self.td3.actor,self.td3.tactor)

        # critic update
        self.td3.critic1_opt.zero_grad()
        # print('ls1: '+str(ls1))
        ls1.backward()
        # self.td3.critic1.grads()
        self.td3.critic1_opt.step()
        
        self.td3.critic2_opt.zero_grad()
        # print('ls2: '+str(ls2))
        ls2.backward()
        # self.td3.critic2.grads()
        self.td3.critic2_opt.step()
        # print('#########')
        # update target critics
        # nr=rand.random()
        # if nr<0.1: 
        self.td3.soft_update(self.td3.critic1,self.td3.tcritic1)
        self.td3.soft_update(self.td3.critic2,self.td3.tcritic2)

        
        return max_del

    def train(self):
        current_state=env.reset()
        epi_rwd=0
        step_cnt=0
        BPerr=[]

        for epi in range(50000):
            act=self.td3.get_action_random()

            # print(act)
            exp=env.step(act.numpy()) # take action
            # env.render()
            exp=(current_state,act)+exp[0:-1]
            # self.epsilon=self.epsilon*0.995
            self.buff.push(exp)         

            if exp[-1]:
                current_state=env.reset()
                step_cnt=0
                epi_rwd=0
            else:
                current_state=exp[2]
                step_cnt=step_cnt+1

        current_state=env.reset()
        epi_rwd=0
        step_cnt=0

        for epi in range(EPOCHS):
            act=self.td3.get_noisy_action(current_state,True)

            # print(act)
            exp=env.step(act.numpy()) # take action
            # env.render()
            exp=(current_state,act)+exp[0:-1]
            # self.epsilon=self.epsilon*0.995
            self.buff.push(exp)
            self.td3.epoch=self.td3.epoch+1
            # check if transition valid
            if not exp[-1]:
                epi_rwd+=exp[3]
                current_state=exp[2]
                step_cnt=step_cnt+1
            else:
                current_state=env.reset()
                print('===========Epi length:'+str(step_cnt)+' Total rwd = '+str(epi_rwd)+'==============')
                step_cnt=0
                epi_rwd=0

            # q updates
            if self.buff.isReady():
                err=self.update()
                # print('Max Bellman Projection Error: '+str(err))
                # BPerr.append(err)

            if epi%5000==0 and epi>0:
                print('testing actor for '+str(epi))
                self.test(1)
                current_state=env.reset()
            # self.epsilon=self.epsilon*0.99

    def test(self,n=15):
        # current_state=env.reset()
        # epi_rwd=0
        # step_cnt=0
        BPerr=[]
        for _ in range(n):
            current_state=env.reset()
            epi_rwd=0
            step_cnt=0
            for _ in range(EPOCHS):
                act=self.td3.get_action(current_state)
                # print(act)
                exp=env.step(act.detach().numpy()) # take action

                exp=exp[0:-1]
                env.render()
                # check if transition valid
                if not exp[-1]:
                    epi_rwd+=exp[1]
                    current_state=exp[0]
                    step_cnt=step_cnt+1
                else:
                    print('Test total reward: '+str(epi_rwd))
                    step_cnt=0
                    epi_rwd=0
                    break


                # if exp[-1]:
                   
                    # return
                    # current_state=env.reset()
                    # print('===========Epi length:'+str(step_cnt)+' Total rwd = '+str(epi_rwd)+'==============')
                    # step_cnt=0
                    # epi_rwd=0


cs=env.reset()
agent=TD3PGAgent()
agent.train()
agent.test()

