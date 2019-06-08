import torch 
import gym
import copy
import random as rand
import numpy as np
import numpy.random as nr

dev=torch.device("cuda")
dfloat= torch.cuda.FloatTensor
env=gym.make("MountainCarContinuous-v0")
BATCH_SIZE=32
MEM_SIZE=10000
EPOCHS = 50000
GAMMA=0.95

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

class OUNoise:
    def __init__(self,action_dimension,mu=None, theta=0.15, sigma=0.3):
        self.action_dimension = action_dimension
        self.mu = np.zeros(action_dimension) if mu==None else mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

class ModelFactory:
    def __init__(self,env):
        self.buff=ReplayMemory(MEM_SIZE)
        self.n_features= env.low_state.shape[0]
        self.n_actions=1

        self.alr=1e-5
        self.clr=1e-4
        self.noise_model=OUNoise(self.n_actions)

        # define models
        self.st_fc=torch.nn.Sequential(torch.nn.Linear(self.n_features,128))
        self.critic=torch.nn.Sequential(torch.nn.Linear(128+self.n_actions,64), \
            torch.nn.ReLU(),torch.nn.Linear(64,1))
        self.tcritic=copy.deepcopy(self.critic)
        self.actor=torch.nn.Sequential(torch.nn.Linear(self.n_features,128), \
             torch.nn.Tanh(),torch.nn.Linear(128,self.n_actions),torch.nn.Tanh())
        self.tactor=copy.deepcopy(self.actor)

        self.actor_opt=torch.optim.Adam(self.actor.parameters(), lr=self.alr)
        self.critic_opt=torch.optim.Adam(self.critic.parameters(), lr=self.clr)

    def limit_action(self,a):
        pass

    def get_action(self,st,targ=False):
        state = torch.from_numpy(st).float()
        return self.tactor(state) if targ else self.actor(state)

    def get_noisy_action(self,st,targ=False):
        state = torch.from_numpy(st).float()
        
        noise=torch.tensor(self.noise_model.noise(),dtype=torch.float)
        # print('noise: '+str(noise))
        act=self.tactor(state) if targ else self.actor(state)
        return act.add(noise)

    def get_q(self, st, targ=False):
        state = torch.from_numpy(st).float()
        cat_dim=len(state.size())-1
        branches=[self.get_action(st,targ),self.st_fc(state)]
        return self.tcritic(torch.cat(branches,cat_dim)) if targ else self.critic(torch.cat(branches,cat_dim))

    def update_tcritic(self):
        primary_weights = list(self.critic.parameters())
        secondary_weights = list(self.tcritic.parameters())
        n = len(primary_weights)
        for i in range(n):
            secondary_weights[i].data[:] = primary_weights[i].data[:]

        self.tcritic.load_state_dict(self.critic.state_dict())

    def update_tactor(self):
        primary_weights = list(self.actor.parameters())
        secondary_weights = list(self.tactor.parameters())
        n = len(primary_weights)
        for i in range(n):
            secondary_weights[i].data[:] = primary_weights[i].data[:]

        self.tactor.load_state_dict(self.actor.state_dict())
        

############################# Q Agent ############################ 
class DDPGAgent:
    def __init__(self):
        self.buff=ReplayMemory(MEM_SIZE)
        self.n_features= env.low_state.shape[0]
        self.n_actions=1
        self.ddpg = ModelFactory(env)
        s=env.reset()

        self.critic_loss=torch.nn.MSELoss(reduction='sum')

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
        ls=0
        for i in range(len(exp)):
            s=exp[i][0]
            a=exp[i][1]
            s_=exp[i][2]
            r=0.1*exp[i][3]
            done=exp[i][4]

            # q update
            ss.append(s)
            
            q_n=self.ddpg.get_q(s_,True)
            q_targ=0.1*r+GAMMA*q_n if not done else r
            # print('prev q '+str((s,a))+' : '+str(self.get_Q(s,a,self.Q)))
            q_s=self.ddpg.get_q(s)
            # print('     qt qs:')
            # print(q_targ)
            # print(q_s)
            ls+=0.5*torch.pow(q_targ-q_s,2)
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
        # q_targ=np.array(q_targ)
        # q_s=np.array(q_s)

        # ls=0.5*torch.sum(torch.pow(torch.from_numpy(q_targ-q_s).float(),2))
        q=self.ddpg.get_q(ss)
        als=-q.mean()

        torch.nn.utils.clip_grad_norm(self.ddpg.critic.parameters(), 1)
        torch.nn.utils.clip_grad_norm(self.ddpg.actor.parameters(), 1)
        self.ddpg.critic_opt.zero_grad()
        self.ddpg.actor_opt.zero_grad()

        # critic update
        ls.backward()
        self.ddpg.critic_opt.step()
        nr=rand.random()
        if nr<0.1: 
            self.ddpg.update_tcritic()

        # actor update
        als.backward()
        self.ddpg.actor_opt.step()
        nr=rand.random()
        if nr<0.1: 
            self.ddpg.update_tactor()

        return max_del

    def train(self):
        current_state=env.reset()
        epi_rwd=0
        step_cnt=0
        BPerr=[]

        for epi in range(EPOCHS):
            act=self.ddpg.get_noisy_action(current_state,True)

            # print(act)
            exp=env.step(act.detach().numpy()) # take action
            # env.render()
            exp=(current_state,act)+exp[0:-1]
            # self.epsilon=self.epsilon*0.995
            self.buff.push(exp)
            # check if transition valid
            if not exp[-1]:
                epi_rwd+=exp[3]
                current_state=exp[2]
                step_cnt=step_cnt+1

            # q updates
            if self.buff.isReady():
                err=self.update()
                # print('Max Bellman Projection Error: '+str(err))
                BPerr.append(err)

            if exp[-1]:
                current_state=env.reset()
                print('===========Epi length:'+str(step_cnt)+' Total rwd = '+str(epi_rwd)+'==============')
                step_cnt=0
                epi_rwd=0
            # self.epsilon=self.epsilon*0.99

    def test(self):
        # current_state=env.reset()
        # epi_rwd=0
        # step_cnt=0
        BPerr=[]
        for _ in range(15):
            current_state=env.reset()
            epi_rwd=0
            step_cnt=0
            for _ in range(EPOCHS):
                act=self.ddpg.get_action(current_state)

                exp=env.step(act) # take action

                exp=exp[0:-1]
                env.render()
                # check if transition valid
                if not exp[-1]:
                    epi_rwd+=exp[-2]
                    current_state=exp[0]
                    step_cnt=step_cnt+1
                else:
                    print('Test total reward: '+str(epi_rwd))
                    break


                if exp[-1]:
                   
                    return
                    # current_state=env.reset()
                    # print('===========Epi length:'+str(step_cnt)+' Total rwd = '+str(epi_rwd)+'==============')
                    # step_cnt=0
                    # epi_rwd=0


cs=env.reset()
agent=DDPGAgent()
agent.train()
agent.test()

