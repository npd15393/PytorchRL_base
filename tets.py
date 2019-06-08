import torch 
import gym
import copy
import random as rand
import numpy as np

dev=torch.device("cuda")
dfloat= torch.cuda.FloatTensor
env=gym.make("MountainCar-v0")
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


############################# Q Agent ############################ 
class QLAgent:
    def __init__(self):
        self.buff=ReplayMemory(MEM_SIZE)
        self.n_features= env.observation_space.shape[0]
        self.n_actions=env.action_space.n

        self.epsilon=0.2
        self.lr=1e-4
        # self.alpha=lambda t:0.9*(1-t/1000)
        # define models
        self.qf=torch.nn.Sequential(torch.nn.Linear(self.n_features,128), \
             torch.nn.Linear(128,64), torch.nn.ReLU(),torch.nn.Linear(64,self.n_actions))
        s=env.reset()

        self.tqf=torch.nn.Sequential(torch.nn.Linear(self.n_features,128), \
             torch.nn.Linear(128,64), torch.nn.ReLU(),torch.nn.Linear(64,self.n_actions))
        # def opt and loss
        self.loss=torch.nn.MSELoss(reduction='sum')
        self.opt=torch.optim.RMSprop(self.qf.parameters(), lr=self.lr)
    
    def greedy(self,st):
        qs=self.qf(torch.from_numpy(st).float())
        idx=torch.argmax(qs).data.numpy()
        return idx

    def rand_act(self):
        return rand.randint(0,self.n_actions-1)

    def ex_policy(self,st):
        a=rand.random()
        if a<self.epsilon:
            return rand.randint(0,self.n_actions-1)
        else:
            qs=self.qf(torch.from_numpy(st).float())
            idx=torch.argmax(qs).data.numpy()
        return idx

    def update_tqf(self):
        primary_weights = list(self.qf.parameters())
        secondary_weights = list(self.tqf.parameters())
        n = len(primary_weights)
        for i in range(0, n):
            secondary_weights[i].data[:] = primary_weights[i].data[:]

        self.tqf.load_state_dict(self.qf.state_dict())

    def update(self):
        """
        Update Q function and policy
        :param exp: Experience tuple from Env
        :return: void
        """
        max_del=0
        exp=self.buff.sampleBatch()
        q_s=[]
        ss=[]
        for i in range(len(exp)):
            s=exp[i][0]
            a=exp[i][1]
            s_=exp[i][2]
            r=0.1*exp[i][3]
            done=exp[i][4]
            # print('exp:'+str(exp))
            # t=t+1
            # p_a=sto_policy(s,a,Q)
            # for _ in range(floor(1/p_a)):
                # q update
            ss.append(s)
            
            q_n=self.tqf(torch.from_numpy(s_).float())
            v=torch.max(q_n).data.numpy()
            q_target=(r+GAMMA*v) if not done else r
            # print('prev q '+str((s,a))+' : '+str(self.get_Q(s,a,self.Q)))
            q_s.append(self.qf(torch.from_numpy(s).float()).data.numpy())

            q_s[-1][a]=q_target#q_s[-1][a]+self.alpha(0)*(q_target-q_s[-1][a])

            # print('s:'+str(s)+' a:'+str(a))
            # print(alpha(ep)*(q_target))
            # max_del=max_del if max_del>abs(q_target-q_s[-1][a]) else abs(q_target-q_s[-1][a])
            # print('new q '+str((s,a))+' : '+str(self.get_Q(s,a,self.Q)))
            # keys=[(s,act) for act in self.actions]
            # tot_q=sum(np.exp(list(get_Q(s).values())))
            # for k in keys:
            # pi_s=get_pi(s)
            # pi_s[option_idxs(a)]= np.exp(get_Q(s,a))/tot_q #math.exp(self.get_Q(k[0],k[1]))/tot_q

        ls=self.loss(self.qf(torch.from_numpy(np.array(ss)).float()),torch.tensor(q_s))
        torch.nn.utils.clip_grad_norm(self.qf.parameters(), 1)
        self.opt.zero_grad()
        ls.backward()
        self.opt.step()
        nr=rand.random()
        if nr<0.2: 
            self.update_tqf()
        return max_del

    def train(self):
        current_state=env.reset()
        epi_rwd=0
        step_cnt=0
        BPerr=[]

        for epi in range(EPOCHS):
            if epi>5000:
                act=self.ex_policy(current_state)
            else:
                act=self.rand_act()

            exp=env.step(act) # take action
            env.render()
            exp=(current_state,act)+exp[0:-1]
            self.epsilon=self.epsilon*0.995
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
                act=self.greedy(current_state)

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
agent=QLAgent()
agent.train()
agent.test()

