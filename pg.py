import torch 
import gym
import copy
import random as rand
import numpy as np
from torch.distributions import *
from torch.autograd import Variable
import torch.nn.functional as F

dev=torch.device("cuda")
dfloat= torch.cuda.FloatTensor
env=gym.make("Acrobot-v1")
BATCH_SIZE=32
# MEM_SIZE=1000000
EPOCHS = 5000000
EPISODE_LENGTH=50000
GAMMA=0.95


############################ Replay memory class #################################
# class ReplayMemory:
#     def __init__(self,n):
#         self.size=n
#         self.expBuffer=[]

#     # Circular memory
#     def push(self,exp):
#         if len(self.expBuffer)<self.size:
#             self.expBuffer=self.expBuffer+[exp]
#         else:
#             self.expBuffer.pop(0)
#             self.expBuffer=self.expBuffer+[exp]

#     # Check if buffer has sufficient experience
#     def isReady(self):
#         return len(self.expBuffer)>=BATCH_SIZE

#     def sampleBatch(self,sz=None):
#         if sz==None:
#             sz=BATCH_SIZE
#         idxs=[np.random.randint(0,len(self.expBuffer)) for _ in range(sz)]
#         return [self.expBuffer[idx] for idx in idxs]

# class AC_net(torch.nn.Module):
#     def __init__(self,n_f,n_a):
#         super(Actor_net,self).__init__()
#         self.common=torch.nn.Sequential(torch.nn.Linear(n_f,128),)
#         self.actor=torch.nn.Sequential(torch.nn.Linear(128,64), torch.nn.ReLU(),torch.nn.Linear(64,n_a),torch.nn.Softmax())
#         self.critic=torch.nn.Sequential(torch.nn.Linear(128,64), torch.nn.ReLU(),torch.nn.Linear(64,n_a))
    
#     def forward(self,st):
#         policy=self.actor(self.common(torch.from_numpy(st).float()))
#         val=self.critic(self.common(torch.from_numpy(st).float()))
#         return policy, val

# class Actor_net(torch.nn.Module):
#     def __init__(self,n_f,n_a):
#         super(Actor_net,self).__init__()
#         self.common=torch.nn.Sequential(torch.nn.Linear(n_f,128),)
#         self.actor=torch.nn.Sequential(torch.nn.Linear(128,64), torch.nn.ReLU(),torch.nn.Linear(64,n_a),torch.nn.Softmax())
#         # self.critic=torch.nn.Sequential(torch.nn.Linear(128,64), torch.nn.ReLU(),torch.nn.Linear(64,n_a))
    
#     def forward(self,st):
#         policy=self.actor(self.common(torch.from_numpy(st).float()))
#         # val=self.critic(self.common(torch.from_numpy(st).float()))
#         return policy

# class Critic_net(torch.nn.Module):
#     def __init__(self,n_f,n_a):
#         super(Critic_net,self).__init__()
#         self.common_st=torch.nn.Sequential(torch.nn.Linear(n_f,32),)
#         # self.common_a=torch.nn.Sequential(torch.nn.Linear(2,32),)
#         # self.actor=torch.nn.Sequential(torch.nn.Linear(128,64), torch.nn.ReLU(),torch.nn.Linear(64,n_a),torch.nn.Softmax())
#         self.critic=torch.nn.Sequential(torch.nn.Linear(32,16), torch.nn.ReLU(),torch.nn.Linear(16,1))
#         self.aeq={"0":torch.from_numpy(np.array([1,0])).float(),"1":torch.from_numpy(np.array([0,0])).float(),"2":torch.from_numpy(np.array([0,1])).float()}
    
#     def forward(self,st,a):
#         branches=[self.common_st(torch.from_numpy(st).float()),self.common_a(self.aeq[str(a.detach().numpy())])]
#         t=torch.cat(branches)
#         # val=self.critic(torch.cat(branches))
#         val=self.critic(torch.from_numpy(st).float())
#         return val

# class Critic_net(torch.nn.Module):
#     def __init__(self,n_f,n_a):
#         super(Critic_net,self).__init__()
#         self.common_st=torch.nn.Sequential(torch.nn.Linear(n_f,128),)
#         self.critic=torch.nn.Sequential(torch.nn.Linear(128,64), torch.nn.ReLU(),torch.nn.Linear(64,n_a))
#         self.aeq={"0":torch.from_numpy(np.array([1,0])).float(),"1":torch.from_numpy(np.array([0,0])).float(),"2":torch.from_numpy(np.array([0,1])).float()}
    
#     def forward(self,st):
#         # branches=[self.common_st(torch.from_numpy(st).float()),self.common_a(self.aeq[str(a.detach().numpy())])]
#         # t=torch.cat(branches)
#         # val=self.critic(torch.cat(branches))
#         val=self.critic(self.common_st(torch.from_numpy(st).float()))
#         return val

class ActorCritic(torch.nn.Module):
    def __init__(self,n_f,n_a):
        super(ActorCritic, self).__init__()
        self.affine = torch.nn.Linear(n_f, 128)
        
        self.action_layer = torch.nn.Linear(128, n_a)
        self.value_layer = torch.nn.Linear(128, 1)
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = F.relu(self.affine(state))
        
        state_value = self.value_layer(state)
        
        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()
    
    def calculateLoss(self, gamma=0.99):
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)   
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

############################# PG Agent ############################ 
class PGAgent:
    def __init__(self):
        super(PGAgent,self).__init__()
        # self.buff=ReplayMemory(MEM_SIZE)
        self.n_features= env.observation_space.shape[0]
        self.n_actions=env.action_space.n

        # self.actor=Actor_net(self.n_features,self.n_actions)
        # self.critic=Critic_net(self.n_features,self.n_actions)
        self.model=ActorCritic(self.n_features,self.n_actions)
        # self.epsilon=0.1
        self.clr=1e-4
        self.alr=1e-5
        self.opt=torch.optim.Adam(self.model.parameters(), lr=self.alr)
        # self.copt=torch.optim.Adam(self.critic.parameters(), lr=self.clr)

    def get_action(self,st):
        p=self.model.forward(st)
        # s=Categorical(p)
        return p#,s.sample()


    def train(self):
        epi_rwd=0
        BPerr=[]

        all_lengths = []
        # average_lengths = []
        all_rewards = []
        avg_rwds=[]

        # epi_Qs=[]
        for epi in range(EPOCHS):
            rewards=[]
            vals=[]
            log_probs=[]
            step_cnt=0
            epi_Q=None
            entropy=0
            v=None
            current_state=env.reset()
            self.model.clearMemory()
            for e in range(EPISODE_LENGTH):
                
                # p,act=self.get_action(current_state)
                act=self.model.forward(current_state)
                # print('p:'+str(p))
                # v=(p*q).sum()
                env.render()
                exp=env.step(act) # take action
                exp=(current_state,act)+exp[0:-1]
                self.model.rewards.append(exp[3])
                rewards.append(exp[3])
                # print('rew:'+str(rewards))
                # vals.append(v.detach().numpy())
                # print('vals========:'+str(vals))
                # print('==============')
                # log_p=torch.log(p)
                # log_probs.append(log_p.detach().numpy())
                # current_state=exp[2]
                step_cnt=step_cnt+1
                # entropy = entropy + np.sum(p.detach().numpy() * np.log(p.detach().numpy()))
                # pn,next_act=self.get_action(current_state)
                # td_err=0.01*exp[3]+GAMMA*(pn.detach()*self.critic.forward(current_state)).sum()-v
                # print(q.size())
                # print(log_p)
                # actor_loss = -v #(-log_p * (q-v.repeat(q.size()))).sum()
                # critic_loss = 0.5 * Variable(td_err).pow(2).sum()
                # ac_loss = self.model.calculateLoss(GAMMA)#actor_loss + critic_loss
                # # critic_loss=

                # # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                # # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                # # self.actor.zero_grad()
                # # self.critic.zero_grad()
                # self.model.zero_grad()
                # ac_loss = self.model.calculateLoss(GAMMA)
                # ac_loss.backward()
                
                # self.copt.step()
                # self.aopt.step()
                self.opt.step()

                if exp[-1] or e==EPISODE_LENGTH-1:
                    print('===========Epi length:'+str(step_cnt)+' Total rwd = '+str(sum(rewards))+'==============')
                    # all_rewards.append(sum(rewards))
                    # if len(all_rewards)<100:
                    #     avg_rwds.append(np.mean(all_rewards))
                    # else:
                    #     avg_rwds.append(np.mean(all_rewards[-100:]))
                    # all_lengths.append(e)
                    # print('rew:'+str(sum(rewards)))
                    break

            ac_loss = self.model.calculateLoss(GAMMA)
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            # self.actor.zero_grad()
            # self.critic.zero_grad()
            self.model.zero_grad()
            ac_loss = self.model.calculateLoss(GAMMA)
            ac_loss.backward()
            # epi_Q=np.zeros_like(vals)

            # v=self.critic.forward(current_state)
            # for t in reversed(range(len(rewards))):
            #     v=rewards[t] + GAMMA * v
            #     epi_Q[t]= v.detach().numpy()
            # for t in range(rewards):
            #     epi_Q=rewards+

            # values = torch.FloatTensor(vals)
            # Qvals = torch.FloatTensor(epi_Q)

            # log_probs = Variable(torch.FloatTensor(log_probs))
            # advantage = Qvals - values
            # actor_loss = (-log_probs * advantage).sum()
            # critic_loss = 0.5 * Variable(advantage).pow(2).sum()
            # ac_loss = torch.tensor(actor_loss + critic_loss, requires_grad = True) 
            # critic_loss=

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
            # self.actor.zero_grad()
            # self.critic.zero_grad()
            # ac_loss.backward()
            # self.copt.step()
            # self.aopt.step()
            # if len(avg_rwds)>500 and abs(avg_rwds[-1]-avg_rwds[-2])<0.00001:
            print('Avg reward:'+str(sum(rewards)))
                # break
            # self.buff.push(exp)
            # # check if transition valid
            # if not exp[-1]:
            #     epi_rwd+=exp[3]
            #     current_state=exp[2]
            #     step_cnt=step_cnt+1
            
            # # q updates
            # if self.buff.isReady():
            #     err=self.update()
            #     # print('Max Bellman Projection Error: '+str(err))
            #     BPerr.append(err)

            # if exp[-1]:
                # current_state=env.reset()
            
            # step_cnt=0
            # epi_rwd=0
            # self.epsilon=self.epsilon*0.99

    def test(self):
        current_state=env.reset()
        epi_rwd=0
        step_cnt=0
        BPerr=[]

        for _ in range(EPOCHS):
            p,v=self.forward(current_state)

            log_p=torch.log(p)

            s=Categorical(p)
            act=s.sample()
            exp=env.step(act) # take action
            env.render()
            # check if transition valid
            if not exp[-1]:
                epi_rwd+=exp[-2]
                current_state=exp[0]
                step_cnt=step_cnt+1
            else:
                break


            if exp[-1]:
                print('Test total reward: '+str(epi_rwd))
                return
                # current_state=env.reset()
                # print('===========Epi length:'+str(step_cnt)+' Total rwd = '+str(epi_rwd)+'==============')
                # step_cnt=0
                # epi_rwd=0


cs=env.reset()
agent=PGAgent()
agent.train()
# agent.test()