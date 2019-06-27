import torch
import torch.optim as optim
import numpy as np
device = torch.device("cpu")
import copy
import collections
compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

class fileReader:
    def __init__(self,filenm,dlim=None):
        self.data_obj=open(filenm,"r")
        data_lst=self.data_obj.readlines()
        self.data=[list(map(float,d.rstrip().split(dlim))) for d in data_lst]
        self.labels=[d.pop(-1) for d in self.data]
        # self.labels=[2*(d.pop(-1))-1 for d in self.data]
        # self.labels=[[d,1-d] for d in self.labels]
        self.labels=torch.tensor(self.labels,dtype=torch.long)

    def normalized_data(self,):
        ndata=np.array(self.data)
        d=np.mean(ndata,axis=0)
        s=np.std(ndata,axis=0)

        return list(((ndata-d)/s).astype(float))



class RBF_classifier:
    def __init__(self):
        self.n_basis=15
        self.n_outputs=2
        # self.wts=torch.randn(self.n_basis,self.n_outputs,device=device, requires_grad=True)
        self.n_features=8
        self.protos=torch.randn(self.n_features, self.n_basis)
        self.beta=0.1
        self.model = torch.nn.Sequential(
        torch.nn.Linear(self.n_basis,self.n_outputs),
        torch.nn.Softmax())
        self.model=self.model.float()

    def eval(self,data_in, cs = None):
        if cs==None:
            cs=self.protos

        dat=torch.tensor(data_in,dtype=torch.float)
        dat=dat.view(*dat.size(),1)
        c=torch.tensor([cs]*len(data_in),dtype=torch.float)
        sh=c.size()
        c=c.view(len(data_in),sh[1],sh[2])
        c=c.transpose(dim0=1,dim1=2)

        out_hid=torch.exp(self.beta*(torch.norm((dat - c).abs(),2, 1)))

        # out=torch.sum(out_hid*self.wts, dim=1)
        out=self.model(torch.tensor(out_hid))

        return out

    # batch: features of data 
    def update_basis(self, batch):
        # perform km clustering
        n_clusters=self.n_basis
        current_centers=np.random.randint(0,len(batch),n_clusters)

        clusters=[[batch[i],] for i in current_centers]
    
        stop=False
        cnt=0
        hist=None
        while not stop:
            for i in range(len(batch)):
                mind=self.feature_dist(torch.tensor(clusters[0][0],dtype=torch.float),torch.tensor(batch[i],dtype=torch.float))
                min_idx=0
                for j in range(n_clusters):
                    cd=self.feature_dist(torch.tensor(clusters[j][0],dtype=torch.float),torch.tensor(batch[i],dtype=torch.float))
                    (mind,min_idx)=(mind,min_idx) if cd>mind else (cd, j)
                clusters[min_idx].append(batch[i])

            cnt=cnt+1
            # print(cnt)

            
            stop=False
            # if not hist==None:
            #     for c,h in zip(clusters,hist):
            #         stop=stop or self.compare_clusters(c,h)
            if cnt==50:
                break

            # if cnt%2==0:
            #     print('stds: '+str(sum([self.cluster_std(c) for c in clusters])))

            hist=copy.deepcopy(clusters)

            # redefine centers
            for i in range(len(clusters)):
                clusters[i]=[self.cluster_mean(clusters[i]),]

            print('mean:')
            print(clusters)

        print(cnt)
        self.cluster_stds=[self.cluster_std(c) for c in clusters]
        self.protos=[list(c[0].data.numpy()) for c in clusters]
        # print(self.protos)
        return clusters, self.cluster_stds
        

    # compare all points in cluster c and h
    def compare_clusters(self, c,h):
        fl=True

        for e in range(len(c)):
            fl=fl and compare(c[e],h[e])
        return fl

    # compute mean of a cluster
    def cluster_mean(self, c):
        m=0
        for e in c:
            y=list(map(float,e))
            m=m+torch.tensor(y)#list(np.array(y).astype(float))) 
        return m/len(c)

    # std of cluster c
    def cluster_std(self,c):
        # g=[list(c[0].data.numpy()) for c in clusters]
        # print(g)
        std=torch.tensor([0],dtype=torch.float)

        for i in range(1,len(c)):
            std=std+torch.pow(self.feature_dist(torch.tensor(c[i],dtype=torch.float),torch.tensor(c[0],dtype=torch.float)),2)
        return torch.pow(std,0.5)

    # euclidean dist bet two pts
    def feature_dist(self,f1,f2):
        return torch.norm((f1 - f2).abs(),2, 0)

    def train_step(self, batch, labels, opt):
        cs, stds = self.update_basis(batch)
        loss = torch.nn.CrossEntropyLoss()# self.forward(batch,cs)
        # loss=torch.nn.LogSoftmax()
        # u=self.eval(batch,self.protos)
        # v= torch.tensor(labels)

        grads=loss(self.eval(batch,self.protos), labels)
        opt.zero_grad()
        grads.backward()
        opt.step()

        # return statistics
        return grads

rbf=RBF_classifier()
f=fileReader("data0.txt",",")
c=RBF_classifier()
# opt=torch.optim.SGD(c.model.parameters(),lr=0.001, momentum=0.9) 
# for epi in range(100):
#     l=c.train_step(f.normalized_data(),f.labels,opt)
#     print(l.data)

# test kmeans
c.update_basis(f.data)
print(c.protos)

# class RBF_regressor(torch.nn.module):
#     def __init__(self):
#         self.n_basis=100
#         self.n_outputs=1
#         self.wts=torch.randn(self.n_basis,self.n_outputs)
#         self.n_features=8
#         self.protos=torch.randn(self.n_features, self.n_basis)
#         self.beta=1

#     def forward(self,data_in):
#         out=torch.exp(self.beta*(torch.norm((data_in - self.protos).abs(),2, 1)))

#     def train_step(self, batch,labels):
#         self.update_basis()
#         loss= torch.nn.CrossEntropyLoss()
#         optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

#         grads = loss(self.forward(batch), labels)
#         grads.backward()
#         optimizer.step()