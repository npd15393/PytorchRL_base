import torch
import torch.optim as optim

class RBF_classifier(torch.nn.module):
    def __init__(self):
        self.n_basis=100
        self.n_outputs=1
        self.wts=torch.randn(self.n_basis,self.n_outputs)
        self.n_features=8
        self.protos=torch.randn(self.n_features, self.n_basis)
        self.beta=1

    def forward(self,data_in, cs = None):
        if cs==None:
            cs=self.protos
        out_hid=torch.exp(self.beta*(torch.norm((data_in - cs).abs(),2, 1)))
        out=torch.sum(out_hid*self.wts, dim=1)
        return torch.nn.Softmax(out)

    # batch: features of data 
    def update_basis(self, batch):
        # perform km clustering
        n_clusters=self.n_basis
        current_centers=np.random.randint(0,len(batch),n_clusters)
        clusters=[[i,] for i in current_centers]
        mind=0
        min_idx=0
        stop=False
        while not stop:
            for i in range(len(batch)):
                for j in range(n_clusters):
                    cd=self.feature_dist(batch[clusters[j][0]],batch[i])
                    mind,min_idx=mind,min_idx if cd>mind else cd, j
                clusters[min_idx].append(i)


            hist=clusters
            # redefine centers
            for i in range(len(clusters)):
                clusters[i]=[self.cluster_mean(clusters[i]),]

            stop=False
            for c,h in zip(clusters,hist):
                stop=stop or not self.compare_clusters(c,h)

        cluster_stds=[self.cluster_std(c) for c in clusters]
        return clusters, self.cluster_stds
        

    # compare all points in cluster c and h
    def compare_clusters(self, c,h):
        fl=True
        for e in range(len(c)):
            fl=fl and torch.equal(c[e],h[e])
        return fl

    # compute mean of a cluster
    def cluster_mean(self, c):
        m=torch.tensor([0])
        for e in c:
            m=m+e 
        return m/len(c)

    # std of cluster c
    def cluster_std(self,c):
        std=torch.tensor([0])
        for i in range(1,len(c)):
            std=std+torch.pow(self.feature_dist(c[i],c[o]),2)
        return torch.pow(std,0.5)

    # euclidean dist bet two pts
    def feature_dist(self,f1,f2):
        return torch.norm((f1 - f2).abs(),2, 1)

    def train_step(self, batch):
        cs, stds = self.update_basis(batch)
        loss=self.forward(batch,cs)



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