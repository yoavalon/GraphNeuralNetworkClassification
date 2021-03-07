import matplotlib.pyplot as plt
import networkx as nx
import dgl
import torch
from dgl.nn.pytorch import GraphConv
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch.utils.tensorboard import SummaryWriter

ra = np.random.randint(0,100000)
print(ra)
writer = SummaryWriter('./log/' + str(ra))

def createGraph() : 

    G = nx.Graph()

    for i in range(12) : 
        a = np.random.randint(0,12)
        b = np.random.randint(0,12)
        G.add_node(a)
        G.add_node(b)
        G.add_edge(a, b)

    lab = nx.is_connected(G)
    if lab :         
        label = np.array([1])
    else :         
        label = np.array([0])

    return G, label

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.in_degrees().view(-1, 1).float()
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

# Create model
model = Classifier(1, 256, 2)  #changed classes
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

for ep in range(100000):
    
    gs = []
    lbs = []
    for i in range(32) : 
        graph, label = createGraph()
        label = torch.FloatTensor(label).type(torch.LongTensor)

        g = dgl.from_networkx(graph)
        gs.append(g)
        
        lbs.append(label)

    bg = dgl.batch(gs)
    lb = torch.stack(lbs)
   
    lb = torch.squeeze(lb)

    prediction = model(bg)
    loss = loss_func(prediction, lb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    re = torch.argmax(prediction, 1)
    acc = ((lb.eq(re.float())).sum()).float()/32

    writer.add_scalar('main/loss', loss, ep)
    writer.add_scalar('main/acc', acc, ep)

    if ep % 100 == 0 : 
        graph, label = createGraph()

        g = dgl.from_networkx(graph)
        prediction = model(g)
        pred = np.argmax(prediction.detach().numpy())
        
        fig = plt.figure()
        plt.title(f'pred= {pred} label= {label[0]}')        
        nx.draw(graph, with_labels=True, font_weight='bold')
        writer.add_figure("test", fig, ep)
