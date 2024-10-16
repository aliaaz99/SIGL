import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from typing import List
import torch.nn as nn
import math
from einops import rearrange
import random
import ot
import matplotlib as mpl
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''Classes'''
# sin activation
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# Siren layer:
class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = 'sine'):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.act = activation

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None

       
        if activation=='sine':
            self.activation=Sine(w0)
        elif activation=='relu':
            self.activation=nn.ReLU(inplace=False)
        elif activation=='id':
            self.activation=nn.Identity()
        elif activation=='sigmoid':
            self.activation=nn.Sigmoid()
        else:
            raise ValueError('No mlp activation specified')


    def init_(self, weight, bias, c, w0):
        dim = self.dim_in
        act = self.act

        if act =='relu':
            w_std = math.sqrt(1/dim)
        else:
            w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        
        weight.uniform_(-w_std, w_std)
        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out

# Siren network:    
class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, activation = 'sine', final_activation = 'sigmoid'):
        super().__init__()

        self.dim_hidden = dim_hidden
        self.num_layers = len(dim_hidden)
        self.layers = nn.ModuleList([])
        for ind in range(self.num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden[ind-1]
            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden[ind],
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                activation = activation
            ))

        final_activation = 'id' if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden[num_layers-1], dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x = x*rearrange(mod, 'd -> () d')   
                # x = x * mod

        return self.last_layer(x)

# GNN network using GCNConv:
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_dimensions):
        super(GCN, self).__init__()        
        self.layers = torch.nn.ModuleList()        
        self.layers.append(GCNConv(in_channels, hidden_channels[0]))        
        for i in range(1, len(hidden_channels)):
            self.layers.append(GCNConv(hidden_channels[i-1], hidden_channels[i]))  
        
        # Final fully connected layer
        self.fc1 = torch.nn.Linear(hidden_channels[-1], num_dimensions)        
    def forward(self, x, edge_index):
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
        x_pre_fc = x
        x_post_fc = self.fc1(x_pre_fc)
        x_post_fc = torch.sigmoid(x_post_fc)
        # Normalize the latent variables to [0,1]
        x_post_fc = (x_post_fc - x_post_fc.min())/(x_post_fc.max() - x_post_fc.min())        

        return x_pre_fc, x_post_fc
    
# SIGL network without pooling:
class SIGL(nn.Module):
    def __init__(self, model1, model2):
        super(SIGL, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, g):
        output_gnn_pre, output_gnn_post = self.model1(g.x, g.edge_index)
        n = output_gnn_post.size(0)
        z = output_gnn_post.squeeze(-1)
        
        x_coords = z.repeat_interleave(n)
        y_coords = z.repeat(n)
        edge_coords = torch.stack([x_coords, y_coords], dim=1)
        edge_coords = edge_coords.to(device)

        output_inr = self.model2(edge_coords)
        return output_inr, output_gnn_post


'''Functions'''
def nx2torch(adjacency_matrix):
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float)
    edge_index = torch.nonzero(adjacency_matrix, as_tuple=False).t().contiguous()
    edge_value = adjacency_matrix.reshape(-1)
    x = torch.randn(adjacency_matrix.size(0), 1)
    data = Data(x=x, edge_index=edge_index, y=edge_value).to(device)
    return data

def graph2XY(graphs, model_SIGL, latent_val=None, hsize=4):
    num_nodes = sum([graph_i.shape[0] for graph_i in graphs])
    X_all = []
    y_all = []
    w_all = []

    # Loop through each graph
    for graph_idx in range(len(graphs)):
        # Sorting:
        graph_test = graphs[graph_idx]
        data_i = nx2torch(graph_test)
        _, output_gnn_post = model_SIGL.model1(data_i.x, data_i.edge_index)
        perm_i = torch.argsort(output_gnn_post.squeeze(-1))
        perm = perm_i.cpu().numpy()  # Convert torch permutation to numpy array
        weight = graph_test.shape[0] / num_nodes
        graph_test_sort = graph_test[perm, :][:, perm]
        
        # Average pooling to have the histogram
        N = graph_test_sort.shape[0]
        if hsize == 1:
            h = 1
        if hsize == 2:
            h = 2
        if hsize == 3:
            h = 3
        if hsize == 4:
            h = int(np.log(N))
        if hsize == 5:
            h = int(np.sqrt(N))
        A = torch.tensor(graph_test_sort, dtype=torch.float, device=device)
        A = A.view(1, 1, N, N)
        H = F.avg_pool2d(A, kernel_size=(h, h))
        H = H.cpu().numpy().squeeze()  # Convert pooled result back to numpy
        k = H.shape[0]

        # Get the coordinates:
        x_coord = (np.arange(k) + 0.5) / k
        y_coord = (np.arange(k) + 0.5) / k
        xx, yy = np.meshgrid(x_coord, y_coord)
        X = np.column_stack((xx.ravel(), yy.ravel()))
        y = H.ravel().reshape(-1, 1)

        w = np.tile(weight, (X.shape[0], 1))
        if latent_val is not None:
            z_this = latent_val[graph_idx]
            zz = np.tile(z_this, (X.shape[0], 1))
            X = np.column_stack((X, zz))

        X_all.append(X)
        y_all.append(y)
        w_all.append(w)

    # Convert lists to tensors
    X_all = torch.tensor(np.concatenate(X_all, axis=0), dtype=torch.float, device=device)
    y_all = torch.tensor(np.concatenate(y_all, axis=0), dtype=torch.float, device=device)
    w_all = torch.tensor(np.concatenate(w_all, axis=0), dtype=torch.float, device=device)

    return X_all, y_all, w_all

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

def synthesize_graphon(r: int = 1000, type_idx: int = 0, alpha: int = 1) -> np.ndarray:
    """
    Synthesize graphons of Table 1
    :param r: the resolution of discretized graphon
    :param type_idx: the type of graphon
    :return:
        w: (r, r) float array, whose element is in the range [0, 1]
    """
    u = ((np.arange(0, r) + 1) / r).reshape(-1, 1)  # (r, 1)
    v = ((np.arange(0, r) + 1) / r).reshape(1, -1)  # (1, r)

    if type_idx == 0:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = u @ v
    elif type_idx == 1:
        w = np.exp(-(u ** 0.7 + v ** 0.7))
    elif type_idx == 2:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 0.25 * (u ** 2 + v ** 2 + u ** 0.5 + v ** 0.5)
    elif type_idx == 3:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 0.5 * (u + v)
    elif type_idx == 4:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 1 / (1 + np.exp(-2 * (u ** 2 + v ** 2)))
    elif type_idx == 5:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 1 / (1 + np.exp(-(np.maximum(u, v) ** 2 + np.minimum(u, v) ** 4)))
    elif type_idx == 6:
        w = np.exp(-np.maximum(u, v) ** 0.75)
    elif type_idx == 7:
        w = np.exp(-0.5 * (np.minimum(u, v) + u ** 0.5 + v ** 0.5))
    elif type_idx == 8:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = np.log(1 + 0.5 * np.maximum(u, v))
    elif type_idx == 9:
        w = np.abs(u - v)
    elif type_idx == 10:
        w = 1 - np.abs(u - v)
    elif type_idx == 101:
        theta = np.pi * 3 / 4
        w = 0.9*np.exp((-(v)**2-(u-1)**2)/alpha**2)+0.9*np.exp((-(u)**2-(v-1)**2)/alpha**2)+0.9*np.exp(-((np.sin(theta)*u+np.cos(theta)*v)/alpha)**2)
    elif type_idx == 11:
        r2 = int(r / 2)
        w = np.kron(np.eye(2, dtype=int), 0.8 * np.ones((r2, r2)))
    elif type_idx == 12:
        r2 = int(r / 2)
        w = np.kron(np.eye(2, dtype=int), np.ones((r2, r2)))
        w = 0.8 * (1 - w)
    
    # parametrized case of graphon 2 and SBM
    elif type_idx == 19:
        w = np.exp(-(u ** 0.7 + v ** 0.7)/alpha)
    elif type_idx == 14:
        w = np.zeros((r, r))
        R = int(r * alpha)
        w[:R, :R] = 0.8
        w[R:, R:] = 0.8
        w[:R, R:] = 0.1
        w[R:, :R] = 0.1
    else:
        raise ValueError('Unknown graphon type')

    np.fill_diagonal(w, 0.)
    return w

def simulate_graphs(w: np.ndarray, seed_gsize: int=123, seed_edge:int=123, num_graphs: int = 10,
                    num_nodes: int = 200, graph_size: str = 'fixed', offset:int=0) -> List[np.ndarray]:
    """
    Simulate graphs based on a graphon
    :param w: a (r, r) discretized graphon
    :param num_graphs: the number of simulated graphs
    :param num_nodes: the number of nodes per graph
    :param graph_size: fix each graph size as num_nodes or sample the size randomly as num_nodes * (0.5 + uniform)
    :return:
        graphs: a list of binary adjacency matrices
    """
    graphs = []
    r = w.shape[0]
	
    if graph_size == 'vary':
        numbers = np.linspace(75+offset,300+offset,num_graphs).astype(int).tolist()

    else:
        numbers = [num_nodes for _ in range(num_graphs)]
    
    np.random.seed(seed_edge) #add random seed for reproducibility
    for n in range(num_graphs):
        node_locs = (r * np.random.rand(numbers[n])).astype('int')
        graph = w[node_locs, :]
        graph = graph[:, node_locs]
        noise = np.random.rand(graph.shape[0], graph.shape[1])
        graph -= noise
        np.fill_diagonal(graph, 0)
        graphs.append((graph > 0).astype('float'))

    return graphs

def gw_distance(graphon: np.ndarray, estimation: np.ndarray) -> float:
    p = np.ones((graphon.shape[0],)) / graphon.shape[0]
    q = np.ones((estimation.shape[0],)) / estimation.shape[0]
    loss_fun = 'square_loss'
    dw2 = ot.gromov.gromov_wasserstein2(graphon, estimation, p, q, loss_fun, log=False, armijo=False)
    return np.sqrt(dw2)

def coords_prediction(inr_dim_hidden, gnn_dim_hidden, n_epochs, epoch_show, w0, graphs, lr):
    # Auxiliary graphon represented by an INR 
    model_inr = SirenNet(dim_in = 2, # input [x,y] coordinate
                dim_hidden = inr_dim_hidden,
                dim_out = 1, # output graphon (edge) probability 
                num_layers = len(inr_dim_hidden), # f_theta number of layers
                final_activation = 'sigmoid',
                w0_initial = w0).to(device)
    # GNN model to output the latent variables of the nodes
    model_gnn = GCN(1, gnn_dim_hidden, 1).to(device) 
    # End-to-end model of Step 1
    model_SIGL = SIGL(model_gnn, model_inr).to(device)
    # Train the model
    graph_data = [nx2torch(graph_i) for graph_i in graphs]
    optimizer = torch.optim.Adam(model_SIGL.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model_SIGL.train()
    for epoch in range(1,n_epochs+1):
        random.shuffle(graph_data)
        loss_e = 0
        for data_i in graph_data: 
            edges_pred, _ = model_SIGL(data_i)
            loss = criterion(edges_pred.squeeze(-1), data_i.y) 
            if torch.isnan(loss):
                break
            loss.backward() 
            optimizer.step() 
            optimizer.zero_grad()
            loss_e = loss_e + loss.item()
        if epoch==1 or epoch % epoch_show == 0:
            print("epoch: ", epoch, " loss: ", loss_e)


    return model_SIGL, loss_e

def train_graphon(inr_dim_hidden, w0, X_all, y_all, w_all, n_epochs, epoch_show, lr, batch_size, isparametric=0):
    inr_model = SirenNet(dim_in = 2 + isparametric , # input [x,y] coordinate
                dim_hidden = inr_dim_hidden,
                dim_out = 1, # output graphon (edge) probability 
                num_layers = len(inr_dim_hidden), # f_theta number of layers
                final_activation = 'sigmoid',
                w0_initial = w0).to(device)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_all, y_all, w_all), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(inr_model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss(reduction='none')
    for epoch in range(1, n_epochs+1):
        loss_e = 0
        for loader_idx, (X, y, w) in enumerate(train_loader):
            X, y, w = X.to(device), y.to(device), w.to(device)
            optimizer.zero_grad()
            y_pred = inr_model(X)
            loss = criterion(y_pred, y)
            weighted_loss = torch.mean(loss * w)
            weighted_loss.backward()
            optimizer.step()
            loss_e = loss_e + weighted_loss.item()
        if epoch==1 or epoch % epoch_show == 0:
            print('Epoch: {}, Loss: {:.5f}'.format(epoch, loss_e))

    return inr_model

def get_graphon(Res, model):
    # Square grid coordinates:
    x_coord = (np.arange(Res) + 0.5) / Res
    y_coord = (np.arange(Res) + 0.5) / Res
    xx, yy = np.meshgrid(x_coord, y_coord)
    X = np.column_stack((xx.ravel(), yy.ravel()))
    X_torch = torch.tensor(X, dtype=torch.float).to(device)
    # get the graphon on grid
    model.eval()
    with torch.no_grad():
        graphon_upper = model(X_torch)
    graphon = graphon_upper.cpu().numpy().reshape(Res, Res)
    graphon = (graphon + graphon.T) / 2
    np.fill_diagonal(graphon, 0)
    return graphon

def plot_smaples(true_graphon, predictions_graphon, name, model_SIGL, offset):
        # plot the true graphon and the predicted graphon
        plt.figure(figsize=(4, 4))
        mpl.rcParams['font.family'] = 'serif'
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
        plt.imshow(true_graphon)
        plt.axis('off')
        plt.title(r'True graphon $\omega$', fontsize=18)
        plt.tight_layout()
        plt.savefig('Plots/'+ name + "/True_graphon.jpg")

        plt.figure(figsize=(4, 4))
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
        plt.imshow(predictions_graphon)
        plt.axis('off')
        plt.title(r'Predicted graphon $f_{\theta^*}(x,y)$', fontsize=18)
        plt.tight_layout()
        plt.savefig('Plots/' + name + "/Predicted_graphon.jpg")

        # Generate a new graph with 150 nodes and plot the learned latent variables of nodes and the sorted graph
        N = 150
        model_SIGL.eval()
        test_graph = simulate_graphs(true_graphon, num_graphs=1, graph_size='fixed', num_nodes=N)[0]
        data_i = nx2torch(test_graph)
        _, coords = model_SIGL(data_i)
        coords = coords.cpu().detach().numpy()
        
        plt.figure(figsize=(4, 4))
        plt.scatter([np.sum(test_graph[:,j]) for j in range(test_graph.shape[0])], coords)
        plt.xlabel(r'Node degree', fontsize=16)
        plt.ylabel(r'$\hat{\boldsymbol{\eta}}$', fontsize=16)
        plt.title(r'Estimated latent variables', fontsize=18)
        plt.tight_layout()
        plt.savefig('Plots/' + name + "/Latent_vars.jpg")

        sorted_idx = np.argsort(coords, axis=0).squeeze()
        test_graph_sorted = test_graph[sorted_idx,:][:, sorted_idx]
        h = int(np.log(N))
        A = torch.tensor(test_graph_sorted, dtype=torch.float, device=device)
        A = A.view(1, 1, N, N)
        H = F.avg_pool2d(A, kernel_size=(h, h))
        H = H.cpu().numpy().squeeze()

        plt.figure(figsize=(4, 4))
        plt.imshow(H)
        plt.axis('off')
        plt.title(r'(Sorted) histogram approximation', fontsize=18)
        plt.tight_layout()
        plt.savefig('Plots/' + name + "/Histogram_app.jpg")
