from tools import *
import numpy as np
import argparse
import torch
import time
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='test_param', help='name of the experiment')
parser.add_argument('--Monotonic', type=int, default=0, help='Which parametric case (1: Monotonic, 0: SBM)')
parser.add_argument('--n_trials', type=int, default=10, help='Number of trials')
parser.add_argument('--n_G', type=int, default=10, help='Number of graphs in each trial')
parser.add_argument('--offset', type=int, default=0, help='Offset for the number of nodes in each graph')
parser.add_argument('--Res', type=int, default=1000, help='Resolution of the graphon')
parser.add_argument('--n-epoch', type=int, default=100, help='Number of traning epochs')
parser.add_argument('--epoch_show', type=int, default=20, help='Training checkpoint')
parser.add_argument('--gnn_dim_hidden', type=str, default="8,8,8", help='Hidden units per layer for GNN in step 1')
parser.add_argument('--inr_dim_hidden', type=str, default="20,20", help='Hidden units per layer for SIREN')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training INR in step 3')
parser.add_argument('--w0', type=float, default=10,help='Default frequency for sine activation of INR')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.parse_args()

args = parser.parse_args()
print("Parameters: \n", args)

name = args.name
isMonotonic = args.Monotonic
n_trials = args.n_trials
n_G = args.n_G
offset = args.offset
Res = args.Res

n_epochs = args.n_epoch
epoch_show = args.epoch_show
gnn_dim_hidden = [int(x) for x in args.gnn_dim_hidden.split(',')]
inr_dim_hidden = [int(x) for x in args.inr_dim_hidden.split(',')]
batch_size = args.batch_size
w0 = args.w0
lr = args.lr

# Initialization
if not os.path.exists('Plots/' + name):
    os.makedirs('Plots/' + name)

if isMonotonic:
    n_graphon = 19
    alpha = [1, 1.2, 1.4, 1.6, 1.8, 2]
else:
    n_graphon = 14
    alpha = [0.1, 0.2, 0.3, 0.4]

errors = np.zeros(n_trials)
All_time = np.zeros(n_trials)
min_error = 1e10
X_test_tensor = torch.stack([torch.linspace(0, 1, Res).repeat_interleave(Res), torch.linspace(0, 1, Res).repeat(Res)], dim=1).to(device)
each_alpha_error = []

for trial in range(n_trials):
    print("Trial: ", trial)
    all_graphs = []
    all_gt = []
    gt_alpha = []
    for alpha_i in alpha:
        true_graphon = synthesize_graphon(r=Res, type_idx=n_graphon, alpha=alpha_i)
        graphs = simulate_graphs(true_graphon, num_graphs=n_G, num_nodes=50, graph_size='vary', offset=offset)
        for i in range(len(graphs)):
            all_graphs.append(graphs[i])
            all_gt.append(true_graphon)
            gt_alpha.append(alpha_i)
    
    print("Number of graphs: ", len(all_graphs))
    print("Number of graphons: ", len(all_gt))
    print("Number of alphas: ", len(gt_alpha))

    ### Learn the coordinates and graphon using SIGL
    print("Learning the coordinates for all graphs in the dataset")
    if n_graphon==19:
        rand_idx = random.sample([0,2,3,4,5,6,7,8], 1)[0]
        true_graphon_coord = synthesize_graphon(r=Res, type_idx=rand_idx)
        myGraphCoords = simulate_graphs(true_graphon_coord, num_graphs=n_G, num_nodes=50, graph_size='vary', offset=offset)

    elif n_graphon==14:
        rand_idx = 12
        true_graphon_coord = synthesize_graphon(r=Res, type_idx=rand_idx)
        myGraphCoords = simulate_graphs(true_graphon_coord, num_graphs=n_G, num_nodes=50, graph_size='vary', offset=offset)

    print("Step1: Pretraining the GNN using graphon: ", rand_idx)
    start_time = time.time()
    model_SIGL, loss_coord = coords_prediction(inr_dim_hidden, gnn_dim_hidden, n_epochs, epoch_show, w0, myGraphCoords, lr)

    if loss_coord==0:
        print("The model did not learn latent variables well. skipping this trial")
        continue
    base_inr = model_SIGL.model2
    
    end_time = time.time()
    time_1 = end_time - start_time
    print("Time for learning the coordinates: ", time_1)

    base_inr.eval()
    with torch.no_grad():
        N_max = int(np.max([graph_i.shape[0] for graph_i in all_graphs]))
        print("N_max: ", N_max)
        X_base_graphon = torch.stack([torch.linspace(0, 1, N_max).repeat_interleave(N_max), torch.linspace(0, 1, N_max).repeat(N_max)], dim=1).to(device)
        predictions = base_inr(X_base_graphon)
    predictions_np = predictions.cpu().numpy()
    base_graphon = predictions_np.reshape(N_max, N_max)
    base_graphon = (base_graphon + base_graphon.T)/2
    np.fill_diagonal(base_graphon, 0.)

    # calculate the gw of all graphs with the base graphon as z_t
    print("Calculating the error of all graphs with the base graphon as z_t")
    start_time = time.time()
    all_error_base = []
    for i in range(len(all_graphs)):
        graph_i = all_graphs[i]
        n_i = graph_i.shape[0]
        base_i = base_graphon
        error_i = gw_distance(graph_i, base_i)
        all_error_base.append(error_i)
    all_error_base = np.array(all_error_base)
    all_error_base = (all_error_base - np.min(all_error_base)) / (np.max(all_error_base) - np.min(all_error_base))

    end_time = time.time()
    time_11 = end_time - start_time
    print("Time for calculating the error of all graphs with the base graphon: ", time_11)
    # plot the error of all graphs with the base graphon
    plt.figure(figsize=(4, 3))
    mpl.rcParams['font.family'] = 'serif'
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.scatter(gt_alpha, all_error_base, s=10)
    # plt.colorbar(label=r'$\alpha$')
    plt.xlabel(r'$\alpha$', fontsize=18)
    plt.ylabel(r'$z_t$', fontsize=18)
    if isMonotonic:
        plt.title(r'$\omega_{\alpha}^{(1)}$', fontsize=18)
    else:
        plt.title(r'$\omega_{\alpha}^{(2)}$', fontsize=18)
    plt.grid(True)  # Adding grid lines
    plt.tight_layout()
    plt.savefig('Plots/' + name + '/z_t.jpg')
    
    ### train INR using pooling based on new coordinates
    num_nodes_all = sum([graph_i.shape[0] for graph_i in all_graphs])
    train_data, test_data, train_gt, test_gt, train_alpha, test_gt_alpha, train_base_error, test_base_error = train_test_split(all_graphs, all_gt, gt_alpha, all_error_base, test_size=0.2)
    print("Number of training graphs: ", len(train_data))
    print("Number of testing graphs: ", len(test_data))
    print("Step2&3: Training the SIGL ")
    X_train, y_train, w_train = graph2XY(train_data, model_SIGL, latent_val=train_base_error)

    start_time = time.time()
    trained_inr = train_graphon(inr_dim_hidden, w0, X_train, y_train, w_train, n_epochs, epoch_show, lr, batch_size=batch_size, isparametric=1)
    end_time = time.time()
    time_2 = end_time - start_time
    print("Time for training the model: ", time_2)

    # get the estimated graphon from inr:
    print("Evaluating the model")
    start_time = time.time()
    trained_inr.eval()
    test_error = []
    all_z_test = []
    for i in range(len(test_data)):
        w_gt_i = test_gt[i]
        alpha_true = test_base_error[i]
        z = torch.tensor(alpha_true).to(device)
        zz = z.repeat(X_test_tensor.size(0), 1)
        X_i = torch.cat((X_test_tensor, zz), 1)
        X_i = X_i.clone().detach().to(dtype=torch.float, device=device)
        predictions_i = trained_inr(X_i)
        predictions_np = predictions_i.cpu().detach().numpy()
        W_pred = predictions_np.reshape(Res, Res)
        W_pred = (W_pred + W_pred.T)/2
        np.fill_diagonal(W_pred, 0.)
        error_i = gw_distance(w_gt_i, W_pred)
        test_error.append(error_i)
        all_z_test.append(alpha_true)


    error_trial = np.mean(test_error)
    print("GW trial : ", error_trial)
    errors[trial] = error_trial
    end_time = time.time()
    time_3 = end_time - start_time
    print("Time for evaluation: ", time_3)

    print("Time for the whole trial: ", time_1 + time_11 + time_2 + time_3)
    All_time[trial] = time_1 + time_11 + time_2 + time_3

    df = pd.DataFrame({'test_gt_alpha': test_gt_alpha, 'all_error': test_error})
    mean_errors_i = df.groupby('test_gt_alpha')['all_error'].mean().reset_index()
    each_alpha_error.append(mean_errors_i['all_error'])
    
    if error_trial <= min_error:
        min_error = error_trial
        # choose 4 random graphs from test set
        plt.figure(figsize=(20, 10))
        myGraphs = random.sample(range(len(test_data)), 4)
        for i in range(4):
            plt.subplot(2, 4, i+1)
            # get a random graph from the test set
            test_graph = test_data[myGraphs[i]]
            z_true = test_base_error[myGraphs[i]]
            alpha_i = test_gt_alpha[myGraphs[i]]
            data_i = nx2torch(test_graph)
            _, coords = model_SIGL(data_i)
            coords = coords.cpu().detach().numpy()
            
            plt.scatter([np.sum(test_graph[:,j]) for j in range(test_graph.shape[0])], coords)
            plt.xlabel('Node Degree')
            plt.ylabel('GNN output')
            plt.title('Node Coordinates in Test Graph ' + str(myGraphs[i]+1))

            # get the predicted graphon
            z = torch.tensor(z_true).to(device)
            zz = z.repeat(X_test_tensor.size(0), 1)
            X_i = torch.cat((X_test_tensor, zz), 1)
            X_i = X_i.clone().detach().to(dtype=torch.float, device=device)
            # X_i = X_test_tensor.clone().detach().to(dtype=torch.float, device=device)
            predictions_i = trained_inr(X_i)
            predictions_np = predictions_i.cpu().detach().numpy()
            W_pred = predictions_np.reshape(Res, Res)
            W_pred = (W_pred + W_pred.T)/2
            np.fill_diagonal(W_pred, 0.)
            
            plt.subplot(2, 4, i+5)
            plt.imshow(W_pred)
            plt.title('predicted graphon, latent z=' + str(np.round(z_true,2)) + ', alpha=' + str(alpha_i))        

        plt.tight_layout()
        plt.savefig('Plots/' + name + "/GNN_coords.jpg")


        # generate 4 random graphs with size 100:
        plt.figure(figsize=(20, 6))
        for j in range(5):
            z_i = (j+1)/5
            z = torch.tensor(z_i).to(device)
            zz = z.repeat(X_test_tensor.size(0), 1)
            X_j = torch.cat((X_test_tensor, zz), 1)
            X_j = X_j.clone().detach().to(dtype=torch.float, device=device)
            predictions_i = trained_inr(X_j)
            predictions_np = predictions_i.cpu().detach().numpy()
            W_pred_j = predictions_np.reshape(Res, Res)
            W_pred_j = (W_pred_j + W_pred_j.T)/2
            np.fill_diagonal(W_pred_j, 0.)

            plt.subplot(1, 5, j+1)
            plt.imshow(W_pred_j)
            plt.title('latent z=' + str(z_i))

        plt.tight_layout()
        plt.savefig('Plots/' + name + "/sampleGraph_60.jpg")



# Plot the results
plt.figure(figsize=(8, 6))
each_alpha_error = np.array(each_alpha_error)
print(each_alpha_error)

each_alpha_error_mean = np.mean(each_alpha_error, axis=0)
print(each_alpha_error_mean)

each_alpha_std = np.std(each_alpha_error, axis=0)
print(each_alpha_std)


# print the average error
print("Average time: ", np.mean(All_time))
# remove nan values from errors
errors = errors[~np.isnan(errors)]
print("Average GW distance between true and predicted graphon: ", np.round(np.mean(errors), 4))
print("Standard deviation of GW distance between true and predicted graphon: ", np.round(np.std(errors), 4))

print("Error for each alpha seperately: ", np.round(each_alpha_error_mean,3))
print("Standard deviation for each alpha seperately: ", np.round(each_alpha_std,3))
print("Average error across all alpha: ", np.round(np.mean(each_alpha_error), 3))

