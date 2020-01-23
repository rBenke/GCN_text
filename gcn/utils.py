import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import os
from itertools import chain

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)



def load_data(dataset_str, max_sample):
    data_path = '../data/'
    files = [f for f in os.listdir(data_path) if f.startswith(dataset_str)]
    files = np.random.choice(files,max_sample,False) 
    
    adj_lst = list()
    features_lst = list()
    labels_lst = list()
    n_nodes_lst = list()
    
    # load and concatenate all graph data
    for i in range(files.size):
        file = open(data_path + files[i],'rb')
        
        adj = pkl.load(file)
        feature = pkl.load(file)
        label = pkl.load(file)
        n_nodes = [i+1]*feature.shape[0]

        adj_lst.append(adj)
        features_lst.append(feature)
        labels_lst.append(label)
        n_nodes_lst.append(n_nodes)
        
        file.close()
    
    # create labels matrix
    labels_mat =  np.vstack([[labels_lst[i]]*len(n_nodes_lst[i]) for i in range(len(labels_lst))])
    
    # create feature matrix
    features_mat =  np.vstack(features_lst)
    features_mat = sp.csr_matrix(features_mat)
    
    # adj block matrix
    nodes_before = 0
    nodes_after = features_mat.shape[0] 
    for i in range(len(adj_lst)):
        n_nodes = len(n_nodes_lst[i])
        nodes_after -= n_nodes
        if nodes_before>0:
            left_mat = sp.hstack([np.zeros((n_nodes, nodes_before)),adj_lst[i]])
        else :
            left_mat = adj_lst[i]
        if nodes_after>0:
          
            adj_lst[i] = sp.hstack([left_mat,np.zeros((n_nodes, nodes_after))])
        else:
            adj_lst[i] = left_mat
            
        nodes_before += n_nodes
    adj_mat = sp.vstack(adj_lst)
    
    # prepare masks (train/val/test sets)
    train_ratio = 0.8
    val_ratio = 0.2
    train_mask = np.zeros(len(files))
    val_mask = np.zeros(len(files))
    test_mask = np.zeros(len(files))
    
    train_mask_ind = np.random.choice(range(len(files)), int(len(files)*train_ratio), False)
    val_mask_ind = np.random.choice(train_mask_ind, int(len(train_mask_ind)*val_ratio), False)
    
    test_mask_ind = set(range(len(files))).difference(set(train_mask_ind))
    train_mask_ind = set(train_mask_ind).difference(set(val_mask_ind))
    
    for ind in train_mask_ind:
        train_mask[ind] = 1
    
    for ind in val_mask_ind:
        val_mask[ind] = 1
    
    for ind in test_mask_ind:
        test_mask[ind] = 1
    
    test_mask_mat =  np.vstack([[[test_mask[i]]*5]*len(n_nodes_lst[i]) for i in range(len(test_mask))])
    val_mask_mat =   np.vstack([[[val_mask[i]]*5]*len(n_nodes_lst[i]) for i in range(len(val_mask))])
    train_mask_mat =  np.vstack([[[train_mask[i]]*5]*len(n_nodes_lst[i]) for i in range(len(train_mask))])
    
    # check if the result is correct
    sum_check = test_mask_mat+val_mask_mat+train_mask_mat
    if (np.max(sum_check)>1):
        sys.exit()
    elif (np.any(sum_check==0)):
        sys.exit()
    
    # use mask for labels masking
    train_labels_mat = np.multiply(labels_mat, train_mask_mat)
    test_labels_mat = np.multiply(labels_mat, test_mask_mat)
    val_labels_mat = np.multiply(labels_mat, val_mask_mat)
    
    # check if the result is correct
    sum_check = train_labels_mat+test_labels_mat+val_labels_mat
    if (np.sum(sum_check)!= test_labels_mat.shape[0]):
        sys.exit()    
    
    return adj_mat, features_mat, train_labels_mat, test_labels_mat, val_labels_mat, train_mask_mat, test_mask_mat, val_mask_mat
