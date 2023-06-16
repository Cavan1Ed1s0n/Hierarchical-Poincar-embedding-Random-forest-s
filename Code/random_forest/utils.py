import numpy as np
import pandas as pd
import os
def get_tree_df(tree_clf):
    n_nodes = tree_clf.tree_.node_count
    children_left = tree_clf.tree_.children_left
    children_right = tree_clf.tree_.children_right
    features = tree_clf.tree_.feature
    thresholds = tree_clf.tree_.threshold

    node_depths = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depths[node_id] = depth

        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True


    pair = []
    rows = []
    for i in range(n_nodes):
        node_left = children_left[i]
        node_right = children_right[i]
        feature = features[i]
        feature_name = feature_names[features[i]]
        threshold = thresholds[i]
        node_depth = node_depths[i]
        is_leave = is_leaves[i]
        row = {
            'node_id': i,
            'node_depth': node_depth,
            'feature': feature,
            'feature_name': feature_name,
            'threshold': threshold,
            'is_leave': is_leave,
            'node_left': node_left,
            'node_right': node_right,
        }
        rows.append(row)
    tree_df = pd.DataFrame(rows)
    return tree_df


def get_name_node(row):
    name = f'{row.feature_name}_{row.name}_{row.node_depth}_{row.threshold}'
    return name

def extract_positive_pair(tree_df):
    
    pairs = []
    edge_pairs = []
    feat_pairs = []
    df = tree_df.set_index('node_id')
    for node_id, row in df.iterrows():
        if not row.is_leave:
            node_left = df.loc[row.node_left]
            node_right = df.loc[row.node_right]

            edge_pairs.append([get_name_node(row), get_name_node(node_left)])
            edge_pairs.append([get_name_node(row), get_name_node(node_right)])

    features = df.feature.unique()
    for feature in features:
        feat_df = df[(~df.is_leave)&(df.feature==feature)]
        for i in range(len(feat_df)):
            for j in range(i + 1, len(feat_df)):
                row = feat_df.iloc[i]
                row2 = feat_df.iloc[j]
                feat_pairs.append([get_name_node(row), get_name_node(row2)])

    print('feat_pairs', len(feat_pairs))
    print('edge_pairs', len(edge_pairs))
    pairs = feat_pairs + edge_pairs
    
    
    return pairs, edge_pairs, feat_pairs

def write_pair_to_file(pairs, edge_pairs, feat_pairs, dst):
    filename = os.path.join(dst, 'data_closure.tsv')
    filename_edge = os.path.join(dst, 'data_hierarchy.tsv')
    filename_feat = os.path.join(dst, 'data_feat.tsv')
    pd.DataFrame(edge_pairs).to_csv(filename, index=None ,header=None, sep='\t')
    pd.DataFrame(edge_pairs).to_csv(filename_edge, index=None ,header=None, sep='\t')
    pd.DataFrame(feat_pairs).to_csv(filename_feat, index=None ,header=None, sep='\t')