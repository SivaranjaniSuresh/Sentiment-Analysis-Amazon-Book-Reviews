import argparse
import os
import random
import time

from graphsage import *
from layers import *
from model import *
from sklearn.model_selection import train_test_split
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
	Training CARE-GNN
	Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
	Source: https://github.com/YingtongDou/CARE-GNN
"""


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset and model dependent args
    parser.add_argument("--data", type=str, default="yelp", help="The dataset name. [yelp, amazon]")
    parser.add_argument("--model", type=str, default="CARE", help="The model name. [CARE, SAGE]")
    parser.add_argument(
        "--inter", type=str, default="GNN", help="The inter-relation aggregator type. [Att, Weight, Mean, GNN]"
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size 1024 for yelp, 256 for amazon.")

    # hyper-parameters
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--lambda_1", type=float, default=2, help="Simi loss weight.")
    parser.add_argument("--lambda_2", type=float, default=1e-3, help="Weight decay (L2 loss weight).")
    parser.add_argument("--emb-size", type=int, default=64, help="Node embedding size at the last layer.")
    parser.add_argument("--num-epochs", type=int, default=31, help="Number of epochs.")
    parser.add_argument("--test-epochs", type=int, default=3, help="Epoch interval to run test set.")
    parser.add_argument("--under-sample", type=int, default=1, help="Under-sampling scale.")
    parser.add_argument("--step-size", type=float, default=2e-2, help="RL action step size")

    # other args
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disables CUDA training.")
    parser.add_argument("--seed", type=int, default=72, help="Random seed.")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def main(args):
    print(f"run on {args.data}")
    device = "cuda" if args.cuda else "cpu"

    # load graph, feature, and label
    if args.data in ("yelp", "amazon"):
        graphs, feat_data, labels = load_data(args.data)
    else:
        # New data loading function
        homo, relations, feat_data, labels = load_data_new(args.data)

    # train_test split
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.data == "yelp":
        index = list(range(len(labels)))
        idx_train, idx_test, y_train, y_test = train_test_split(
            index, labels, stratify=labels, test_size=0.60, random_state=2, shuffle=True
        )
    elif args.data == "amazon":  # amazon
        # 0-3304 are unlabeled nodes
        index = list(range(3305, len(labels)))
        idx_train, idx_test, y_train, y_test = train_test_split(
            index, labels[3305:], stratify=labels[3305:], test_size=0.60, random_state=2, shuffle=True
        )
    else:
        raise ValueError(f"Invalid data {args.data}")

    # split pos neg sets for under-sampling
    train_pos, train_neg = pos_neg_split(idx_train, y_train)

    # initialize model input
    features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
    feat_data = normalize(feat_data)
    features.weight.data = torch.FloatTensor(feat_data)
    features = features.to(device)

    # set input graph
    if args.model == "SAGE":
        adj_lists = homo
    else:
        adj_lists = relations

    print(f"Model: {args.model}, Inter-AGG: {args.inter}, emb_size: {args.emb_size}.")

    # build one-layer models
    if args.model == "CARE":
        intra_list = []
        for i in range(len(relations)):
            intra_list.append(IntraAgg(features, feat_data.shape[1], cuda=args.cuda))
        inter1 = InterAgg(
            features,
            feat_data.shape[1],
            args.emb_size,
            adj_lists,
            intra_list,
            inter=args.inter,
            step_size=args.step_size,
            cuda=args.cuda,
        )
        gnn_model = OneLayerCARE(2, inter1, args.lambda_1)

    elif args.model == "SAGE":
        agg1 = MeanAggregator(features, cuda=args.cuda)
        enc1 = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg1, gcn=True, cuda=args.cuda)

        enc1.num_samples = 5
        gnn_model = GraphSage(2, enc1)
    else:
        raise ValueError(f"Invalid model {args.model}")

    gnn_model = gnn_model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr, weight_decay=args.lambda_2
    )
    times = []
    performance_log = []

    # train the model
    for epoch in range(args.num_epochs):
        # randomly under-sampling negative nodes for each epoch
        sampled_idx_train = undersample(train_pos, train_neg, scale=1)
        rd.shuffle(sampled_idx_train)

        # send number of batches to model to let the RLModule know the training progress
        num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
        if args.model == "CARE":
            inter1.batch_num = num_batches

        loss = 0.0
        epoch_time = 0

        # mini-batch training
        for batch in range(num_batches):
            start_time = time.time()
            i_start = batch * args.batch_size
            i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
            batch_nodes = sampled_idx_train[i_start:i_end]
            batch_label = labels[np.array(batch_nodes)]

            optimizer.zero_grad()
            loss = gnn_model.loss(batch_nodes, torch.tensor(batch_label, dtype=torch.long, device=device))
            loss.backward()
            optimizer.step()

            end_time = time.time()
            epoch_time += end_time - start_time
            loss += loss.item()

        print(f"Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s")

        # testing the model for every $test_epoch$ epoch
        if epoch % args.test_epochs == 0:
            if args.model == "SAGE":
                test_sage(idx_test, y_test, gnn_model, args.batch_size)
            else:
                gnn_auc, label_auc, gnn_recall, label_recall = test_care(idx_test, y_test, gnn_model, args.batch_size)
                performance_log.append([gnn_auc, label_auc, gnn_recall, label_recall])


if __name__ == "__main__":
    args = parse_args()
    main(args)
