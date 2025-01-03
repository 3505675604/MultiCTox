import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="./output/", help="the path of output model")
    # parser.add_argument("-c", "--class_num", type=int, default=11, help="the dimension of output")
    parser.add_argument("-i", "--iterations", type=int, default=10, help="the number of running model")
    parser.add_argument("-e", "--epoch", type=int, default=2000, help="the max number of epoch")
    parser.add_argument("-s", "--seed", type=int, default=1, help="random seed")

    parser.add_argument('--hidden_feats', type=list, default=[192, 384], help="the size of node representations after the i-th GAT layer")
    parser.add_argument('--rnn_embed_dim', type=int, default=128, help="the embedding size of each SMILES token")
    parser.add_argument('--rnn_hidden_dim', type=int, default=384, help="the number of features in the RNN hidden state")
    parser.add_argument('--rnn_layers', type=int, default=2, help="the number of rnn layers")
    parser.add_argument('--fp_dim', type=int, default=512, help="the hidden size of fingerprints module")
    parser.add_argument('--head', type=int, default=12, help="the head size of attention")
    parser.add_argument('--p', type=float, default=0.5, help="dropout probability")
    parser.add_argument('--lr', type=float, default=0.0001,  help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=float, default=64)#cav 64 nav 256 herg 1024  esol 32
    parser.add_argument('--batch_size1', type=float, default=32)# esol 32


    parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--epochs1', type=int, default=10000)
    parser.add_argument('--max_lr', type=float, default=2e-4, help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4, help='Final learning rate')
    return parser.parse_args()