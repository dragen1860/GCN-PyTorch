import  argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', default='cora')
args.add_argument('--model', default='gcn')
args.add_argument('--learning_rate', type=float, default=0.01)
args.add_argument('--epochs', type=int, default=200)
args.add_argument('--hidden', type=int, default=16)
args.add_argument('--dropout', type=float, default=0.5)
args.add_argument('--weight_decay', type=float, default=5e-4)
args.add_argument('--early_stopping', type=int, default=10)
args.add_argument('--max_degree', type=int, default=3)


args = args.parse_args()
print(args)