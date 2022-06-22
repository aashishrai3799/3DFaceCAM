import argparse, os, sys
import numpy as np
import torch
from models.AE import AE

parser = argparse.ArgumentParser(
        description='Main function to call training for different AutoEncoders')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--model', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
print(args)
args.cuda = True

ae = AE(args)
architectures = {'AE':  ae}

print(args.model)
if __name__ == "__main__":

    try:
        autoenc = architectures[args.model]
    except KeyError:
        print('---------------------------------------------------------')
        print('Model architecture not supported. ', end='')
        print('Maybe you can implement it?')
        print('---------------------------------------------------------')
        sys.exit()

    try:
        for epoch in range(1, args.epochs + 1):
            autoenc.train(epoch)
            # autoenc.test(epoch)
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")
