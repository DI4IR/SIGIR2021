import random
from argparse import ArgumentParser

import torch

from deep_impact import parameters
from deep_impact.training import trainer
from deep_impact.utils import create_directory


def main():
    parser = ArgumentParser(description='Training a DeepImpact model with <query, positive passage, negative passage> triples.')

    # The learning rate to use during training with the AdamW optimizer
    parser.add_argument('--lr', dest='lr', default=3e-06, type=float)
    # The maximum number of training epochs
    parser.add_argument('--maxsteps', dest='maxsteps', default=400000, type=int)
    # The batch size
    parser.add_argument('--bsize', dest='bsize', default=32, type=int)
    # The number of gradient accumulation steps
    parser.add_argument('--accum', dest='accumsteps', default=2, type=int)
    # The training file
    parser.add_argument('--triples', dest='triples', default='triples.train.small.tsv')
    # The folder for the output files
    parser.add_argument('--output_dir', dest='output_dir', default='outputs.train/')
    # The maximum lenght of a document (in terms or tokens?)
    parser.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)
    # The random seed to use
    parser.add_argument('--seed', dest='seed', default=42, type=int)

    args = parser.parse_args()

    assert args.bsize % args.accumsteps == 0, \
        f"The batch size {args.bsize} must be divisible by the number of gradient accumulation steps {args.accumsteps}."
    assert args.doc_maxlen <= 512, \
        f"The document max length {args.doc_maxlen} must be not larger than 512"

    parameters.MAX_LENGTH = args.doc_maxlen

    create_directory(args.output_dir)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainer.train(args.bsize, args.triples, args.maxsteps, args.lr, args.accumsteps)


if __name__ == "__main__":
    main()
