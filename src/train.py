import argparse
from src.training.training import train


def main():

    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument("--fp16", action="store_true",
                        help="If passed, will use FP16 training.")
    parser.add_argument('--lr', dest='lr', default=3e-06, type=float)
    parser.add_argument(
        '--warmup_steps', dest='warmup_steps', default=100, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        default=16, type=int)
    parser.add_argument('--max_length', dest='max_length',
                        default=300, type=int)
    parser.add_argument('--local_rank', dest='rank', default=0, type=int)
    parser.add_argument('--gradient_accumulation_steps',
                        dest='gradient_accumulation_steps', default=1, type=int)

    parser.add_argument('--checkpoint', dest='checkpoint', required=False)
    parser.add_argument('--triples', dest='triples', required=True)
    parser.add_argument('--queries', dest='queries', required=True)
    parser.add_argument('--collection', dest='collection', required=True)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
