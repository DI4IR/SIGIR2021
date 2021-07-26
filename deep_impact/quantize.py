import math
from functools import partial

from argparse import ArgumentParser
from tqdm import tqdm

def read_batch_iter(data_file, bsize):

    with open(data_file, mode="r", encoding="utf-8") as reader:
        batch = []
        for line in reader:
            pid, *tokens = line.split('\t')
            batch.append((pid, tokens))
            if len(batch) == bsize:
                yield batch
                batch = []
        if batch:
            yield batch


def process_doc_batch(quantizer, batch, output):

    lines = []
    for pid, tokens in batch:
        ts_dict = {x.split(':')[0]: quantizer(float(x.split(':')[-1])) for x in tokens[1:] if float(x.split(':')[-1]) > 0}
        line = pid + '\t' + ' '.join([f"{t}:{s}" for t, s in ts_dict.items() if s > 0])
        lines.append(line)
    output.write('\n'.join(lines) + "\n")
    output.flush()


def quantize(value, scale):

    return int(math.ceil(value * scale))


def compute_max(filename):

    max_value = float('-inf')
    with open(filename, 'rt') as input_f:
        for line in tqdm(input_f, desc='Computing min and max scores'):
            tokens = line.strip().split('\t')
            scores = [float(x.split(':')[-1]) for x in tokens[1:]]
            if scores:
                max_value = max(max_value, max(scores))
    return max_value


def main():

    parser = ArgumentParser(description='Quantize a collection trained DeepImpact model.')

    # The input file
    parser.add_argument('--input', dest='input', required=True)
    # The output file
    parser.add_argument('--output', dest='output', required=True)
    # The number of quantization bits
    parser.add_argument('--bits', dest='bits', default=8, type=int)
    # The batch size
    parser.add_argument('--bsize', dest='bsize', default=1024, type=int)

    args = parser.parse_args()

    max_value = compute_max(args.input)
    print(f'Upper bound is {max_value}')

    with open(args.output, 'w') as output:
        quantizer = partial(quantize, scale=(1 << args.bits)/max_value)
        for super_batch in tqdm(read_batch_iter(args.input, args.bsize), desc='Quantizing', unit=' passages', unit_scale=args.bsize):
            process_doc_batch(quantizer, super_batch, output)


if __name__ == "__main__":
    main()
