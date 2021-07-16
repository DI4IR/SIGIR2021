from argparse import ArgumentParser
from tqdm import tqdm
from more_itertools import chunked

import torch

from deep_impact.model import DeepImpactModel
from deep_impact import parameters, utils


def read_batch_iter(data_file, bsize):

    with open(data_file, mode="r", encoding="utf-8") as reader:
        batch = []
        for line in reader:
            pid, passage = line.split('\t')
            batch.append((pid, passage))
            if len(batch) == bsize:
                batch = sorted(batch, key=lambda x: x[1])
                yield batch
                batch = []


def process_doc_batch(di_model, super_batch, output):

    with torch.no_grad():

        for batch_chunk in chunked(super_batch, 128):
            pids = [x[0] for x in batch_chunk]
            docs = [x[1] for x in batch_chunk]

            _, term_scores = di_model.forward(docs, docs)

            lines = []
            for pid, t_s in zip(pids, term_scores):
                line = pid + '\t' + '\t'.join([f"{t}:{s}" for t, s in t_s.items()])
                lines.append(line)

            output.write('\n'.join(lines) + "\n")
            output.flush()


def main():

    parser = ArgumentParser(description='Indexing a collection with a trained DeepImpact model.')

    # The input file, e.g., queries.dev.small.tsv
    parser.add_argument('--input', dest='input', required=True)

    # The output file, e.g., queries.dev.test.txt
    parser.add_argument('--output', dest='output', required=True)

    # The deep impact trained, e.g., deepimpact.dnn
    parser.add_argument('--model', dest='model', required=True)

    # The batch size
    parser.add_argument('--bsize', dest='bsize', default=1024, type=int)

    args = parser.parse_args()

    utils.print_message("#> Loading model checkpoint.")
    di_model = DeepImpactModel.from_pretrained('bert-base-uncased')
    di_model = di_model.to(parameters.DEVICE)
    utils.load_checkpoint(args.model, di_model)
    di_model.eval()

    with open(args.output, 'w') as output:
        for super_batch in tqdm(read_batch_iter(args.input, args.bsize), desc='Indexing', unit=' passages', unit_scale=args.bsize):
            process_doc_batch(di_model, super_batch, output)

if __name__ == "__main__":
    main()
