from argparse import ArgumentParser
from tqdm import tqdm

import pyterrier as pt
pt.init()

def tf2text(tokens):

    if tokens is None:
        return ''

    doc = []
    for t_i in tokens.split(' '):
        try:
            term, impact = t_i.split(':')
            doc += [term] * int(impact)
        except:
            print(tokens)
    return ' '.join(doc)

def deep_impact_generate(filename):
    with pt.io.autoopen(filename, 'rt') as corpusfile:
        for line in tqdm(corpusfile, desc='Indexing', unit=' documents'):
            docno, tokens = line.split("\t")
            yield {'docno' : docno, 'text' : tf2text(tokens)}


def main():

    parser = ArgumentParser(description='Compute a Terrier index using PyTerrier from the quantize.py output.')

    # The input file
    parser.add_argument('--input', dest='input', required=True)
    # The output file
    parser.add_argument('--output', dest='output', required=True)

    args = parser.parse_args()

    iter_indexer = pt.IterDictIndexer(args.output)
    # no stemming, no stopwords removal
    iter_indexer.setProperty("termpipelines", "")
    iter_indexer.index(deep_impact_generate(args.input))


if __name__ == "__main__":
    main()
