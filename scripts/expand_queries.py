import json
from tqdm import tqdm
from argparse import ArgumentParser


def process(input_filename, output_filename):
    with open(input_filename) as input_file, open(output_filename, "w+") as output_file:
        for line in tqdm(input_file):
            query_id, line = line.split("\t")
            terms = ""
            for t in line.strip().split(","):
                split_list = t.strip().split(":")
                if len(split_list) == 2:
                    term, score = split_list
                    terms += " ".join([term] * int(score))
                    terms += " "
            output_file.write("{}\t{}\n".format(query_id, terms))


def main():
    parser = ArgumentParser(description='Expand queries according to the query weights.')
    parser.add_argument('--input', dest='input', required=True)
    parser.add_argument('--output', dest='output', required=True)
    args = parser.parse_args()
    process(args.input, args.output)

if __name__ == "__main__":
    main()
