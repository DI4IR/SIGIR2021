import json
from tqdm import tqdm
from argparse import ArgumentParser


def process(input_filename, output_filename):
    with open(input_filename) as input_file, open(output_filename, "w+") as output_file:
        for docid, line in tqdm(enumerate(input_file)):
            data = {}
            data["id"] = docid
            data["contents"] = ""
            data["vector"] = {}
            for t in line.strip().split(","):
                split_list = t.strip().split(":")
                if len(split_list) == 2:
                    term, score = split_list
                    data["vector"][term] = float(score)
            json.dump(data, output_file)
            output_file.write('\n')


def main():
    parser = ArgumentParser(description='Convert a DeepImpact collection into an Anserini JsonVectorCollection.')
    parser.add_argument('--input', dest='input', required=True)
    parser.add_argument('--output', dest='output', required=True)
    args = parser.parse_args()
    process(args.input, args.output)

if __name__ == "__main__":
    main()

