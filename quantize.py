from tqdm import tqdm
import json
from argparse import ArgumentParser

import math

QUANTIZATION_BITS = 8

def quantize(value, scale):
    return int(math.ceil(value * scale))

def find_max_value(input_filename):
    max_val = 0
    with open(input_filename) as input_file:
        for docid, line in tqdm(enumerate(input_file)):
            for t in line.strip().split(","):
                split_list = t.strip().split(": ")
                if len(split_list) == 2:
                    term, score = split_list
                    max_val = max(max_val, float(score))
    return max_val

def process(input_filename, output_filename):
    max_val = find_max_value(input_filename)
    print(max_val)
    input()
    scale = (1<<QUANTIZATION_BITS)/max_val
    with open(input_filename) as input_file, open(output_filename, "w+") as output_file:
        for docid, line in tqdm(enumerate(input_file)):
            data = {}
            data["id"] = docid
            data["contents"] = ""
            data["vector"] = {}
            for t in line.strip().split(","):
                split_list = t.strip().split(": ")
                if len(split_list) == 2:
                    term, score = split_list
                    assert float(score) <= max_val
                    new_score = quantize(float(score), scale)
                    data["vector"][term] = float(score)
                    #output_file.write("{}:{},".format(term, quantize(float(score), scale)))
                    if "##" in term:
                        data["contents"] += term.split("##")[0]
                    else:
                        data["contents"] += " " + term
            json.dump(data, output_file)
            output_file.write("\n")


def main():
    parser = ArgumentParser(description='Quantize a DeepImpact collection.')
    parser.add_argument('--input', dest='input', required=True)
    parser.add_argument('--output', dest='output', required=True)
    args = parser.parse_args()
    process(args.input, args.output)

if __name__ == "__main__":
    main()
