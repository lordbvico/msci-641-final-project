#!/usr/bin/python3
import argparse
import json
import csv
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 2 that spoils each clickbait post with the title of the linked page.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The spoiled posts in jsonl format.', required=False)

    return parser.parse_args()


def predict(inputs):
    for index, i in enumerate(inputs):
        yield {'id': index, 'spoiler': i['targetTitle']}

def run_baseline(input_file, output_file):
    output_data = []
    with open(input_file, 'r') as inp, open(output_file, 'w') as out:
        inp = [json.loads(i) for i in inp]
        for output in predict(inp):
            #out.write(json.dumps(output) + '\n')
            output_data.append(output)
    
    df = pd.DataFrame(output_data)
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    args = parse_args()
    run_baseline(args.input, args.output)

