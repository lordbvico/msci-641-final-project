import argparse
import json
import csv
import pandas as pd

train = "train.jsonl"
test = "/content/drive/MyDrive/clickbait-detection-msci641-s23/test.jsonl"
#output = "output_file.jsonl"
output = "output_task.csv"
def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)

    return parser.parse_args()

def run_baseline(input_file, output_file):
    output_data = []
    with open(input_file, 'r') as inp:
        for index, i in enumerate(inp):
            i = json.loads(i)
            prediction = {'id': index, 'spoilerType': 'passage'}
            output_data.append(prediction)

    df = pd.DataFrame(output_data)
    df.to_csv(output_file, index=False)

#args = parse_args()
run_baseline(test, output)

#run_baseline(args.input, args.output)
#if __name__ == '__main__':
 #   args = parse_args()
  #  run_baseline(args.input, args.output)