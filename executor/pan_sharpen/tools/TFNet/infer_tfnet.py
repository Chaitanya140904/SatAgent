import argparse
import shutil
import time

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
args, _ = parser.parse_known_args()

shutil.copy(args.input_path, args.output_path)
print(f"Executed dummy script, copied to {args.output_path}")
