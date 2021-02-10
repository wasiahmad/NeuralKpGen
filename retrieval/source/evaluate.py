import json
import argparse


def MAP(examples):
    map = 0
    found = []
    for ex in examples:
        average_precision = 0
        has_found = False
        for j, ctx in enumerate(ex["ctxs"]):
            if ctx["id"] == ex["answers"][0]:
                average_precision = 1 / (j + 1)
                has_found = True
                break
        found.append(has_found)
        map += average_precision

    return map / len(examples), found


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Log JSON file')
    args = parser.parse_args()

    with open(args.input_file) as f:
        examples = json.load(f)

    map, found = MAP(examples)
    print("MAP - ", round(map, 3))
    top_k_acc = sum(found) / len(examples)
    print("Top-k accuracy - ", round(top_k_acc, 3))
