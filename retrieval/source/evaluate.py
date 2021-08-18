import json
import argparse


def MAP(examples):
    map = 0
    precision, recall = [], []
    for ex in examples:
        average_precision, num_rel = 0.0, 0.0
        for j, ctx in enumerate(ex["ctxs"]):
            if ctx["id"] in ex["answers"]:
                num_rel += 1
                average_precision += num_rel / (j + 1)
        if num_rel > 0:
            average_precision = average_precision / num_rel

        precision.append(num_rel / len(ex["ctxs"]))
        recall.append(num_rel / len(ex["answers"]))
        map += average_precision

    return map / len(examples), precision, recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Log JSON file')
    args = parser.parse_args()

    with open(args.input_file) as f:
        examples = json.load(f)

    map, precision, recall = MAP(examples)
    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)
    print("MAP - ", round(map, 3))
    print("Precision - ", round(precision, 3))
    print("Recall - ", round(recall, 3))
