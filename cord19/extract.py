import argparse
import csv
import os


def main(args):
    src_writer = open(os.path.join(args.out_dir, 'test.source'), 'w', encoding='utf8')
    idx_writer = open(os.path.join(args.out_dir, 'test.idx'), 'w', encoding='utf8')
    empty_title, empty_abstract = 0, 0
    with open(args.csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            cord_uid = row[0]
            title = row[3]
            abstract = row[8]
            if len(title) == 0:
                empty_title += 1
            if len(abstract) == 0:
                empty_abstract += 1
            source = title + " [sep] " + abstract
            idx_writer.write(cord_uid + '\n')
            src_writer.write(source + '\n')

    src_writer.close()
    idx_writer.close()
    if empty_title > 0:
        print('{} empty title found'.format(empty_title))
    if empty_abstract > 0:
        print('{} empty abstract found'.format(empty_abstract))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", help='path to the .csv file')
    parser.add_argument("--out_dir", help='directory path where to save data')
    args = parser.parse_args()

    main(args)
