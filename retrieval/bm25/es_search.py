import json
import argparse
import jsonlines

from tqdm import tqdm
from elasticsearch import Elasticsearch


def read_jsonlines(file_name):
    lines = []
    print("loading examples from {0}".format(file_name))
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def search_es(es_obj, index_name, question_text, n_results=5):
    # construct query
    query = {
        'query': {
            'match': {
                'document_text': question_text
            }
        }
    }

    res = es_obj.search(index=index_name, body=query, size=n_results)

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_name', type=str, required=True, help='Path to index')
    parser.add_argument('--input_data_file', type=str, required=True, help='Path to index')
    parser.add_argument('--port', type=int, required=True, help='Port number')
    parser.add_argument('--output_fp', type=str, required=True)
    parser.add_argument('--keyword', type=str, required=True)
    parser.add_argument('--n_docs', type=int, default=100)

    args = parser.parse_args()
    input_data = read_jsonlines(args.input_data_file)
    config = {'host': 'localhost', 'port': args.port}
    es = Elasticsearch([config])
    result = {}

    for item in tqdm(input_data):
        if args.keyword == 'all':
            question = ' ; '.join(item["present"] + item["absent"])
        else:
            if len(item[args.keyword]) == 0:
                continue
            question = ' ; '.join(item[args.keyword])

        res = search_es(
            es_obj=es, index_name=args.index_name, question_text=question, n_results=args.n_docs
        )
        result[item["id"]] = {
            "hits": res["hits"]["hits"],
            "question": question
        }

    # evaluate top n accuracy
    for q_id in result:
        hits = result[q_id]["hits"]
        for hit in hits:
            if q_id == hit["_source"]["document_title"]:
                result[q_id]["found"] = True
                break

    with open(args.output_fp, 'w') as outfile:
        json.dump(result, outfile)

    top_n_accuracy = len([q_id for q_id, item in result.items() if item["found"] is True]) / len(result)
    print(top_n_accuracy)


if __name__ == '__main__':
    main()
