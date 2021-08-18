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


def search_es(es_obj, index_name, keywords, match_type='match', n_results=5):
    # construct query
    match = [{match_type: {"document_text": kw}} for kw in keywords]
    query = {
        "query": {
            "bool": {
                "should": match
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
        if args.keyword == 'query':
            keywords = [item['query']]
        elif args.keyword == 'all':
            keywords = item["present"] + item["absent"]
        else:
            keywords = item[args.keyword]

        if len(keywords) == 0:
            continue

        res = search_es(
            es_obj=es,
            index_name=args.index_name,
            keywords=keywords,
            match_type='match',
            n_results=args.n_docs
        )
        result[item["id"]] = {
            "question": keywords,
            "found": False,
            "hits": res["hits"]["hits"],
        }

    # evaluate top n accuracy
    for q_id in result:
        hits = result[q_id]["hits"]
        for hit in hits:
            if q_id == hit["_source"]["document_title"]:
                result[q_id]["found"] = True
                break
        # filtering fields to store less data
        result[q_id]["hits"] = [
            {
                'doc_id': h["_source"]["document_title"],
                'score': h["_score"]
            }
            for h in hits
        ]

    with open(args.output_fp, 'w') as outfile:
        json.dump(result, outfile, indent=True)

    top_n_accuracy = len([q_id for q_id, item in result.items() if item["found"]]) / len(result)
    print(top_n_accuracy)


if __name__ == '__main__':
    main()
