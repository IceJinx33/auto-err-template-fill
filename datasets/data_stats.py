import json, argparse
import numpy as np
from nltk.tokenize import word_tokenize

def mention_tokens_index(doc, mention):
        """
        This function returns the starting and ending indexes of the tokenized mention
        in the tokenized document text.
        If the mention token list is not present (in order) in
        the list of document tokens, this function
        returns the start index as 1 and the end index as 0.
        :param doc: List of document tokens
        :type doc: List[strings]
        :param mention: List of mention tokens
        :type mention: List[strings]
        """
        start, end = -1, -1
        if len(mention) == 0:
            return 1, 0
        for i in range(len(doc)):
            if doc[i : i + len(mention)] == mention:
                start = i
                end = i + len(mention) - 1
                break
        if start == -1 and end == -1:
            return 1, 0
        return start, end

def add_script_args(parser):
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        help="The path to the data directory containing the train.json, dev.json and test.json files",
    )
    return parser

if __name__ == "__main__":

    parser = add_script_args(
            argparse.ArgumentParser(
                usage='Use "python3 data_stats.py --help" for more information',
                formatter_class=argparse.RawTextHelpFormatter,
            )
        )
    args = parser.parse_args()

    docid = 0

    min_tokens = np.infty
    max_tokens = 0
    total_tokens = 0

    min_temp = np.infty
    max_temp = 0
    total_temp = 0

    train_docs = 0
    dev_docs = 0
    test_docs = 0

    num_docs = 0

    min_dist = np.infty
    max_dist = -1
    total_dist = 0

    min_span = np.infty
    max_span = 0
    total_span = 0

    num_irr = 0
    num_rev = 0

    for file_name in ["train.json", "dev.json", "test.json"]:
        f = open(str(args.data_dir) + "/" + file_name)
        for line in f:
            num_docs += 1
            if file_name == "train.json":
                train_docs += 1
            elif file_name == "dev.json":
                dev_docs += 1
            elif file_name == "test.json":
                test_docs += 1
            ex = json.loads(line)
            doc_tokens = word_tokenize(ex["doctext"])
            num_tokens = len(doc_tokens)
            if num_tokens < min_tokens:
                min_tokens = num_tokens
            if num_tokens > max_tokens:
                max_tokens = num_tokens
            total_tokens += num_tokens

            num_temp = len(ex["templates"])
            if num_temp == 0:
                num_irr += 1
            else:
                num_rev += 1
            if num_temp < min_temp:
                min_temp = num_temp
            if num_temp > max_temp:
                max_temp = num_temp
            total_temp += num_temp

            min_start = len(doc_tokens) - 1
            max_end = 0

            
            mentions_list = [mentions[0] for template in ex["templates"] for role in template for corefs in template[role] for mentions in corefs if type(template[role]) == list]
            mentions_list += [template[role] for template in ex["templates"] for role in template if type(template[role]) != list]
    
            for mention in mentions_list:
                start, end = mention_tokens_index(doc_tokens, word_tokenize(mention))
                if start > end:
                    continue
                else:
                    if start < min_start:
                        min_start = start
                    if end > max_end:
                        max_end = end

            num_span = 0
            for mention in mentions_list:
                start, end = mention_tokens_index(doc_tokens, word_tokenize(mention))
                if start >= min_start and end <= max_end:
                    num_span += 1
                else:
                    continue

            token_dist = max_end - min_start

            if token_dist < min_dist:
                min_dist = token_dist
            if token_dist > max_dist:
                max_dist = token_dist
            total_dist += token_dist

            if num_span < min_span:
                min_span = num_span
            if num_span > max_span:
                max_span = num_span
                docid = ex["docid"]
            total_span += num_span


    print("Number of documents in train set: " + str(train_docs)) 
    print("Number of documents in dev set: " + str(dev_docs))
    print("Number of documents in test set: " + str(test_docs))
    print("******************************************************")
    print("Total no. of documents: " + str(num_docs))
    print("\n")
    print("Minimum no. of tokens in a document: " + str(min_tokens))
    print("Maximum no. of tokens in a document: " + str(max_tokens))
    print("Average no. of tokens in a document: " + str(total_tokens/num_docs))
    print("\n")
    print("Percentage of documents with 0 templates: " + str((num_irr/num_docs)*100))
    print("Maximum no. of templates in relevant documents: " + str(max_temp))
    print("Average no. of templates in relevant document: " + str(total_temp/num_rev))
    print("\n")
    print("Minimum no. of templates in a document: " + str(min_temp))
    print("Maximum no. of templates in a document: " + str(max_temp))
    print("Average no. of templates in a document: " + str(total_temp/num_docs))
    print("\n")
    print("Minimum token distance between extracted mentions in a document: " + str(min_tokens))
    print("Maximum token distance between extracted mentions in a document: " + str(max_tokens))
    print("Average token distance between extracted mentions in a document: " + str(total_tokens/num_docs))
    print("\n")
    print("Minimum no. of mention spans in maximal span range in a document: " + str(min_span))
    print("Maximum no. of mention spans in maximal span range in a document: " + str(max_span))
    print("Average no. of mention spans in maximal span range in a document: " + str(total_span/num_docs))