import json, argparse
from collections import OrderedDict

# Used for creating the heuristic for clustering the DyGIE++ output to templates

# ProMed
# role_names = ["Status", "Country", "Disease", "Victims"]

# SciREX
role_names = ["Material", "Method", "Metric", "Task"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", default=None, type=str, required=False, help="Dygie output file.")
    parser.add_argument("-o", "--output_file", default=None, type=str, required=False, help="Formatted template file for eval.py.")
    args = parser.parse_args()

    input_file = open(args.input_file, "r")

    docs = {}

    for line in input_file:
        ex = json.loads(line)
        doc = {}

        doc_tokens = ex["sentences"][0]
        doc["pred_templates"] = []
        temp = {}

        for start, end, role_name, _, _ in ex["predicted_ner"][0]:
          if role_name in role_names:
            if role_name in temp:
              temp[role_name].append([" ".join(doc_tokens[start: end+1])])
            else:
              temp[role_name] = [[" ".join(doc_tokens[start: end+1])]]

        pred_temp = OrderedDict()
        for role_name in role_names:
          if role_name not in temp:
            pred_temp[role_name] = []
          else:
            pred_temp[role_name] = temp[role_name]

        doc["pred_templates"] = [pred_temp]

        if role_names[0] == "Status":
            docs[ex["doc_key"].split("-")[-1]] = doc
        else:
            docs[ex["doc_key"]] = doc

    input_file.close()
    output_file = open(args.output_file, "w")

    output_file.write(json.dumps(docs, indent=4))

    output_file.close()
