import json, argparse, copy
from collections import OrderedDict

# Used for creating the heuristic for clustering the DyGIE++ output to templates

role_names = ["Status", "Country", "Disease", "Victims"]

def create_new_template(init_val):
    temp = OrderedDict()

    for role_name in role_names:
      temp[role_name] = copy.deepcopy(init_val)

    return temp

def is_empty_template(temp):
    for role_name in temp:
        if temp[role_name] != []:
            return False
    return True

def remove_dup_template(templates):
    no_dup = []
    for temp in templates:
        if temp not in no_dup:
            no_dup.append(temp)
    return no_dup

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        default=None,
        type=str,
        required=False,
        help="Dygie output file.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        default=None,
        type=str,
        required=False,
        help="Formatted template file for eval.py.",
    )
    args = parser.parse_args()

    input_file = open(args.input_file, "r")

    docs = {}

    for line in input_file:
        ex = json.loads(line)
        doc = {}

        doc_tokens = ex["sentences"][0]
        doc["pred_templates"] = []

        temp_role_ind = create_new_template(0)
        templates = [create_new_template([]) for i in range(20)]

        for start, end, role_name, _, _ in ex["predicted_ner"][0]:
            if role_name in role_names:
                pred_mention = [" ".join(doc_tokens[start : end + 1])]
                if role_name == "Country" or role_name == "Disease":
                    if len(templates[temp_role_ind[role_name]][role_name]) != 0:
                        found = False
                        pos = -1
                        for i in range(temp_role_ind[role_name]+1):
                            for mention in templates[i][role_name]:
                                if pred_mention[0] in mention[0] or mention[0] in pred_mention[0]:
                                    found = True
                                    pos = i
                        if found:
                            templates[pos][role_name].append(pred_mention)
                        else:
                            temp_role_ind[role_name] += 1
                            templates[temp_role_ind[role_name]][role_name].append(pred_mention)
                    else:
                        templates[temp_role_ind[role_name]][role_name].append(pred_mention)
                else:
                    templates[temp_role_ind[role_name]][role_name].append(pred_mention)

        doc["pred_templates"] = [pred_temp for pred_temp in templates if not is_empty_template(pred_temp)]

        docs[ex["doc_key"].split("-")[-1]] = doc

    input_file.close()
    output_file = open(args.output_file, "w")

    output_file.write(json.dumps(docs, indent=4))

    output_file.close()