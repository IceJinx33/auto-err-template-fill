import json
import re
import string
import copy
import argparse
import textwrap
from tqdm import tqdm
import numpy as np
import spacy
import Error_Analysis

nlp = spacy.load("en_core_web_sm")

# List of names of roles (keys for rows in each template)
role_names = ["PerpInd", "PerpOrg", "Target", "Weapon", "Victim"]

class TemplateTransformation:

    translation = {
        "Span_Error": "Alter_Span",
        "Spurious_Role_Filler": "Remove_Role_Filler",
        "Missing_Role_Filler": "Introduce_Role_Filler",
        "Spurious_Template": "Remove_Template",
        "Missing_Template": "Introduce_Template",
        "Incorrect_Role": "Change_Role",
    }

    def __init__(self, template1, template2):
        assert template1 is not None or template2 is not None
        self.predicted = template1
        self.gold = template2
        self.template_transformations = []
        self.role_transformations = {}

        self.transformed_template = None

        if self.predicted is None:
            self.template_transformations += ["Introduce_Template"]

            roles = {}
            for role in self.gold.roles:
                mentions = []
                for entity in self.gold.roles[role].mentions:
                    mentions.append(entity.mentions[0])
                roles[role] = Role(self.gold.doc_id, mentions, False)
            self.transformed_template = Template(self.gold.doc_id, self.gold.incident_type, roles, False)
            
        elif self.gold is None:
            self.template_transformations += ["Remove_Template"]
            self.transformed_template = None
        else:
            errors = Template.compare(self.predicted, self.gold).error
            for error in list(errors):
                if error in self.translation:
                    errors[self.translation[error]] = errors[error]
                    errors.pop(error)
            self.role_transformations = invert_dict(errors)
            
            self.transformed_template = copy.deepcopy(self.predicted)
            roles = {}
            for role in self.transformed_template.roles:
                if role in self.role_transformations:
                    mentions = []
                    for entity in self.gold.roles[role].mentions:
                        mentions.append(entity.mentions[0])
                    self.transformed_template.roles[role] = Role(self.gold.doc_id, mentions, False)
                else:
                    self.transformed_template.roles[role].gold = False

    def __str__(self):
        result = ""
        for transformation in self.template_transformations:
            if "Introduce_Template" == transformation:
                result += "[Introduce template]: \n"
                result += str(self.gold)
                result += '\n'
            if "Remove_Template" == transformation:
                result += "[Remove template]: \n"
                result += str(self.predicted)
                result += '\n'
        for role in self.role_transformations.keys():
            result += role + ': ' + Mentions.str_from_doc(self.predicted.roles[role]) + "===" + str(
                self.role_transformations[role]) + "===>" + \
                      Mentions.str_from_doc(self.gold.roles[role]) + '\n'
        return result

def transform(predicted_summary, gold_summary, best_matching):
    output_file.write("----------\nDoc ID: " + str(predicted_summary.doc_id) + " Transformations:\n")

    transformed_templates = []
    for i, j in best_matching["pairs"]:
        transformation = TemplateTransformation(predicted_summary.templates[i],
                                     gold_summary.templates[j])
        output_file.write(str(transformation) + "\n")
        transformed_templates.append(transformation.transformed_template)

    for i in best_matching["unmatched_predicted"]:
        transformation = TemplateTransformation(predicted_summary.templates[i], None)
        output_file.write(str(transformation) + "\n")

    for j in best_matching["unmatched_gold"]:
        transformation = TemplateTransformation(None, gold_summary.templates[j])
        output_file.write(str(transformation) + "\n")
        transformed_templates.append(transformation.transformed_template)

    transformed_pred_summary = Summary(predicted_summary.doc_id, transformed_templates, False)
    transformed_data.append((transformed_pred_summary, gold_summary))



class MUC_Result(Error_Analysis.Result):

    # List of names of error types
    error_names = ["Span_Error", "Spurious_Role_Filler", "Missing_Role_Filler",
          "Spurious_Template", "Missing_Template", "Incorrect_Role"]
    log = ""

    def __init__(self):
        self.stats = {}
        for key in role_names + ["total"]:
            self.stats[key] = {"p_num": 0, "p_den": 0, "r_num": 0, "r_den": 0, "p": 0, "r": 0, "f1": 0}
        self.error = {}
        for key in self.error_names:
            self.error[key] = []
        self.confusion_matrices = []
        self.spans = []

    def __str__(self, verbose=True):
        re = "Result:\n\n"
        for key in ["total"] + roles:
            re += key + ": f1: {0:.4f}, precision:{1:.4f}, recall: {2:.4f}\n".format(self.values[key]["f1"],
                                                                                     self.values[key]["p"],
                                                                                     self.values[key]["r"])
            if verbose: re += " p_num:" + str(self.values[key]["p_num"]) + " p_den:" + str(self.values[key]["p_den"]) + \
                              " r_num:" + str(self.values[key]["r_num"]) + " r_den:" + str(
                self.values[key]["r_den"]) + "\n"
        re += "\n"
        for key in errors:
            re += key + ": " + str(len(self.error[key])) + "\n"
        return re

    def __eq__(self, other):
        pass

    def __lt__(self, other):
        #  is None or (result is not None and (result.score() > best_score or (result.score() == best_score and len(result.error["Span_Error"]) > best_spans))
        pass 

    @staticmethod
    def combine(result1, result2):
        if result1 is None or result2 is None: return None
        result = Result()
        for key in result.values.keys():
            for stat in ["p_num", "p_den", "r_num", "r_den"]:
                result.values[key][stat] = result1.values[key][stat] + result2.values[key][stat]
        for key in result.error.keys():
            result.error[key] = result1.error[key] + result2.error[key]
        result.confusion_matrices = result1.confusion_matrices + result2.confusion_matrices
        result.spans = result1.spans + result2.spans
        return result

    @staticmethod
    def compute_scores(p_num, p_den, r_num, r_den, beta=1):
        p = 0 if p_den == 0 else p_num / float(p_den)
        r = 0 if r_den == 0 else r_num / float(r_den)
        d = beta * beta * p + r
        f1 = 0 if d == 0 else (1 + beta * beta) * p * r / d
        return (p, r, f1)

    def update_stats(self):
        for _, role in self.values.items():
            role["p"], role["r"], role["f1"] = Result.compute_scores(role["p_num"], role["p_den"], role["r_num"],
                                                                     role["r_den"])
        return

    def score(self):
        if self is None: return 0
        self.update_stats()
        return self.values["total"]["f1"]

    def update(self, comparison_event, args = {}):
        if comparison_event == "Spurious_Template":
            self.values["total"]["p_den"] += 1  # for the incident_type
            self.error["Spurious_Template"].append(args["predicted_template"])
        elif comparison_event == "Matched_Role_Filler":
            correct = False
            span_error = False
            min_span_diff = np.infty
            best_gold_mention = None
            for gold_mention in gold_mentions.mentions:
                lower = max(predicted_mention.span[0], gold_mention.span[0])
                upper = min(predicted_mention.span[1], gold_mention.span[1])
                if lower <= upper:
                    if gold_mention.span == predicted_mention.span:
                        correct = True
                    else:
                        span_error = True
                        diff = (abs(predicted_mention.span[0] - gold_mention.span[0]) + 
                        abs(predicted_mention.span[1] - gold_mention.span[1]))
                        if diff < min_span_diff:
                            min_span_diff = diff
                            best_gold_mention = gold_mention

            if correct:
                result.values[role_name]["r_num"] += 1
                result.values[role_name]["p_num"] += 1
                result.values["total"]["r_num"] += 1
                result.values["total"]["p_num"] += 1
            elif span_error:
                result.error["Span_Error"].append(role_name)
                
                # extracting missing/extra parts of the spans that cause span errors
                # m - missing, e - extra
                if best_gold_mention != None:
                    diff_1 = predicted_mention.span[0] - best_gold_mention.span[0]
                    diff_2 = predicted_mention.span[1] - best_gold_mention.span[1]
                    docid_str = "Doc ID: " + predicted_mention.doc_id
                    if diff_1 > 0:
                        chars = extract_span_diff(best_gold_mention.literal, diff_1, True)
                        result.spans.append((docid_str, chars, "m"))
                    elif diff_1 < 0:
                        chars = extract_span_diff(predicted_mention.literal, -diff_1, True)
                        result.spans.append((docid_str, chars, "e"))
                    else:
                        pass
                    if diff_2 > 0:
                        chars = extract_span_diff(predicted_mention.literal, diff_2, False)
                        result.spans.append((docid_str, chars, "e"))
                    elif diff_2 < 0:
                        chars = extract_span_diff(best_gold_mention.literal, -diff_2, False)
                        result.spans.append((docid_str, chars, "m"))
                    else:
                        pass

            else:
                result.error["Missing_Role_Filler"].append(role_name)
                result.error["Spurious_Role_Filler"].append(role_name)


def normalize_string(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    s = re.sub(regex, ' ', s.lower())
    return ' '.join([c for c in s if c.isalnum()])

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
        if (doc[i: i + len(mention)] == mention):
            start = i
            end = i + len(mention) - 1
            break
    if start == -1 and end == -1:
        return 1, 0
    return start, end

def add_script_args(parser):
    parser.add_argument("-i", "--input_file", type=str,
                    help="The path to the input file given to the system")
    parser.add_argument("-v", "--verbose", action="store_true",
                    help="Increase output verbosity")
    parser.add_argument("-at", "--analyze_transformed", action="store_true",
                    help="Analyze transformed data")
    parser.add_argument("-s", "--scoring_mode", type=str, choices=["all", "msp", "mmi", "mat"],
                    help= textwrap.dedent('''\
                        Choose scoring mode according to MUC:
                        all - All Templates
                        msp - Matched/Spurious
                        mmi - Matched/Missing
                        mat - Matched Only
                    '''), 
                    default="All_Templates")
    parser.add_argument("-o", "--output_file", type=str,
                    help="The path to the output file the system writes to")
    return parser

def from_file(input_file, result_type):
    """
    This function returns the data structure and tokenized documents
    for error analysis given the input file [input_file].
    The data structure is a List of tuples, each tuple containing 2 Summary
    objects for a document, the first Summary object contains the predicted
    templates, the second contains the gold templates.
    The tokenized documents consists of a dictionary with keys as doc ids
    and respective tokenized documents as values.
    :params input_file: valid path to input file
    :type input_file: string
    """
    data = []
    documents = {}

    with open(input_file, encoding="utf-8") as f:
        inp_dict = json.load(f)

    for docid in inp_dict:

        pred_templates = []
        gold_templates = []

        example = inp_dict[docid]
        doc_tokens = normalize_string(example["doctext"].replace(" ##", "")).split()
        documents[docid] = doc_tokens

        for pred_temp in example["pred_templates"]:
            roles = {}
            for role in pred_temp:
                if role == "incident_type":
                    incident = pred_temp["incident_type"]
                    continue
                mentions = []
                for entity in pred_temp[role]:
                    for mention in entity:
                        mention_tokens = normalize_string(mention).split()
                        span = mention_tokens_index(doc_tokens, mention_tokens)
                        mentions.append(Error_Analysis.Mention(docid, span, mention, result_type))
                roles[role] = Error_Analysis.Role(docid, mentions, False, result_type)
            pred_templates.append(Error_Analysis.Template(docid, incident, roles, False, result_type))

        for gold_temp in example["gold_templates"]:
            roles = {}
            for role in gold_temp:
                if role == "incident_type":
                    incident = gold_temp["incident_type"]
                    continue
                coref_mentions = []
                for entity in gold_temp[role]:
                    mentions = []
                    for mention in entity:
                        mention_tokens = normalize_string(mention).split()
                        span = mention_tokens_index(doc_tokens, mention_tokens)
                        mentions.append(Error_Analysis.Mention(docid, span, mention, result_type))
                    coref_mentions.append(Error_Analysis.Mentions(docid, mentions, result_type))
                roles[role] = Error_Analysis.Role(docid, coref_mentions, True, result_type)
            gold_templates.append(Error_Analysis.Template(docid, incident, roles, True, result_type))

        pred_summary = Error_Analysis.Summary(docid, pred_templates, False, result_type)
        gold_summary = Error_Analysis.Summary(docid, gold_templates, True, result_type)

        data.append((pred_summary, gold_summary))

    return data, documents

def analyze(predicted_summary, gold_summary, verbose):
    output_file.write("Comparing Prediction:\n")
    output_file.write(str(predicted_summary) + "\n")
    output_file.write("\nTo Gold:\n")
    output_file.write(str(gold_summary) + "\n\n")
    return Error_Analysis.Summary.compare(predicted_summary, gold_summary, verbose)

if __name__ == "__main__":
    parser = add_script_args(argparse.ArgumentParser(usage=
    'Use "python MUC_Error_Analysis_Operation.py --help" for more information',
        formatter_class=argparse.RawTextHelpFormatter))
    args = parser.parse_args()

    input_file = args.input_file
    verbose = args.verbose
    analyze_transformed = args.analyze_transformed

    output_file = open(args.output_file, "w")

    if args.scoring_mode == "all":
        output_file.write("\nUsing scoring mode - All Templates\n")
        scoring_mode = "All_Templates"
    elif args.scoring_mode == "msp": 
        output_file.write("\nUsing scoring mode - Matched/Spurious\n")
        scoring_mode = "Matched/Spurious"
    elif args.scoring_mode == "mmi":
        output_file.write("\nUsing scoring mode - Matched/Missing\n")
        scoring_mode = "Matched/Missing"
    elif args.scoring_mode == "mat":
        output_file.write("\nUsing scoring mode - Matched Only\n")
        scoring_mode = "Matched_Only"
    else:
        output_file.write("\nUsing default scoring mode - All Templates\n")
        scoring_mode = "All_Templates"

    data, docs = from_file(input_file, MUC_Result)

    transformed_data = []

    output_file.write("\nANALYZING DATA AND APPLYING TRANSFORMATIONS ...\n")

    total_result_before = MUC_Result()

    for pair in tqdm(data, desc="Analyzing Data and Applying Transformations: "):
        output_file.write("\n-----------------------------------\n")
        best_matching, best_res = analyze(*pair, verbose)
        total_result_before = MUC_Result.combine(total_result_before, best_res)
        output_file.write("\n")
        transform(*pair, best_matching)
        output_file.write("\n-----------------------------------\n")

    if analyze_transformed:
        output_file.write("ANALYZING TRANSFORMED DATA ...\n")

        total_result_after = MUC_Result()

        for pair in tqdm(transformed_data, desc="Analyzing Transformed Data: "):
            output_file.write("\n-----------------------------------\n")
            _, best_res = analyze(*pair, verbose)
            total_result_after = MUC_Result.combine(total_result_after, best_res)
            output_file.write("\n-----------------------------------\n")
    
    total_result_before.update()
    output_file.write("\n************************************\nTotal Result Before Transformation : \n************************************\n\n" + 
    str(total_result_before) + "\n")
    
    if analyze_transformed:
        total_result_after.update()
        output_file.write("\n***********************************\nTotal Result After Transformation : \n***********************************\n\n" + 
        str(total_result_after) + "\n")
    
    output_file.close()

    # Giving POS tags to Missing/Extra Span tokens
    missing_span = {}
    extra_span = {}
    for _, span, me in total_result_before.spans:
        st = nlp(span)
        for token in st:
            pos = token.pos_
            print((token, pos))
            if me == "m":
                try:
                    missing_span[pos] += 1
                except:
                    missing_span[pos] = 1
            else:
                try:
                    extra_span[pos] += 1
                except:
                    extra_span[pos] = 1
    
    ex = nlp("Shining Path")
    for token in ex:
        pos = token.pos_
        print((token, pos))

    print("Missing span tokens - POS counts \n" + str(missing_span) + "\n")
    print("Extra span tokens - POS counts \n" + str(extra_span))
