import argparse
import json
import re
import textwrap

import spacy
from tqdm import tqdm

import Error_Analysis
from Error_Analysis import *

nlp = spacy.load("en_core_web_sm")

# List of names of roles (keys for rows in each template)
role_names = ["incident_type", "PerpInd", "PerpOrg", "Target", "Weapon", "Victim"]

"""
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
            result += role + ': ' + Error_Analysis.Mentions.str_from_doc(self.predicted.roles[role]) + "===" + str(
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
"""


class MUC_Result(Error_Analysis.Result):
    # List of names of error types
    error_names = ["Span_Error", "Spurious_Role_Filler", "Missing_Role_Filler",
                   "Spurious_Template", "Missing_Template", "Incorrect_Role"]

    def __init__(self):
        self.valid = True

        self.stats = {}
        for role_name in role_names + ["total"]:
            self.stats[role_name] = {"p_num": 0, "p_den": 0, "r_num": 0, "r_den": 0, "p": 0, "r": 0, "f1": 0}

        self.errors = {}
        for error_name in self.error_names:
            self.errors[error_name] = []

        self.transformations = []
        self.role_confusion_matrices = []
        self.spans = []
        self.span_error = 0
        self.log = ""

    def write_log(self, s):
        self.log += "\n" + s

    def __str__(self, verbose=True):
        output_string = "Result:"
        if verbose: output_string += "\n"+self.log
        for role_name in ["total"] + role_names:
            output_string += "\n"+ role_name + ": f1: {0:.4f}, precision:{1:.4f}, recall: {2:.4f}".format(
                self.stats[role_name]["f1"],
                self.stats[role_name]["p"],
                self.stats[role_name]["r"])
            if verbose: output_string += "\np_num:" + str(self.stats[role_name]["p_num"]) + " p_den:" + str(
                self.stats[role_name]["p_den"]) + \
                                         " r_num:" + str(self.stats[role_name]["r_num"]) + " r_den:" + str(
                self.stats[role_name]["r_den"])
        # output_string += "\n"
        # for error_name in self.error_names:
        #     output_string += error_name + ": " + str(len(self.error[error_name])) + "\n"
        return output_string

    def __gt__(self, other):
        self.update_stats()
        other.update_stats()
        return not other.valid or (self.stats["total"]["f1"] > other.stats["total"]["f1"]) or \
               (self.stats["total"]["f1"] == other.stats["total"]["f1"] and self.span_error < other.span_error)

    @staticmethod
    def combine(result1, result2):
        result = MUC_Result()
        result.valid = result1.valid and result2.valid
        for key in result.stats.keys():
            for stat in ["p_num", "p_den", "r_num", "r_den"]:
                result.stats[key][stat] = result1.stats[key][stat] + result2.stats[key][stat]
        for key in result.errors.keys():
            result.errors[key] = result1.errors[key] + result2.errors[key]
        result.transformations = result1.transformations + result2.transformations
        result.role_confusion_matrices = result1.role_confusion_matrices + result2.role_confusion_matrices
        result.spans = result1.spans + result2.spans
        result.span_error = result1.span_error + result2.span_error
        result.log = result1.log + "\n" + result2.log
        return result

    @staticmethod
    def compute_scores(p_num, p_den, r_num, r_den, beta=1):
        p = 0 if p_den == 0 else p_num / float(p_den)
        r = 0 if r_den == 0 else r_num / float(r_den)
        d = beta * beta * p + r
        f1 = 0 if d == 0 else (1 + beta * beta) * p * r / d
        return (p, r, f1)

    def update_stats(self):
        for _, role in self.stats.items():
            role["p"], role["r"], role["f1"] = MUC_Result.compute_scores(role["p_num"], role["p_den"], role["r_num"],
                                                                         role["r_den"])
        return

    @staticmethod
    def span_scorer(span1, span2, mode="absolute"):
        # Lower is better - 0 iff exact match, 1 iff no intersection, otherwise between 0 and 1
        length1, length2 = abs(span1[1] - span1[0]), abs(span2[1] - span2[0])
        if mode == "absolute":
            val = (abs(span1[0] - span2[0]) + abs(span1[1] - span2[1])) / (length1 + length2)
            return val if val < 1 else 1
        elif mode == "geometric_mean":
            intersection = max(0, min(span1[1], span2[1]) - max(span1[0], span2[0]))
            return 1 - ((length1 * length2 / (intersection ** 2)) if intersection > 0 else 0)

    def update(self, comparison_event, args=None):

        if args is None: args = {}

        self.write_log(".")
        if not self.valid:
            self.write_log("Invalid matching.")
            return
            
        self.write_log(comparison_event)
        for k, v in args.items():
            if "Template" not in comparison_event:
                self.write_log(k+": "+str(v))
            else: self.write_log(str(v))

        if comparison_event == "Spurious_Role_Filler":
            self.stats[args["role_name"]]["p_den"] += 1
            self.stats["total"]["p_den"] += 1
            self.errors["Spurious_Role_Filler"].append(args["predicted_mention"])

        elif comparison_event == "Missing_Role_Filler":
            self.stats[args["role_name"]]["r_den"] += 1
            self.stats["total"]["r_den"] += 1
            self.errors["Missing_Role_Filler"].append(args["gold_mentions"])

        elif comparison_event == "Matched_Role_Filler":
            min_span_error = 1
            best_gold_mention = None
            predicted_mention = args["predicted_mention"]
            for gold_mention in args["gold_mentions"].mentions:
                span_error = MUC_Result.span_scorer(predicted_mention.span, gold_mention.span)
                if span_error < min_span_error:
                    min_span_error = span_error
                    best_gold_mention = gold_mention

            self.stats[args["role_name"]]["r_den"] += 1
            self.stats[args["role_name"]]["p_den"] += 1
            self.stats["total"]["r_den"] += 1
            self.stats["total"]["p_den"] += 1
            if min_span_error == 0:
                self.stats[args["role_name"]]["r_num"] += 1
                self.stats[args["role_name"]]["p_num"] += 1
                self.stats["total"]["r_num"] += 1
                self.stats["total"]["p_num"] += 1
            elif min_span_error == 1:
                self.errors["Missing_Role_Filler"].append(args["role_name"])
                self.errors["Spurious_Role_Filler"].append(args["role_name"])
            else:
                self.span_error += min_span_error
                self.errors["Span_Error"].append(args["role_name"])
                self.spans += Error_Analysis.extract_span(predicted_mention, best_gold_mention)

        elif comparison_event == "Spurious_Template":
            self.errors["Spurious_Template"].append(args["predicted_template"])

        elif comparison_event == "Missing_Template":
            self.errors["Missing_Template"].append(args["gold_template"])

        elif comparison_event == "Matched_Template":
            if args["predicted_template"].roles["incident_type"] != args["gold_template"].roles["incident_type"]:
                self.valid = False

        elif comparison_event == "Incorrect_Role":
            pass

        else:
            raise Exception("Illegal comparison event: " + comparison_event)


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
                        help=textwrap.dedent('''\
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

    for docid, example in inp_dict.items():
        pred_templates = []
        gold_templates = []

        doc_tokens = normalize_string(example["doctext"].replace(" ##", "")).split()
        documents[docid] = doc_tokens

        for pred_temp in example["pred_templates"]:
            roles = {}
            for role_name, role_data in pred_temp.items():
                if role_name == "incident_type":
                    mention = Error_Analysis.Mention(docid, (1,0), role_data, result_type)
                    roles[role_name] = Error_Analysis.Role(docid, [mention], False, result_type)
                    continue
                mentions = []
                for entity in role_data:
                    for mention in entity:
                        mention_tokens = normalize_string(mention).split()
                        span = mention_tokens_index(doc_tokens, mention_tokens)
                        mentions.append(Error_Analysis.Mention(docid, span, mention, result_type))
                roles[role_name] = Error_Analysis.Role(docid, mentions, False, result_type)
            pred_templates.append(Error_Analysis.Template(docid, roles, False, result_type))

        for gold_temp in example["gold_templates"]:
            roles = {}
            for role_name, role_data in gold_temp.items():
                if role_name == "incident_type":
                    mention = Error_Analysis.Mention(docid, (1,0), role_data, result_type)
                    mentions = Error_Analysis.Mentions(docid, [mention], result_type)
                    roles[role_name] = Error_Analysis.Role(docid, [mentions], True, result_type)
                    continue
                coref_mentions = []
                for entity in role_data:
                    mentions = []
                    for mention in entity:
                        mention_tokens = normalize_string(mention).split()
                        span = mention_tokens_index(doc_tokens, mention_tokens)
                        mentions.append(Error_Analysis.Mention(docid, span, mention, result_type))
                    coref_mentions.append(Error_Analysis.Mentions(docid, mentions, result_type))
                roles[role_name] = Error_Analysis.Role(docid, coref_mentions, True, result_type)
            gold_templates.append(Error_Analysis.Template(docid, roles, True, result_type))

        pred_summary = Error_Analysis.Summary(docid, pred_templates, False, result_type)
        gold_summary = Error_Analysis.Summary(docid, gold_templates, True, result_type)

        data.append((pred_summary, gold_summary))

    return data, documents


def analyze(predicted_summary, gold_summary, verbose):
    output_file.write("Comparing:\n"+str(predicted_summary)+"\n"+str(gold_summary)+"\n\n\t--\n\n")
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
        output_file.write("\n\n\t---\n\n")
        result = analyze(*pair, verbose)
        output_file.write(str(result))
        total_result_before = MUC_Result.combine(total_result_before, result)
        # output_file.write("\n")
        # transform(*pair, best_matching)
        # output_file.write("\n-----------------------------------\n")

    # if analyze_transformed:
    #     output_file.write("ANALYZING TRANSFORMED DATA ...\n")

    #     total_result_after = MUC_Result()

    #     for pair in tqdm(transformed_data, desc="Analyzing Transformed Data: "):
    #         output_file.write("\n-----------------------------------\n")
    #         _, best_res = analyze(*pair, verbose)
    #         total_result_after = MUC_Result.combine(total_result_after, best_res)
    #         output_file.write("\n-----------------------------------\n")

    # total_result_before.update()
    # output_file.write(
    #     "\n************************************\nTotal Result Before Transformation : \n************************************\n\n" +
    #     str(total_result_before) + "\n")

    # if analyze_transformed:
    #     total_result_after.update()
    #     output_file.write(
    #         "\n***********************************\nTotal Result After Transformation : \n***********************************\n\n" +
    #         str(total_result_after) + "\n")

    # output_file.close()

    """
    # Giving POS tags to Missing/Extra Span tokens
    missing_span = {}
    extra_span = {}
    for _, span, me in total_result_before.spans:
        st = nlp(span)
        for token in st:
            pos = token.pos_
            # print((token, pos))
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
        # print((token, pos))
    """
    # print("Missing span tokens - POS counts \n" + str(missing_span) + "\n")
    # print("Extra span tokens - POS counts \n" + str(extra_span))
