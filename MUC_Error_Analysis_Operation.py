import json
import re
import string
import copy
import argparse
import textwrap
from tqdm import tqdm
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

# List of names of roles (keys for rows in each template)
roles = ["PerpInd", "PerpOrg", "Target", "Weapon", "Victim"]
# List of names of error types
errors = ["Span_Error", "Spurious_Role_Filler", "Missing_Role_Filler",
          "Spurious_Template", "Missing_Template", "Incorrect_Role"]

def all_matchings(a, b):
    """returns a list of all matchings, where each matching is a dictionary with keys "pairs," "unmatched_gold," "unmatched_predicted." 
    A matching is a set of pairs (i,j) where i in range(a), j is in range(b), unmatched_predicted is a subset of range(a), 
    unmatched_gold is a subset of range(b), every element of range(a) occurs exactly once in unmatched_predicted 
    or in the first position of a pair, and every element of range(b) occurs exactly once in unmatched_goldd 
    or in the second position of a pair."""
    
    matchings = [{"pairs": [], "unmatched_gold": list(range(b)), "unmatched_predicted": list(range(a))}]
    for i in range(a):  # number of input indices paired
        new_matchings = []
        for matching in matchings:
            for unmatched in matching["unmatched_gold"]:
                new_matchings += [{"pairs": matching["pairs"] + [(i, unmatched)],
                                   "unmatched_gold": [item for item in matching["unmatched_gold"] if item != unmatched],
                                   "unmatched_predicted": [item for item in matching["unmatched_predicted"] if
                                                           item != i]}]
        matchings = matchings + new_matchings
    return matchings


def normalize_string_old(s, for_doc=False):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_hyphens(text):
        regex = re.compile(r'( *-+)+ *', re.UNICODE)
        return re.sub(regex, '', text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(remove_hyphens(lower(s)))))  


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


def from_file(input_file):
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
                        mentions.append(Mention(docid, span, mention))
                roles[role] = Role(docid, mentions, False)
            pred_templates.append(Template(docid, incident, roles, False))

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
                        mentions.append(Mention(docid, span, mention))
                    coref_mentions.append(Mentions(docid, mentions))
                roles[role] = Role(docid, coref_mentions, True)
            gold_templates.append(Template(docid, incident, roles, True))

        pred_summary = Summary(docid, pred_templates, False)
        gold_summary = Summary(docid, gold_templates, True)

        data.append((pred_summary, gold_summary))

    return data, documents


def invert_dict(d):
    # Credit https://stackoverflow.com/questions/35491223/inverting-a-dictionary-with-list-values
    inverse = dict()
    for key in d:
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse:
                # If not create a new list
                inverse[item] = [key]
            else:
                inverse[item].append(key)
    return inverse

def extract_span_diff(string1, diff, start):
    """
    This functions returns a string containing [diff] number of consecutive 
    alphanumeric characters from [string1] as well as any non-alphanumeric 
    characters it encounters while searching for alphanumeric characters. If 
    [start] = True, extraction starts from the beginning of the string, otherwise,
    extraction begins at the end of the string.
    :params string1: the input string
    :type string1: string
    :params diff: the number of alphanumeric characters to extract
    :type diff: [diff] is an int > 0
    :params start: whether extraction starts at the beginning ([start] = True)
    or end of [string1] ([start] = False)
    :type beg: [start] is an bool
    """
    if start == False:
        string1 = string1[::-1]
    d = 0
    s = ""
    for c in string1:
        s += c
        if c.isalnum():
            d += 1
        else:
            continue
        if d == diff:
            break
    if start == False:
        return s[::-1]
    else:
        return s

# A single mention
class Mention:
    def __init__(self, doc_id, span, literal):
        self.doc_id = doc_id
        self.span = span  # a pair of indices delimiting a string in the doc
        self.literal = literal

    @staticmethod
    def compare(predicted_mention, gold_mentions,
                role):  # predicted_mention is of type Mention, gold_mentions is of type Mentions
        result = Result()

        if gold_mentions is None:
            result.values[role]["p_den"] += 1
            result.values["total"]["p_den"] += 1
            result.error["Spurious_Role_Filler"].append(role)
            return result

        if predicted_mention is None:
            result.values[role]["r_den"] += 1
            result.values["total"]["r_den"] += 1
            result.error["Missing_Role_Filler"].append(role)
            return result

        result.values[role]["r_den"] += 1
        result.values[role]["p_den"] += 1
        result.values["total"]["r_den"] += 1
        result.values["total"]["p_den"] += 1
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
            result.values[role]["r_num"] += 1
            result.values[role]["p_num"] += 1
            result.values["total"]["r_num"] += 1
            result.values["total"]["p_num"] += 1
        elif span_error:
            result.error["Span_Error"].append(role)
            
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
            result.error["Missing_Role_Filler"].append(role)
            result.error["Spurious_Role_Filler"].append(role)

        return result

    def from_doc(self):
        # doc_tokens = docs[self.doc_id]  # global variable docs - tokenized documents
        # start, end = self.span
        #return ' '.join(doc_tokens[start:end + 1])
        return self.literal

    def __str__(self):
        re = str(self.span)
        re += " - " + self.from_doc()
        return re


# An entity in a gold template, with a list of coref mentions
class Mentions:
    def __init__(self, doc_id, mentions):
        self.doc_id = doc_id
        self.mentions = mentions  # a list of coref mentions
        assert all(
            mention.doc_id == self.doc_id for mention in mentions), "only mentions for the same doc can form a list"

    def __str__(self):
        return "[" + ", ".join([str(mention) for mention in self.mentions]) + "]"

    @staticmethod
    def str_from_doc(r):
        result = ""
        for mention_or_mentions in r.mentions:
            if isinstance(mention_or_mentions, Mention):
                result+='('
                mention = mention_or_mentions
                result += mention.from_doc()
                if mention != r.mentions[-1]:
                    result += "), "
                else:
                    result += ")"
            else:
                result += "("
                for mention in mention_or_mentions.mentions:
                    result += mention.from_doc()
                    if mention != mention_or_mentions.mentions[-1]:
                        result += ", "
                result += ")"
        return result


# The data associated with a field in a template
class Role:
    def __init__(self, doc_id, mentions, gold):
        self.doc_id = doc_id
        self.mentions = mentions  # a list of Mention (not Gold) or Mentions (gold)
        self.gold = gold
        assert all(
            mention.doc_id == self.doc_id for mention in self.mentions), "mentions must be for the same doc as its role"

    def __str__(self, outer=True):
        return "[" + ", ".join([str(mention) for mention in self.mentions]) + "]"

    @staticmethod
    def compare_matching(matching, predicted_role, gold_role, role, verbose=False):
        result = Result()
        for i, j in matching["pairs"]:
            if verbose: 
                output_file.write(" - " + str(predicted_role.mentions[i]) + " -- matched with -- " + str(gold_role.mentions[j]) + "\n")
            result = Result.combine(result, Mention.compare(predicted_role.mentions[i], gold_role.mentions[j], role))
        for i in matching["unmatched_predicted"]:
            if verbose: 
                output_file.write(" - Spurious Role Filler:" + str(predicted_role.mentions[i]) + "\n")
            result = Result.combine(result, Mention.compare(predicted_role.mentions[i], None, role))
        for i in matching["unmatched_gold"]:
            if verbose: 
                output_file.write(" - Missing Role Filler:" + str(gold_role.mentions[i]) + "\n")
            result = Result.combine(result, Mention.compare(None, gold_role.mentions[i], role))
        return result

    @staticmethod
    def compare(predicted_role, gold_role, role, verbose=False):
        if gold_role is None:
            result = Result()
            for mention in predicted_role.mentions:
                result = Result.combine(result, Mention.compare(mention, None, role))
            return result

        if predicted_role is None:
            result = Result()
            for mentions in gold_role.mentions:
                result = Result.combine(result, Mention.compare(None, mentions, role))
            return result

        best_score = 0
        best_spans = 0
        best_result = None
        best_matching = None
        for matching in all_matchings(len(predicted_role.mentions), len(gold_role.mentions)):
            result = Role.compare_matching(matching, predicted_role, gold_role, role)
            if best_result is None or (result is not None and (result.score() > best_score or (
                    result.score() == best_score and len(result.error["Span_Error"]) > best_spans))):
                best_result = result
                best_score = result.score()
                best_spans = len(result.error["Span_Error"])
                best_matching = matching
        if verbose: Role.compare_matching(best_matching, predicted_role, gold_role, role, True)

        return best_result


# A data structure containing structured information about an event in a document
class Template:
    def __init__(self, doc_id, incident_type, roles, gold):
        self.doc_id = doc_id
        self.incident_type = incident_type  # a string
        self.roles = roles  # a dictionary, indexed by strings in roles, with Role values
        self.gold = gold
        assert all(role.gold == self.gold for _, role in
                   self.roles.items()), "roles must be in the same format as its template"
        assert all(role.doc_id == self.doc_id for _, role in
                   self.roles.items()), "roles must be for the same doc as its template"

    def __str__(self, outer=True):
        re = "Template" + ((" (gold)" if self.gold else " (predicted)") if outer else "") + ":\n"
        if outer: re += "Doc ID: " + str(self.doc_id) + "\n"
        re += "Incident Type: " + str(self.incident_type)
        for role in roles:
            re += "\n" + role + ": " + self.roles[role].__str__(False)
        return re

    @staticmethod
    def compare(predicted_template, gold_template, verbose=False):

        if gold_template is None:
            result = Result()
            result.values["total"]["p_den"] += 1  # for the incident_type
            result.error["Spurious_Template"].append(predicted_template)
            if scoring_mode in ["All_Templates", "Matched/Spurious"]:
                for key, role in predicted_template.roles.items():
                    result = Result.combine(result, Role.compare(role, None, key))
            return result

        if predicted_template is None:
            result = Result()
            result.values["total"]["r_den"] += 1  # for the incident_type
            result.error["Missing_Template"].append(gold_template)
            if scoring_mode in ["All_Templates", "Matched/Missing"]:
                for key, role in gold_template.roles.items():
                    result = Result.combine(result, Role.compare(None, role, key))
            return result

        if predicted_template.incident_type != gold_template.incident_type:
            if verbose: 
                #print("Templates have different incident types")
                output_file.write("Templates have different incident types\n")
            result = Result()
            result = Result.combine(result, Template.compare(predicted_template, None))
            result = Result.combine(result, Template.compare(None, gold_template))
            return result

        result = Result()
        # For getting the incident_type right
        result.values["total"]["p_num"] += 1
        result.values["total"]["p_den"] += 1
        result.values["total"]["r_num"] += 1
        result.values["total"]["r_den"] += 1
        confusion_matrix = []
        for role in roles:
            if verbose: output_file.write("Comparing " + role + ":\n")
            comparison = Role.compare(predicted_template.roles[role], gold_template.roles[role], role, verbose)
            result = Result.combine(result, comparison)
            confusion_row = []
            for other_role in roles:
                confusion_score = Role.compare(predicted_template.roles[role], gold_template.roles[other_role], role, False).score()
                confusion_row.append(confusion_score)
                if confusion_score > comparison.score():
                    result.error["Incorrect_Role"].append(role)
            confusion_matrix.append(confusion_row)
        result.confusion_matrices.append(confusion_matrix)
        return result


# Represents a list of templates extracted from a single document
class Summary:
    def __init__(self, doc_id, templates, gold):
        self.doc_id = doc_id
        self.templates = templates  # a list of Template
        self.gold = gold
        assert all(template.gold == self.gold for template in
                   self.templates), "summary must be in the same format as its templates"
        assert all(template.doc_id == self.doc_id for template in
                   self.templates), "summary must be for the same doc as its templates"

    def __str__(self, outer=True):
        re = "Summary" + ((" (gold)" if self.gold else " (predicted)") if outer else "") + ":\n"
        if outer: re += "Doc ID: " + str(self.doc_id) + "\n--------------------"
        for template in self.templates:
            re += "\n" + template.__str__(False) + "\n--------------------"
        return re

    @staticmethod
    def compare(predicted_summary, gold_summary, verbose=False):
        best_score = 0
        best_spans = 0
        best_result = None
        best_matching = None
        for matching in all_matchings(len(predicted_summary.templates), len(gold_summary.templates)):
            result = Summary.compare_matching(matching, predicted_summary, gold_summary)
            if best_result is None or (result is not None and (result.score() > best_score or (
                    result.score() == best_score and len(result.error["Span_Error"]) > best_spans))):
                best_result = result
                best_score = result.score()
                best_spans = len(result.error["Span_Error"])
                best_matching = matching
        if verbose: Summary.compare_matching(best_matching, predicted_summary, gold_summary, True)
        return best_matching, best_result

    @staticmethod
    def compare_matching(matching, predicted_summary, gold_summary, verbose=False):
        result = Result()
        for i, j in matching["pairs"]:
            if verbose:
                s = "\n" + str(predicted_summary.templates[i]) + "\n" + "  -- matched with --  " + "\n" + str(gold_summary.templates[j]) + "\n\n"
                output_file.write(s)
            result = Result.combine(result, Template.compare(predicted_summary.templates[i], gold_summary.templates[j],
                                                             verbose))
        for i in matching["unmatched_predicted"]:
            if verbose:
                s = "\nSpurious Template:" + "\n" + str(predicted_summary.templates[i]) + "\n\n"
                output_file.write(s)
            result = Result.combine(result, Template.compare(predicted_summary.templates[i], None))
        for i in matching["unmatched_gold"]:
            if verbose:
                s = "\nMissing Template:" + "\n" + str(gold_summary.templates[i]) + "\n\n"
                output_file.write(s)
            result = Result.combine(result, Template.compare(None, gold_summary.templates[i]))
        return result


class Result:
    def __init__(self):
        self.values = {}
        for key in roles + ["total"]:
            self.values[key] = {"p_num": 0, "p_den": 0, "r_num": 0, "r_den": 0, "p": 0, "r": 0, "f1": 0}
        self.error = {}
        for key in errors:
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

    def update(self):
        for _, role in self.values.items():
            role["p"], role["r"], role["f1"] = Result.compute_scores(role["p_num"], role["p_den"], role["r_num"],
                                                                     role["r_den"])
        return

    def score(self):
        if self is None: return 0
        self.update()
        return self.values["total"]["f1"]


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


def analyze(predicted_summary, gold_summary, verbose):
    output_file.write("Comparing Prediction:\n")
    output_file.write(str(predicted_summary) + "\n")
    output_file.write("\nTo Gold:\n")
    output_file.write(str(gold_summary) + "\n\n")
    best_matching, best_res = Summary.compare(predicted_summary, gold_summary, verbose)
    return best_matching, best_res

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

    data, docs = from_file(input_file)

    transformed_data = []

    output_file.write("\nANALYZING DATA AND APPLYING TRANSFORMATIONS ...\n")

    total_result_before = Result()

    for pair in tqdm(data, desc="Analyzing Data and Applying Transformations: "):
        output_file.write("\n-----------------------------------\n")
        best_matching, best_res = analyze(*pair, verbose)
        total_result_before = Result.combine(total_result_before, best_res)
        output_file.write("\n")
        transform(*pair, best_matching)
        output_file.write("\n-----------------------------------\n")

    if analyze_transformed:
        output_file.write("ANALYZING TRANSFORMED DATA ...\n")

        total_result_after = Result()

        for pair in tqdm(transformed_data, desc="Analyzing Transformed Data: "):
            output_file.write("\n-----------------------------------\n")
            _, best_res = analyze(*pair, verbose)
            total_result_after = Result.combine(total_result_after, best_res)
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
