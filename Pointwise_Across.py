import json, re, argparse, textwrap
from tqdm import tqdm

def span_scorer(span1, span2, mode="geometric_mean"):
    # Lower is better - 0 iff exact match, 1 iff no intersection, otherwise between 0 and 1
    if span1 == span2: return 0
    length1, length2 = abs(span1[1] - span1[0]), abs(span2[1] - span2[0])
    if mode == "absolute":
        val = (abs(span1[0] - span2[0]) + abs(span1[1] - span2[1])) / (
            length1 + length2
        )
        return min(val, 1.0)
    elif mode == "geometric_mean":
        intersection = max(0, min(span1[1], span2[1]) - max(span1[0], span2[0]))
        return 1 - (( (intersection ** 2) / (length1 * length2) ) if length1 * length2 > 0 else 0)

def transform(point1, point2, matching, fast = True, inner = False, clean_inner = False):
    assert point1 is not None or point2 is not None, "cannot transform None to None"
    assert None in [point1, point2] or len(point1) == len(point2), "matched points must have the same length"
    if not fast: point1_string, point2_string = (None if point1 is None else "/".join([str(p) for p in point1])), (None if point2 is None else "/".join([str(p) for p in point2]))
    result = Result()
    if (point1 is not None and len(point1) == 1) or (point2 is not None and len(point2) == 1):
        if not fast:
            if point1 is None: 
                result.log += "Add template: "+point2_string#point2_string[:-2]+"P-"+point2_string[-2:]+" <-> "+point2_string
                result.errors["Missing_Template"] += [point2_string]
            elif point2 is None: 
                result.log += "Remove template: "+point1_string
                result.errors["Spurious_Template"] += [point1_string]
            else: result.log += point1_string+" <-> "+point2_string
    else:
        if point1 is not None and point1[1] not in result.stats:
            result.stats[point1[1]] = {
                "p_num": 0,
                "p_den": 0,
                "r_num": 0,
                "r_den": 0,
                "p": 0,
                "r": 0,
                "f1": 0,
            }
        if point2 is not None and point2[1] not in result.stats:
            result.stats[point2[1]] = {
                "p_num": 0,
                "p_den": 0,
                "r_num": 0,
                "r_den": 0,
                "p": 0,
                "r": 0,
                "f1": 0,
            }
        if point1 is None: 
            if not fast: result.log += "Add item: "+point2_string
            result.stats[point2[1]]["r_den"] += 1
            result.stats["total"]["r_den"] += 1
            if not fast: result.errors["Missing_Role_Filler"] += [point2_string]
            result.error += 1
        elif point2 is None: 
            if not fast: result.log += "Remove item: "+point1_string
            result.stats[point1[1]]["p_den"] += 1
            result.stats["total"]["p_den"] += 1
            if not fast: result.errors["Spurious_Role_Filler"] += [point1_string]
            result.error += 1
        else:
            if not inner:
                result.stats[point1[1]]["p_den"] += 1
                result.stats[point1[1]]["r_den"] += 1
                result.stats["total"]["p_den"] += 1
                result.stats["total"]["r_den"] += 1

            if point1[0] != point2[0]:
                matched = ([point1[0]],[point2[0]]) in matching
                if not matched:
                    if not fast: result.errors["Incorrect_Template"] += [point1_string]
                    result.error += 1
                    if not fast: result.log += ("\n" if inner else "")+point1_string+" -> (change template)"
                elif not fast: result.log += ("\n" if inner else "")+point1_string+" -> (update template)"
                result = Result.combine(result, transform([point2[0]]+point1[1:], point2, matching, fast, inner = True, clean_inner = matched), close = True)
            elif point1[1] != point2[1]:
                if not fast: result.log += ("\n" if inner else "")+point1_string+" => (change role)"
                if not fast: result.errors["Incorrect_Role"] += [point1_string]
                result.error += 1
                result = Result.combine(result, transform([point1[0], point2[1]]+point1[2:], point2, matching, fast, inner = True), close = True)
            elif (isinstance(point1[2], str) and point1[2] != point2[2]) or (type(point2[2]) is list and point1[2] not in point2[2]):
                if isinstance(point1[2], str) and isinstance(point2[2], str):
                    if not fast: result.log += ("\n" if inner else "")+point1_string+" => (change incident type)"
                    result.error += 1
                    result.valid = False
                    if not fast: result.errors["Incorrect_Incident_Type"] += [point1_string]
                    result = Result.combine(result, transform(point1[:2]+[point2[2]]+point1[3:], point2, matching, fast, inner = True), close = True)
                elif type(point1[2]) is tuple and type(point2[2]) is list:
                    best_score = 1
                    best_span = point2[0]
                    for span in point2[2]:
                        score = span_scorer(point1[2], span)
                        if score < best_score: 
                            best_score = score
                            best_span = span
                    if not fast: 
                        if best_score < 1:
                            result.log += ("\n" if inner else "")+point1_string+" => (alter span)"
                            result.errors["Span_Error"] += [point1_string]
                        else: 
                            result.log += ("\n" if inner else "")+point1_string+" => (change mention)"
                    result.error += best_score
                    result = Result.combine(result, transform(point1[:2]+[best_span]+point1[3:], point2, matching, fast, inner = True), close = True)
                elif not fast:
                    result.log += ("\n" if inner else "")+"ERROR"
            else: 
                if not fast: 
                    if inner: result.log += "\n"+point2_string+' =| (done)'
                    else: result.log += point1_string+" = "+point2_string+' =| (done)'
                if not inner or clean_inner:
                    result.stats[point1[1]]["p_num"] += 1
                    result.stats[point1[1]]["r_num"] += 1
                    result.stats["total"]["p_num"] += 1
                    result.stats["total"]["r_num"] += 1
    return result

class Solution:
    """Represents a matching of data points between predicted and gold summaries.
    Contains a list of pairs of data points. """
    def __init__(self, matching):
        self.matching = matching

    def compute(self, fast = True):
        result = Result()
        result.log = "Solution:"
        for pair in self.matching:
            result = Result.combine(result, transform(*pair, self.matching, fast))
            if not result.valid: break
        return result

    def from_data(template_matching, predicted_templates, gold_templates, point_matching, predicted_singles, gold_singles):
        matching = []
        for pair in template_matching["pairs"]:
            matching.append((predicted_templates[pair[0]], gold_templates[pair[1]]))
        for unmatched_predicted in template_matching["unmatched_predicted"]:
            matching.append((predicted_templates[unmatched_predicted], None))
        for unmatched_gold in template_matching["unmatched_gold"]:
            matching.append((None, gold_templates[unmatched_gold]))
        for pair in point_matching["pairs"]:
            matching.append((predicted_singles[pair[0]], gold_singles[pair[1]]))
        for unmatched_predicted in point_matching["unmatched_predicted"]:
            matching.append((predicted_singles[unmatched_predicted], None))
        for unmatched_gold in point_matching["unmatched_gold"]:
            matching.append((None, gold_singles[unmatched_gold]))
        return Solution(matching)

class Result():
    error_names = [
        "Span_Error",
        "Spurious_Role_Filler",
        "Missing_Role_Filler",
        "Spurious_Template",
        "Missing_Template",
        "Incorrect_Role",
        "Incorrect_Incident_Type",
        "Incorrect_Template"
    ]

    def __init__(self):
        self.valid = True
        self.log = ""
        self.stats = {}
        self.stats["total"] = {
            "p_num": 0,
            "p_den": 0,
            "r_num": 0,
            "r_den": 0,
            "p": 0,
            "r": 0,
            "f1": 0,
        }
        
        self.errors = {}
        for error_name in self.error_names:
            self.errors[error_name] = []
        
        self.error = 0

    def __str__(self, verbosity = 4):
        if not self.valid: return "INVALID MATCHING"
        output_string = self.log
        self.update_stats()
        output_string += "\n\n---\n"
        for role_name in ["total"] + (list(self.stats.keys()) if verbosity >= 3 else []):
            output_string += (
                "\n"
                + role_name
                + ": f1: {0:.4f}, precision:{1:.4f}, recall: {2:.4f}".format(
                    self.stats[role_name]["f1"],
                    self.stats[role_name]["p"],
                    self.stats[role_name]["r"],
                )
            )
            if verbosity >= 2:
                output_string += (
                    "\np_num:"
                    + str(self.stats[role_name]["p_num"])
                    + " p_den:"
                    + str(self.stats[role_name]["p_den"])
                    + " r_num:"
                    + str(self.stats[role_name]["r_num"])
                    + " r_den:"
                    + str(self.stats[role_name]["r_den"])
                )
        if verbosity >= 1:
            output_string += "\n."
            for error_name, error_list in self.errors.items():
                output_string += "\n" + error_name + ": " + str(len(error_list))
            output_string += "\n."
            output_string += "\nTotal Error: "+str(self.error)
        return output_string

    def __gt__(self, other):
        self.update_stats()
        other.update_stats()
        return (not other.valid) or self.stats["total"]["f1"] > other.stats["total"]["f1"] or \
        (self.stats["total"]["f1"] == other.stats["total"]["f1"] and self.error < other.error)

    def combine(result1, result2, close = False):
        result = Result()
        result.valid = result1.valid and result2.valid
        result.log = result1.log + ("" if "" in [result1.log, result2.log] or close else "\n.\n") + result2.log
        for key in list(result1.stats.keys()) + list(result2.stats.keys()):
            result.stats[key] = {}
            for stat in ["p_num", "p_den", "r_num", "r_den"]:
                result.stats[key][stat] = ((result1.stats[key][stat] if key in result1.stats.keys() else 0) + (result2.stats[key][stat] if key in result2.stats.keys() else 0))
        for key in result.errors.keys():
            result.errors[key] = result1.errors[key] + result2.errors[key]
        result.error = result1.error + result2.error
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
            role["p"], role["r"], role["f1"] = Result.compute_scores(
                role["p_num"], role["p_den"], role["r_num"], role["r_den"]
            )
        return

def generate_solutions(predicted_points, gold_points):
    predicted_templates = [p for p in predicted_points if len(p) == 1]
    gold_templates = [p for p in gold_points if len(p) == 1]
    predicted_singles = [p for p in predicted_points if len(p) > 1]
    gold_singles = [p for p in gold_points if len(p) > 1]
    data = (predicted_templates, gold_templates, predicted_singles, gold_singles)

    predicted_templates_sorted = {}
    gold_templates_sorted = {}
    for predicted_single in predicted_singles:
        if predicted_single[1] == "incident_type":
            if predicted_single[2] not in predicted_templates_sorted.keys(): 
                predicted_templates_sorted[predicted_single[2]] = []
            predicted_templates_sorted[predicted_single[2]].append(predicted_single[0])
    for gold_single in gold_singles:
        if gold_single[1] == "incident_type":
            if gold_single[2] not in gold_templates_sorted.keys(): 
                gold_templates_sorted[gold_single[2]] = []
            gold_templates_sorted[gold_single[2]].append(gold_single[0])

    print(predicted_templates_sorted, gold_templates_sorted)


    for template_matching in generate_matchings(len(predicted_templates), len(gold_templates), data, True):

        for point_matching in generate_matchings(len(predicted_singles), len(gold_singles), data):
            yield Solution.from_data(template_matching, predicted_templates, gold_templates, point_matching, predicted_singles, gold_singles)

def generate_matchings(a, b, data, templates = False, i = None):
    if i is None: i = a-1
    if i == -1: yield {"pairs": [], "unmatched_predicted": list(range(a)), "unmatched_gold": list(range(b))}
    else:
        for matching in generate_matchings(a, b, data, templates, i-1):
            yield matching
            for j in matching["unmatched_gold"]:
                if templates and not valid(i,j,data): 
                    continue
                yield {"pairs": matching["pairs"]+[(i,j)], 
                "unmatched_predicted": [n for n in matching["unmatched_predicted"] if n != i],
                "unmatched_gold": [n for n in matching["unmatched_gold"] if n != j]}

def valid(i,j, data):
    predicted_template, gold_template = data[0][i][0], data[1][j][0]
    predicted_type = None
    for point in data[2]:
        if point[0] == predicted_template: 
            predicted_type = point[2]
            break
    for point in data[3]:
        if point[0] == gold_template:
            return point[2] == predicted_type
    return False

def analyze(predicted_points, gold_points, verbose = False):
    best_result = None
    best_solution = None
    for solution in generate_solutions(predicted_points, gold_points):
        result = solution.compute(fast = True)
        if best_result is None or result > best_result:
            best_result = result
            best_solution = solution
    return best_solution.compute(fast = False)

def normalize_string(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    s = re.sub(regex, " ", s.lower())
    return " ".join([c for c in s if c.isalnum()])

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

    for docid, example in inp_dict.items():
        pred_summary = []
        gold_summary = []

        doc_tokens = normalize_string(example["doctext"].replace(" ##", "")).split()
        documents[docid] = doc_tokens

        for i in range(len(example["pred_templates"])):
            pred_temp = example["pred_templates"][i]
            template_text = ["templateP"+str(i)]
            pred_summary.append(template_text)
            for role_name, role_data in pred_temp.items():
                role_text = template_text+[role_name]
                if role_name == "incident_type":
                    pred_summary.append(role_text+[role_data])
                    continue
                for entity in role_data:
                    for mention in entity:                        
                        mention_tokens = normalize_string(mention).split()
                        span = mention_tokens_index(doc_tokens, mention_tokens)
                        pred_summary.append(role_text+[span])

        for i in range(len(example["gold_templates"])):
            gold_temp = example["gold_templates"][i]
            template_text = ["templateG"+str(i)]
            gold_summary.append(template_text)
            for role_name, role_data in gold_temp.items():
                role_text = template_text+[role_name]
                if role_name == "incident_type":
                    gold_summary.append(role_text+[role_data])
                    continue
                for entity in role_data:
                    mentions = []
                    for mention in entity:                        
                        mention_tokens = normalize_string(mention).split()
                        span = mention_tokens_index(doc_tokens, mention_tokens)
                        mentions.append(span)
                    gold_summary.append(role_text+[mentions])

        data.append((pred_summary, gold_summary))

    return data, documents

def add_script_args(parser):
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        help="The path to the input file given to the system",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity"
    )
    parser.add_argument(
        "-at",
        "--analyze_transformed",
        action="store_true",
        help="Analyze transformed data",
    )
    parser.add_argument(
        "-s",
        "--scoring_mode",
        type=str,
        choices=["all", "msp", "mmi", "mat"],
        help=textwrap.dedent(
            """\
                        Choose scoring mode according to MUC:
                        all - All Templates
                        msp - Matched/Spurious
                        mmi - Matched/Missing
                        mat - Matched Only
                    """
        ),
        default="All_Templates",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="The path to the output file the system writes to",
    )
    return parser

if __name__ == "__main__":
    parser = add_script_args(
        argparse.ArgumentParser(
            usage='Use "python MUC_Error_Analysis_Operation.py --help" for more information',
            formatter_class=argparse.RawTextHelpFormatter,
        )
    )
    args = parser.parse_args()

    input_file = args.input_file
    verbose = args.verbose
    analyze_transformed = args.analyze_transformed

    output_file = open(args.output_file, "w")

    if args.scoring_mode == "all":
        output_file.write("Using scoring mode - All Templates\n")
        scoring_mode = "All_Templates"
    elif args.scoring_mode == "msp":
        output_file.write("Using scoring mode - Matched/Spurious\n")
        scoring_mode = "Matched/Spurious"
    elif args.scoring_mode == "mmi":
        output_file.write("Using scoring mode - Matched/Missing\n")
        scoring_mode = "Matched/Missing"
    elif args.scoring_mode == "mat":
        output_file.write("Using scoring mode - Matched Only\n")
        scoring_mode = "Matched_Only"
    else:
        output_file.write("Using default scoring mode - All Templates\n")
        scoring_mode = "All_Templates"

    data, docs = from_file(input_file)

    transformed_data = []

    output_file.write("\nANALYZING DATA AND APPLYING TRANSFORMATIONS ...")

    total_result_before = Result()

    for pair in tqdm(data, desc="Analyzing Data and Applying Transformations: "):
        output_file.write("\n\n\t---\n\n")
        output_file.write("Comparing:")
        for p in pair[0]: output_file.write("\n  "+"/".join([str(i) for i in p]))
        output_file.write("\n -to- ")
        for p in pair[1]: output_file.write("\n  "+"/".join([str(i) for i in p]))
        result = analyze(*pair, verbose)
        output_file.write("\n\n"+result.__str__(verbosity = 4))
        total_result_before = Result.combine(total_result_before, result)

    output_file.close()

