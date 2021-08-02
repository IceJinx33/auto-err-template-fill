import json, re, argparse, textwrap
from tqdm import tqdm

role_names = ["incident_type", "PerpInd", "PerpOrg", "Target", "Weapon", "Victim"]
error_names = [
        "Span_Error",
        "Spurious_Role_Filler",
        "Missing_Role_Filler",
        "Spurious_Template",
        "Missing_Template",
        "Incorrect_Role",
    ]
    
# Modes: "MUC", "MUC_Errors", "Errors"
def analyze(predicted_templates, gold_templates, mode = "MUC_Errors", scoring_mode = "All_Templates", verbose = False):

    class Result:
        def __init__(self):
            self.valid = True
            self.stats = {
                "num": 0,
                "p_den": 0,
                "r_den": 0,
                "p": 0,
                "r": 0,
                "f1": 0 
                }
            self.error_score = 0

            self.errors = {}
            for error_name in error_names: self.errors[error_name] = 0

        def __str__(self, verbosity = 4):
            result_string = "Result:"
            for k, v in self.stats.items():
                result_string += "\n" + k + ": " + str(v)
            result_string += "\nError Score: " + str(self.error_score)
            return result_string

        def update_stats(self):

            def compute_scores(num, p_den, r_den, beta=1):
                p = 0 if p_den == 0 else num / float(p_den)
                r = 0 if r_den == 0 else num / float(r_den)
                d = beta * beta * p + r
                f1 = 0 if d == 0 else (1 + beta * beta) * p * r / d
                return (p, r, f1)

            self.stats["p"], self.stats["r"], self.stats["f1"] = compute_scores(
                self.stats["num"], self.stats["p_den"], self.stats["r_den"]
            )
            return

        def __gt__(self, other):
            if not other.valid: return True
            self.update_stats()
            other.update_stats()
            if self.stats["f1"] != other.stats["f1"]: 
                return self.stats["f1"] > other.stats["f1"]
            return self.error_score < other.error_score

        def combine(result1, result2):
            result = Result()
            result.valid = result1.valid and result2.valid
            for key in ["num", "p_den", "r_den"]:
                result.stats[key] = result1.stats[key] + result2.stats[key]
            result.error_score = result1.error_score + result2.error_score
            return result

        def compute(self):
            """Generate the log and transformations for this matching"""
            return

    def template_matches(predicted_templates, gold_templates):
        if len(predicted_templates) == 0: yield [(None, gold_template) for gold_template in gold_templates]
        else:
            for matching in template_matches(predicted_templates[1:], gold_templates):
                yield [(predicted_templates[0], None)]+matching
            for i in range(len(gold_templates)):
                if mode == "Errors" or (predicted_templates[0]["incident_type"] == gold_templates[i]["incident_type"]):
                    for matching in template_matches(predicted_templates[1:], gold_templates[:i]+gold_templates[i+1:]):
                        yield [(predicted_templates[0], gold_templates[i])] + matching

    def analyze_template_matching(template_matching):
        
        def mention_matches(predicted_mentions, gold_mentions):
            if len(predicted_mentions) == 0: yield [(None, gold_mention) for gold_mention in gold_mentions]
            else:
                for matching in mention_matches(predicted_mentions[1:], gold_mentions):
                    yield [(predicted_mentions[0], None)]+matching
                for i in range(len(gold_mentions)):
                    best_score = 1
                    #best_span = None
                    for mention in gold_mentions[i]:
                        span = mention[0]
                        score = span_scorer(predicted_mentions[0][0], span)
                        if score < best_score: 
                            best_score = score
                            #best_span = span
                    if mode == "MUC" and best_score == 1: continue
                    for matching in mention_matches(predicted_mentions[1:], gold_mentions[:i]+gold_mentions[i+1:]):
                        yield [(predicted_mentions[0], gold_mentions[i], best_score)] + matching
        
        def span_scorer(span1, span2, mode="geometric_mean"):
            # Lower is better - 0 iff exact match, 1 iff no intersection, otherwise between 0 and 1
            if span1 == span2: return 0
            length1, length2 = abs(span1[1] - span1[0]), abs(span2[1] - span2[0])
            if mode == "absolute":
                val = (abs(span1[0] - span2[0]) + abs(span1[1] - span2[1])) / ( length1 + length2 )
                return min(val, 1.0)
            elif mode == "geometric_mean":
                intersection = max(0, min(span1[1], span2[1]) - max(span1[0], span2[0]))
                return 1 - (( (intersection ** 2) / (length1 * length2) ) if length1 * length2 > 0 else 0)

        result = Result()
        for template_pair in template_matching:
            # TODO: deal with incident type
            for role_name in role_names:
                if role_name == "incident_type": continue
                rolewise_result = Result()
                if template_pair[0] is None and scoring_mode in ["All_Templates", "Matched/Missing"]:
                    for _ in template_pair[1]:
                        rolewise_result.stats["r_den"] += 1
                        rolewise_result.error_score += 1
                        if mode in ["MUC_Errors", "Errors"]: rolewise_result.errors["Missing_Role_Filler"] += 1
                elif template_pair[1] is None and scoring_mode in ["All_Templates", "Matched/Spurious"]:
                    for _ in template_pair[0]:
                        rolewise_result.stats["r_den"] += 1
                        rolewise_result.error_score += 1
                        if mode in ["MUC_Errors", "Errors"]: rolewise_result.errors["Spurious_Role_Filler"] += 1
                else:
                    rolewise_result = None
                    for mention_matching in mention_matches(template_pair[0][role_name], template_pair[1][role_name]):
                        matching_result = Result()
                        for mention_pair in mention_matching:
                            if mention_pair[0] is None: 
                                matching_result.stats["r_den"] += 1
                                matching_result.error_score += 1
                                if mode in ["MUC_Errors", "Errors"]: matching_result.errors["Missing_Role_Filler"] += 1
                            elif mention_pair[1] is None:
                                matching_result.stats["p_den"] += 1
                                matching_result.error_score += 1
                                if mode in ["MUC_Errors", "Errors"]: matching_result.errors["Spurious_Role_Filler"] += 1
                            else:
                                matching_result.stats["num"] += int(mention_pair[2] == 0)
                                matching_result.stats["p_den"] += 1
                                matching_result.stats["r_den"] += 1
                                matching_result.error_score += mention_pair[2]
                                if mode in ["MUC_Errors", "Errors"] and mention_pair[2] > 0: matching_result.errors["Span_Error"] += 1
                        if matching_result.valid and (rolewise_result is None or matching_result > rolewise_result):
                            rolewise_result = matching_result
                result = Result.combine(result, rolewise_result)
        return result
                    
    best_result = Result()
    for template_matching in template_matches(predicted_templates, gold_templates):
        result = analyze_template_matching(template_matching)
        if result > best_result: best_result = result
    best_result.compute()
    return best_result

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
                    roles[role_name] = role_data
                    continue
                mentions = []
                for entity in role_data:
                    for mention in entity:
                        mention_tokens = normalize_string(mention).split()
                        span = mention_tokens_index(doc_tokens, mention_tokens)
                        mentions.append((span, mention))
                roles[role_name] = mentions
            pred_templates.append(roles)

        for gold_temp in example["gold_templates"]:
            roles = {}
            for role_name, role_data in gold_temp.items():
                if role_name == "incident_type":
                    roles[role_name] = role_data
                    continue
                coref_mentions = []
                for entity in role_data:
                    mentions = []
                    for mention in entity:
                        mention_tokens = normalize_string(mention).split()
                        span = mention_tokens_index(doc_tokens, mention_tokens)
                        mentions.append((span, mention))
                    coref_mentions.append(mentions)
                roles[role_name] = coref_mentions
            gold_templates.append(roles)

        data.append((pred_templates, gold_templates))

    return data, documents

def main():
    
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

    for pair in tqdm(data, desc="Analyzing Data and Applying Transformations: "):
        output_file.write("\n\n\t---\n\n")
        output_file.write("Comparing:")
        output_file.write(str(pair[0])+"\n -to- "+str(pair[1]))
        result = analyze(*pair, "MUC_Errors", scoring_mode, verbose)
        output_file.write("\n\n"+result.__str__(verbosity = 4))

    output_file.close()

if __name__ == "__main__": main()
