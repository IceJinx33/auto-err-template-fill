import json, re, argparse, textwrap
from tqdm import tqdm

# Modes: "MUC", "MUC_Errors", "Errors"
def analyze(predicted_templates, gold_templates, mode = "MUC_Errors", verbose = False):

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

        def update_stats(self):

            def compute_scores(p_num, p_den, r_num, r_den, beta=1):
                p = 0 if p_den == 0 else p_num / float(p_den)
                r = 0 if r_den == 0 else r_num / float(r_den)
                d = beta * beta * p + r
                f1 = 0 if d == 0 else (1 + beta * beta) * p * r / d
                return (p, r, f1)

            for _, role in self.stats.items():
                role["p"], role["r"], role["f1"] = compute_scores(
                    role["p_num"], role["p_den"], role["r_num"], role["r_den"]
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

        def compute():
            """Generate the log and transformations for this matching"""
            pass

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
                    for span in gold_mentions[i]:
                        score = span_scorer(predicted_mentions[0], span)
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
                val = (abs(span1[0] - span2[0]) + abs(span1[1] - span2[1])) / (
                    length1 + length2
                )
                return min(val, 1.0)
            elif mode == "geometric_mean":
                intersection = max(0, min(span1[1], span2[1]) - max(span1[0], span2[0]))
                return 1 - (( (intersection ** 2) / (length1 * length2) ) if length1 * length2 > 0 else 0)

        result = Result()
        for template_pair in template_matching:
            # TODO: deal with incident type
            for role_name in role_names:
                if template_pair[0] is None: 
                    for _ in template_pair[1]:
                        rolewise_result.stats["r_den"] += 1
                        rolewise_result.error_score += 1
                        if mode in ["MUC_Errors", "Errors"]: rolewise_result.errors["unmatched_role_filler"] += 1
                elif template_pair[1] is None: 
                    for _ in template_pair[0]:
                        rolewise_result.stats["r_den"] += 1
                        rolewise_result.error_score += 1
                        if mode in ["MUC_Errors", "Errors"]: rolewise_result.errors["unmatched_role_filler"] += 1
                else:
                    best_result = None
                    for mention_matching in mention_matches(template_pair[0][role_name], template_pair[1][role_name]):
                        rolewise_result = Result()
                        for mention_pair in mention_matching:
                            if mention_pair[0] is None: 
                                rolewise_result.stats["r_den"] += 1
                                rolewise_result.error_score += 1
                                if mode in ["MUC_Errors", "Errors"]: rolewise_result.errors["unmatched_role_filler"] += 1
                            elif mention_pair[1] is None:
                                rolewise_result.stats["p_den"] += 1
                                rolewise_result.error_score += 1
                                if mode in ["MUC_Errors", "Errors"]: rolewise_result.errors["spurious_role_filler"] += 1
                            else:
                                rolewise_result.stats["num"] += int(mention_pair[2] == 0)
                                rolewise_result.stats["p_den"] += 1
                                rolewise_result.stats["r_den"] += 1
                                rolewise_result.error_score += mention_pair[2]
                                if mode in ["MUC_Errors", "Errors"] and mention_pair[2] > 0: rolewise_result.errors["span_error"] += 1
                        if rolewise_result.valid and (best_result is None or rolewise_result > best_result):
                            best_result = rolewise_result
                    result = Result.combine(result, best_result)
        return result
                    
    best_matching = Matching()
    for template_matching in template_matches(predicted_templates, gold_templates):
        matching = analyze_template_matching(template_matching)
        if matching > best_matching: best_matching = matching
    best_matching.compute()
    return best_matching

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
        for p in pair[0]: output_file.write("\n  "+"/".join([str(i) for i in p]))
        output_file.write("\n -to- ")
        for p in pair[1]: output_file.write("\n  "+"/".join([str(i) for i in p]))
        result = analyze(*pair, "MUC_Errors", verbose)
        output_file.write("\n\n"+result.__str__(verbosity = 4))

    output_file.close()

if __name__ == "__main__": main()
