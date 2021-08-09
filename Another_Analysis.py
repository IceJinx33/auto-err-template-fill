import json, re, argparse, textwrap
from tqdm import tqdm

#role_names = ["incident_type", "PerpInd", "PerpOrg", "Target", "Weapon", "Victim"]
role_names = ["Status", "Country", "Disease", "Victims"]

error_names = [
    "Span_Error",
    "Spurious_Role_Filler",
    "Duplicate_Role_Filler",
    "Cross_Template_Spurious_Role_Filler",
    "Incorrect_Role",
    "Unrelated_Spurious_Role_Filler",
    "Missing_Role_Filler",
    "Spurious_Template",
    "Missing_Template",
    "Incorrect_Incident_Type",
]


def summary_to_str(templates, mode):
    result_string = "Summary:"
    for template in templates:
        if template is None:
            result_string += " None"
            continue
        if mode == "MUC_Errors":
            result_string += "\n|-Template (" + template["incident_type"] + "):"
        else:
            result_string += "\n|-Template:"
        for k, v in template.items():
            if mode == "MUC_Errors" and k == "incident_type":
                continue
            result_string += "\n| |-" + k + ": " + ", ".join([str(i) for i in v])
    return result_string


class Result:
    def __init__(self):
        self.valid = True
        self.stats = {"num": 0, "p_den": 0, "r_den": 0, "p": 0, "r": 0, "f1": 0}
        self.error_score = 0

        self.errors = {}
        for error_name in error_names:
            self.errors[error_name] = 0

        self.spurious_rfs = []

    def __str__(self, verbosity=4):
        result_string = "Result:"
        for k, v in self.stats.items():
            result_string += "\n" + k + ": " + str(v)
        result_string += "\nError Score: " + str(self.error_score)
        for k, v in self.errors.items():
            if k in [
                "Duplicate_Role_Filler",
                "Cross_Template_Spurious_Role_Filler",
                "Incorrect_Role",
                "Unrelated_Spurious_Role_Filler",
            ]:
                result_string += "\n| |-" + k + ": " + str(v)
            else:
                result_string += "\n" + k + ": " + str(v)
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
        if not other.valid:
            return True
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

        result.spurious_rfs = result1.spurious_rfs + result2.spurious_rfs

        for error_name in error_names:
            result.errors[error_name] = (
                result1.errors[error_name] + result2.errors[error_name]
            )

        return result

    def compute(self):
        """Generate the log and transformations for this matching"""
        return


# Modes: "MUC", "MUC_Errors", "Errors"
def analyze(
    predicted_templates,
    gold_templates,
    mode="MUC_Errors",
    scoring_mode="All_Templates",
    verbose=False,
):
    def template_matches(predicted_templates, gold_templates):
        if len(predicted_templates) == 0:
            yield [(None, gold_template) for gold_template in gold_templates]
        else:
            for matching in template_matches(predicted_templates[1:], gold_templates):
                yield [(predicted_templates[0], None)] + matching
            for i in range(len(gold_templates)):
                if mode == "Errors" or (
                    predicted_templates[0]["incident_type"]
                    == gold_templates[i]["incident_type"]
                ):
                    for matching in template_matches(
                        predicted_templates[1:],
                        gold_templates[:i] + gold_templates[i + 1 :],
                    ):
                        yield [(predicted_templates[0], gold_templates[i])] + matching

    def analyze_template_matching(template_matching):
        def mention_matches(predicted_mentions, gold_mentions):
            if len(predicted_mentions) == 0:
                yield [(None, gold_mention) for gold_mention in gold_mentions]
            else:
                for matching in mention_matches(predicted_mentions[1:], gold_mentions):
                    yield [(predicted_mentions[0], None)] + matching
                for i in range(len(gold_mentions)):
                    best_score = 1
                    # best_span = None
                    for mention in gold_mentions[i]:
                        span = mention[0]
                        score = span_scorer(predicted_mentions[0][0], span)
                        if score < best_score:
                            best_score = score
                            # best_span = span
                    if mode == "MUC_Errors" and best_score == 1:
                        continue
                    for matching in mention_matches(
                        predicted_mentions[1:],
                        gold_mentions[:i] + gold_mentions[i + 1 :],
                    ):
                        yield [
                            (predicted_mentions[0], gold_mentions[i], best_score)
                        ] + matching

        def span_scorer(span1, span2, span_mode="geometric_mean"):
            # Lower is better - 0 iff exact match, 1 iff no intersection, otherwise between 0 and 1
            if span1 == span2:
                return 0
            length1, length2 = abs(span1[1] - span1[0]), abs(span2[1] - span2[0])
            if span_mode == "absolute":
                val = (abs(span1[0] - span2[0]) + abs(span1[1] - span2[1])) / (
                    length1 + length2
                )
                return min(val, 1.0)
            elif span_mode == "geometric_mean":
                intersection = max(0, min(span1[1], span2[1]) - max(span1[0], span2[0]))
                return 1 - (
                    ((intersection ** 2) / (length1 * length2))
                    if length1 * length2 > 0
                    else 0
                )

        result = Result()
        for template_pair in template_matching:
            pairwise_result = Result()
            print(
                summary_to_str([template_pair[0]], mode),
                summary_to_str([template_pair[1]], mode),
            )
            if template_pair[0] is None and scoring_mode in [
                "All_Templates",
                "Matched/Missing",
            ]:
                if mode in ["MUC_Errors", "Errors"]:
                    pairwise_result.errors["Missing_Template"] += 1
                for role_name, mentions in template_pair[1].items():
                    if mode == "MUC_Errors" and role_name == "incident_type":
                        pairwise_result.stats["r_den"] += 1
                        pairwise_result.error_score += 1
                    else:
                        for _ in mentions:
                            pairwise_result.stats["r_den"] += 1
                            pairwise_result.error_score += 1
                            if mode in ["MUC_Errors", "Errors"]:
                                pairwise_result.errors["Missing_Role_Filler"] += 1
            elif template_pair[1] is None and scoring_mode in [
                "All_Templates",
                "Matched/Spurious",
            ]:
                if mode in ["MUC_Errors", "Errors"]:
                    pairwise_result.errors["Spurious_Template"] += 1
                for role_name, mentions in template_pair[0].items():
                    if mode == "MUC_Errors" and role_name == "incident_type":
                        pairwise_result.stats["p_den"] += 1
                        pairwise_result.error_score += 1
                    else:
                        for pred_mention in mentions:
                            pairwise_result.stats["p_den"] += 1
                            pairwise_result.error_score += 1
                            if mode in ["MUC_Errors", "Errors"]:
                                pairwise_result.errors["Spurious_Role_Filler"] += 1
                                pairwise_result.spurious_rfs.append(
                                    (template_pair[0], role_name, pred_mention)
                                )
            else:
                for role_name in role_names:
                    print(role_name)
                    rolewise_result = Result()
                    if mode == "MUC_Errors" and role_name == "incident_type":
                        match = (
                            template_pair[0][role_name] == template_pair[1][role_name]
                        )
                        if mode in ["MUC", "MUC_Errors"]:
                            assert match, "incompatible matching"
                        rolewise_result.stats["num"] += int(match)
                        rolewise_result.stats["p_den"] += 1
                        rolewise_result.stats["r_den"] += 1
                        rolewise_result.error_score += int(not match)
                        if mode in ["MUC_Errors", "Errors"] and not match:
                            rolewise_result.errors["Incorrect_Incident_Type"] += 1
                    else:
                        rolewise_result = None
                        for mention_matching in mention_matches(
                            template_pair[0][role_name], template_pair[1][role_name]
                        ):
                            print(mention_matching)
                            matching_result = Result()
                            for mention_pair in mention_matching:
                                if mention_pair[0] is None:
                                    matching_result.stats["r_den"] += 1
                                    matching_result.error_score += 1
                                    if mode in ["MUC_Errors", "Errors"]:
                                        matching_result.errors[
                                            "Missing_Role_Filler"
                                        ] += 1
                                elif mention_pair[1] is None:
                                    matching_result.stats["p_den"] += 1
                                    matching_result.error_score += 1
                                    if mode in ["MUC_Errors", "Errors"]:
                                        matching_result.errors[
                                            "Spurious_Role_Filler"
                                        ] += 1
                                        matching_result.spurious_rfs.append(
                                            (
                                                template_pair[0],
                                                role_name,
                                                mention_pair[0],
                                            )
                                        )
                                else:
                                    matching_result.stats["num"] += int(
                                        mention_pair[2] == 0
                                    )
                                    matching_result.stats["p_den"] += 1
                                    matching_result.stats["r_den"] += 1
                                    matching_result.error_score += mention_pair[2]
                                    if (
                                        mode in ["MUC_Errors", "Errors"]
                                        and 0 < mention_pair[2] < 1
                                    ):
                                        matching_result.errors["Span_Error"] += 1
                                    if (
                                        mode in ["MUC_Errors", "Errors"]
                                        and mention_pair[2] == 1
                                    ):
                                        matching_result.errors[
                                            "Missing_Role_Filler"
                                        ] += 1
                                        matching_result.errors[
                                            "Spurious_Role_Filler"
                                        ] += 1
                                        matching_result.spurious_rfs.append(
                                            (
                                                template_pair[0],
                                                role_name,
                                                mention_pair[0],
                                            )
                                        )
                            if matching_result.valid and (
                                rolewise_result is None
                                or matching_result > rolewise_result
                            ):
                                rolewise_result = matching_result
                            print(matching_result)
                    pairwise_result = Result.combine(pairwise_result, rolewise_result)
            result = Result.combine(result, pairwise_result)
        print(result)
        return result

    best_result = None
    best_matching = None
    for template_matching in template_matches(predicted_templates, gold_templates):
        result = analyze_template_matching(template_matching)
        if result.valid and (best_result is None or result > best_result):
            best_result = result
            best_matching = template_matching

    def handle_spurious_rfs(best_result, best_matching):

        for pred_template, pred_role_name, pred_mention in best_result.spurious_rfs:

            error_found = False
            matched_gold_template = None

            for template_pair in best_matching:
                if template_pair[0] == pred_template:
                    matched_gold_template = template_pair[1]
                    for role_name in role_names:
                        if matched_gold_template != None and pred_mention in [
                            mention
                            for corefs in matched_gold_template[role_name]
                            for mention in corefs
                        ]:
                            if role_name != pred_role_name:
                                best_result.errors["Incorrect_Role"] += 1
                                error_found = True
                                break
                            else:
                                best_result.errors["Duplicate_Role_Filler"] += 1
                                error_found = True
                                break

                    break

            if error_found:
                continue
            else:

                """
                for role_name in role_names:
                    for gold_template in gold_templates:

                        if matched_gold_template != None and gold_template != matched_gold_template:
                            for corefs in gold_template[role_name]:
                                for mention in corefs:

                                    if pred_mention == mention:
                                        best_result.errors["Cross_Template_Spurious_Role_Filler"] += 1
                                        if pred_role_name != role_name:
                                            best_result.errors["Incorrect_Role"] += 1
                                        error_found = True
                                        break
                                if error_found: break
                        if error_found: break
                    if error_found: break
                """

                for role_name in role_names:
                    gold_mention_lst = []
                    for gold_template in gold_templates:
                        if (
                            matched_gold_template != None
                            and gold_template != matched_gold_template
                        ):
                            gold_mention_lst += [
                                mention
                                for corefs in gold_template[role_name]
                                for mention in corefs
                            ]
                    try:
                        idx = gold_mention_lst.index(pred_mention)
                        best_result.errors["Cross_Template_Spurious_Role_Filler"] += 1
                        if pred_role_name != role_name:
                            best_result.errors["Incorrect_Role"] += 1
                        error_found = True
                        break
                    except:
                        continue

                if not error_found:
                    best_result.errors["Unrelated_Spurious_Role_Filler"] += 1

    handle_spurious_rfs(best_result, best_matching)
    best_result.compute()

    return best_result


def from_file(input_file, mode):
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
                if mode == "MUC_Errors" and role_name == "incident_type":
                    roles[role_name] = role_data
                    continue
                if mode == "Errors" and type(role_data) != list:
                    try:
                        rdata_tokens = normalize_string(str(role_data))
                        span = mention_tokens_index(doc_tokens, rdata_tokens)
                        roles[role_name] = [(span, str(role_data))]
                    except:
                        raise Exception(
                            "The datatype associated with the role "
                            + str(role_name)
                            + " could not be converted to a string."
                        )
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
                if mode == "Errors" and type(role_data) != list:
                    try:
                        rdata_tokens = normalize_string(str(role_data))
                        span = mention_tokens_index(doc_tokens, rdata_tokens)
                        roles[role_name] = [[(span, str(role_data))]]
                    except:
                        raise Exception(
                            "The datatype associated with the role "
                            + str(role_name)
                            + " could not be converted to a string."
                        )
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
            "-m",
            "--mode",
            type=str,
            choices=["MUC_Errors", "Errors"],
            help=textwrap.dedent(
                """\
                            Choose evaluation mode:
                            MUC_Errors - MUC evaluation with added constraint of incident_types of templates needing to match
                            Errors - General evaluation with no added constraints
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

    if args.mode == "MUC_Errors":
        mode = args.mode
    else:
        mode = "Errors"

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

    data, docs = from_file(input_file, mode)

    transformed_data = []

    output_file.write("\nANALYZING DATA AND APPLYING TRANSFORMATIONS ...")

    total_result_before = Result()

    for pair in tqdm(data, desc="Analyzing Data and Applying Transformations: "):
        output_file.write("\n\n\t---\n\n")
        output_file.write("Comparing:")
        output_file.write(
            "\n"
            + summary_to_str(pair[0], mode)
            + "\n -to- \n"
            + summary_to_str(pair[1], mode)
        )
        result = analyze(*pair, mode, scoring_mode, verbose)
        output_file.write("\n\n" + result.__str__(verbosity=4))
        total_result_before = Result.combine(total_result_before, result)

    total_result_before.update_stats()
    output_file.write(
        "\n\n************************************\nTotal Result Before Transformation : \n************************************"
    )
    output_file.write("\n\n" + total_result_before.__str__(verbosity=4))
    output_file.close()


if __name__ == "__main__":
    main()
