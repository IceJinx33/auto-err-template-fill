import time

start = time.time()

import json, re, argparse, textwrap, copy
from tqdm import tqdm
import psutil, os

def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
  
# decorator function
def profile(func):
    def wrapper(*args, **kwargs):
  
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print("{}:consumed memory: {:,}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before))
        return result
    return wrapper

# MUC - MUC_Errors
# role_names = ["incident_type", "PerpInd", "PerpOrg", "Target", "Weapon", "Victim"]

# ProMed - Errors
# role_names = ["Status", "Country", "Disease", "Victims"]

# SciREX - Errors
role_names = ["Material", "Method", "Metric", "Task"]

error_names = [
    "Span_Error",
    "General_Spurious_Role_Filler",
    "Duplicate_Role_Filler",
    "Duplicate_Partially_Matched_Role_Filler",
    "Within_Template_Incorrect_Role",
    "Within_Template_Incorrect_Role + Partially_Matched_Filler",
    "Wrong_Template_For_Role_Filler",
    "Wrong_Template_For_Partially_Matched_Role_Filler",
    "Wrong_Template + Wrong_Role",
    "Wrong_Template + Wrong_Role + Partially_Matched_Filler",
    "Spurious_Role_Filler",
    "Missing_Role_Filler",
    "Spurious_Template",
    "Missing_Template"
]

transformation_names = [
    "Alter_Span",
    "Remove_Duplicate_Role_Filler",
    "Remove_Cross_Template_Spurious_Role_Filler",
    "Alter_Role",
    "Remove_Unrelated_Spurious_Role_Filler",
    "Introduce_Missing_Role_Filler",
    "Remove_Spurious_Template",
    "Introduce_Missing_Template"
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

        self.stats = {}
        for key in role_names + ["total"]:
            self.stats[key] = {"num": 0, "p_den": 0, "r_den": 0, "p": 0, "r": 0, "f1": 0}

        self.error_score = 0

        self.errors = {}
        for error_name in error_names:
            self.errors[error_name] = 0

        self.spurious_rfs = []
        self.missing_rfs = []
        self.transformations = []
        self.valid_trans = []
        self.transformed_data = []

    def __str__(self, verbosity=4):
        
        result_string = ""
        pair_count = 0
        if verbosity == 2 or verbosity == 4:
            result_string += "Transformations:"
            for tidx, trans in enumerate(self.transformations):
                if trans == "\n":
                    result_string += "\n\n"
                    pair_count += 1
                    result_string += "Template Pair " + str(pair_count) + ":"
                elif trans[2][0] == "Alter_Role" and self.valid_trans[tidx]:
                    result_string += "\n|-" + " -> ".join([transform for transform in trans[2]]) + ":"
                    result_string += "\n  From " + trans[0] + ": " + str(trans[1]) + " to " + trans[3] + ": " + str(trans[1]) 
                    if len(trans[2]) != 1:
                        result_string += " to None"
                else:
                    if self.valid_trans[tidx]:
                        result_string += "\n|-" + " -> ".join([transform for transform in trans[2]]) + ":"
                        result_string += "\n  " + str(trans[0]) + ": From " + str(trans[1]) + " to " + str(trans[3])
            if verbosity == 4: result_string += "\n\n"

        if verbosity == 3 or verbosity == 4:
            result_string += "Result:\n"

            for key in ["total"] + role_names:
                result_string += key + ": Precision : {1:.4f}, Recall : {2:.4f}, F1 : {0:.4f}\n".format(self.stats[key]["f1"],
                                                                                     self.stats[key]["p"],
                                                                                     self.stats[key]["r"])

            result_string += "\nError Score: " + str(self.error_score)
            for k, v in self.errors.items():
                if k in [
                    "Duplicate_Role_Filler",
                    "Duplicate_Partially_Matched_Role_Filler",
                    "Within_Template_Incorrect_Role",
                    "Within_Template_Incorrect_Role + Partially_Matched_Filler",
                    "Wrong_Template_For_Role_Filler",
                    "Wrong_Template_For_Partially_Matched_Role_Filler",
                    "Wrong_Template + Wrong_Role",
                    "Wrong_Template + Wrong_Role + Partially_Matched_Filler",
                    "Spurious_Role_Filler"
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

        for key, role in self.stats.items():
            self.stats[key]["p"], self.stats[key]["r"], self.stats[key]["f1"] = compute_scores(role["num"], role["p_den"],
                                                                     role["r_den"])

        return

    def __gt__(self, other):
        if not other.valid:
            return True
        self.update_stats()
        other.update_stats()
        if self.stats["total"]["f1"] != other.stats["total"]["f1"]:
            return self.stats["total"]["f1"] > other.stats["total"]["f1"]
        return self.error_score < other.error_score

    def combine(result1, result2):
        result = Result()
        result.valid = result1.valid and result2.valid

        for key in result.stats.keys():
            for stat in ["num", "p_den", "r_den"]:
                result.stats[key][stat] = result1.stats[key][stat] + result2.stats[key][stat]

        result.error_score = result1.error_score + result2.error_score

        result.spurious_rfs = result1.spurious_rfs + result2.spurious_rfs
        result.missing_rfs = result1.missing_rfs + result2.missing_rfs

        result.transformations = result1.transformations + result2.transformations
        result.valid_trans = result1.valid_trans + result2.valid_trans

        result.transformed_data = result1.transformed_data + result2.transformed_data

        for error_name in error_names:
            result.errors[error_name] = (
                result1.errors[error_name] + result2.errors[error_name]
            )

        return result

    def compute(self, template_matching, docid):

        """Generate the transformed templates for this matching"""

        pair_count = -1
        pred_templates = [None]*len(template_matching)
        org_pred_templates = [pair[0] for pair in template_matching]
        gold_templates = [pair[1] for pair in template_matching]

        hand_sprfs = {}
        for template, _, mention in self.spurious_rfs:
            if str(template) in hand_sprfs:
                hand_sprfs[str(template)].append(mention)
            else:
                hand_sprfs[str(template)] = [(mention)]

        hand_mprfs = {}
        for template, role_name, corefs in self.missing_rfs:
            if str(template) in hand_mprfs:
                if role_name in hand_mprfs[str(template)]:
                    hand_mprfs[str(template)][role_name] += [mention for mention in corefs]
                else:
                    hand_mprfs[str(template)][role_name] = [mention for mention in corefs]
            else:
                hand_mprfs[str(template)] = {}
                hand_mprfs[str(template)][role_name] = [mention for mention in corefs]

        temp_pos = -1
        altered_transformations = []
        remove_transformations = []
        for ind, trans in enumerate(self.transformations):
            if trans == "\n":
                altered_transformations.append(trans)
                temp_pos = ind + 1
            elif trans[2] == ["Alter_Role"]:
                altered_transformations.insert(temp_pos, trans)
                mis = (trans[3], None, ["Introduce_Missing_Role_Filler"], trans[1])
                remove_transformations.append(mis)
            else:
                altered_transformations.append(trans)

        for trans in altered_transformations:
            try:
                idx = remove_transformations.index(trans)
                self.valid_trans.append(False)
                self.errors["Missing_Role_Filler"] -= 1
            except:
                self.valid_trans.append(True)

        self.transformations = altered_transformations

        for tidx, trans in enumerate(self.transformations):       
            if trans == "\n":
                pair_count += 1
                if pred_templates[pair_count] == None:
                    pred_templates[pair_count] = copy.deepcopy(org_pred_templates[pair_count])
            elif trans[2] == ["Alter_Span"] and pred_templates[pair_count] != None:
                idx = pred_templates[pair_count][trans[0]].index(trans[1])
                if trans[3] in pred_templates[pair_count][trans[0]]:
                    pred_templates[pair_count][trans[0]].pop(idx)
                    continue
                else:
                    pred_templates[pair_count][trans[0]][idx] = trans[3]
            elif trans[2] == ["Remove_Duplicate_Role_Filler"] :
                if pred_templates[pair_count] != None:
                    idx = pred_templates[pair_count][trans[0]].index(trans[1])
                    _ = pred_templates[pair_count][trans[0]].pop(idx)
            elif trans[2] == ["Alter_Role"]:
                if pred_templates[pair_count] != None:
                    idx = pred_templates[pair_count][trans[0]].index(trans[1])
                    _ = pred_templates[pair_count][trans[0]].pop(idx)
                    
                    if trans[1] in pred_templates[pair_count][trans[3]]:
                        continue
                    else:
                        try:
                           if trans[1] in hand_mprfs[str(org_pred_templates[pair_count])][trans[3]]:
                               pred_templates[pair_count][trans[3]].append(trans[1])
                        except:
                            continue
                        
            elif trans[2] == ["Remove_Cross_Template_Spurious_Role_Filler"]:
                if pred_templates[pair_count] != None:
                    idx = pred_templates[pair_count][trans[0]].index(trans[1])
                    _ = pred_templates[pair_count][trans[0]].pop(idx)

                    temp_idx = gold_templates.index(trans[4])
                    if pred_templates[temp_idx] == None:
                        pred_templates[temp_idx] = copy.deepcopy(org_pred_templates[temp_idx])
                    
                    if org_pred_templates[temp_idx] != None and pred_templates[temp_idx] != "Removed":
                        if (trans[1] in pred_templates[temp_idx][trans[0]]):
                            continue
                        else:
                            try:
                                if trans[1] in hand_mprfs[str(org_pred_templates[temp_idx])][trans[0]] and trans[1] not in hand_sprfs[str(org_pred_templates[temp_idx])]:
                                    pred_templates[temp_idx][trans[0]].append(trans[1])
                            except:
                                continue

            elif trans[2] == ["Alter_Role", "Remove_Cross_Template_Spurious_Role_Filler"]:
                if pred_templates[pair_count] != None:
                    idx = pred_templates[pair_count][trans[0]].index(trans[1])
                    _ = pred_templates[pair_count][trans[0]].pop(idx)
                    temp_idx = gold_templates.index(trans[4])
                    if pred_templates[temp_idx] == None:
                        pred_templates[temp_idx] = copy.deepcopy(org_pred_templates[temp_idx])
                    if org_pred_templates[temp_idx] != None and pred_templates[temp_idx] != "Removed":
                        if trans[1] in pred_templates[temp_idx][trans[3]]:
                            continue
                        else:
                            try:
                                if trans[1] in hand_mprfs[str(org_pred_templates[temp_idx])][trans[3]] and trans[1] not in hand_sprfs[str(org_pred_templates[temp_idx])]:
                                    pred_templates[temp_idx][trans[3]].append(trans[1])
                            except:
                                continue
                        
            elif trans[2] == ["Remove_Unrelated_Spurious_Role_Filler"]:
                if pred_templates[pair_count] != None:
                    idx = pred_templates[pair_count][trans[0]].index(trans[1])
                    _ = pred_templates[pair_count][trans[0]].pop(idx)
            elif trans[2] == ["Introduce_Missing_Role_Filler"]:
                if pred_templates[pair_count] != None and self.valid_trans[tidx]:
                    if trans[3] in pred_templates[pair_count][trans[0]]:
                        continue
                    else:
                        pred_templates[pair_count][trans[0]].append(trans[3])
            elif trans[2] == ["Remove_Spurious_Template"]:
                pred_templates[pair_count] = "Removed"
            elif trans[2] == ["Introduce_Missing_Template"]:
                pred_templates[pair_count] = {}
                for role_key in trans[3]:
                    pred_templates[pair_count][role_key] = []
                    if trans[3][role_key] == []:
                        continue
                    elif type(trans[3][role_key]) != list:
                        pred_templates[pair_count][role_key] = trans[3][role_key]
                    else:
                        for coref in trans[3][role_key]:
                            pred_templates[pair_count][role_key].append(coref[0])
                            
            else:
                raise Exception("Incorrect transformation type")
        
        proc_pred_templates = [temp for temp in pred_templates if not(temp == None or temp == "Removed")]
        proc_gold_templates = [temp for temp in gold_templates if not(temp == None or temp == "Removed")] 
        self.transformed_data = [(docid, (proc_pred_templates, proc_gold_templates))]
        

# Modes: "MUC", "MUC_Errors", "Errors"
def analyze(
    docid,
    predicted_templates,
    gold_templates,
    mode="MUC_Errors",
    scoring_mode="All_Templates",
    verbose=False
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
                    best_gold_mention = None
                    for mention in gold_mentions[i]:
                        span = mention[0]
                        score = span_scorer(predicted_mentions[0][0], span)
                        if score < best_score:
                            best_score = score
                            best_gold_mention = mention
                    if best_score == 1:
                        continue
                    for matching in mention_matches(
                        predicted_mentions[1:],
                        gold_mentions[:i] + gold_mentions[i + 1 :],
                    ):
                        yield [
                            (predicted_mentions[0], gold_mentions[i], best_score, best_gold_mention)
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
            pairwise_result.transformations.append("\n")
            if template_pair[0] is None and scoring_mode in [
                "All_Templates",
                "Matched/Missing",
            ]:
                if mode in ["MUC_Errors", "Errors"]:
                    pairwise_result.errors["Missing_Template"] += 1
                    pairwise_result.transformations.append(("", None, ["Introduce_Missing_Template"], template_pair[1]))
                for role_name, mentions in template_pair[1].items():
                    if mode == "MUC_Errors" and role_name == "incident_type":
                        pairwise_result.stats[role_name]["r_den"] += 1
                        pairwise_result.stats["total"]["r_den"] += 1
                        pairwise_result.error_score += 1
                    else:
                        for gold_mention in mentions:
                            pairwise_result.stats[role_name]["r_den"] += 1
                            pairwise_result.stats["total"]["r_den"] += 1
                            pairwise_result.error_score += 1
                            if mode in ["MUC_Errors", "Errors"]:
                                continue
                                pairwise_result.errors["Missing_Role_Filler"] += 1
                                pairwise_result.transformations.append((role_name, None, ["Introduce_Missing_Role_Filler"], gold_mention[0]))
                                pairwise_result.missing_rfs.append(
                                            (
                                                template_pair[0],
                                                role_name,
                                                gold_mention
                                            )
                                        )

            elif template_pair[1] is None and scoring_mode in [
                "All_Templates",
                "Matched/Spurious",
            ]:
                if mode in ["MUC_Errors", "Errors"]:
                    pairwise_result.errors["Spurious_Template"] += 1
                    pairwise_result.transformations.append(("", template_pair[0], ["Remove_Spurious_Template"], None))
                for role_name, mentions in template_pair[0].items():
                    if mode == "MUC_Errors" and role_name == "incident_type":
                        pairwise_result.stats[role_name]["p_den"] += 1
                        pairwise_result.stats["total"]["p_den"] += 1
                        pairwise_result.error_score += 1
                    else:
                        for pred_mention in mentions:
                            pairwise_result.stats[role_name]["p_den"] += 1
                            pairwise_result.stats["total"]["p_den"] += 1
                            pairwise_result.error_score += 1
                            continue
                            if mode in ["MUC_Errors", "Errors"]:
                                pairwise_result.errors["Spurious_Role_Filler"] += 1
                                pairwise_result.spurious_rfs.append(
                                    (template_pair[0], role_name, pred_mention)
                                )
                            pairwise_result.transformations.append(("", "", ["Remove_Spurious_Role_Filler"], ""))    
            else:
                for role_name in role_names:
                    
                    rolewise_result = Result()
                    if mode == "MUC_Errors" and role_name == "incident_type":
                        match = (
                            template_pair[0][role_name] == template_pair[1][role_name]
                        )
                        if mode in ["MUC", "MUC_Errors"]:
                            assert match, "incompatible matching"
                        rolewise_result.stats[role_name]["num"] += int(match)
                        rolewise_result.stats[role_name]["p_den"] += 1
                        rolewise_result.stats[role_name]["r_den"] += 1

                        rolewise_result.stats["total"]["num"] += int(match)
                        rolewise_result.stats["total"]["p_den"] += 1
                        rolewise_result.stats["total"]["r_den"] += 1
                        rolewise_result.error_score += int(not match)

                        #if mode in ["MUC_Errors", "Errors"] and not match:
                            #rolewise_result.errors["Incorrect_Incident_Type"] += 1
                    else:
                        rolewise_result = None
                        for mention_matching in mention_matches(
                            template_pair[0][role_name], template_pair[1][role_name]
                        ):
                            
                            matching_result = Result()
                            for mention_pair in mention_matching:
                                if mention_pair[0] is None:
                                    matching_result.stats[role_name]["r_den"] += 1
                                    matching_result.stats["total"]["r_den"] += 1
                                    matching_result.error_score += 1
                                    if mode in ["MUC_Errors", "Errors"]:
                                        matching_result.errors[
                                            "Missing_Role_Filler"
                                        ] += 1
                                        matching_result.transformations.append((role_name, None, ["Introduce_Missing_Role_Filler"], mention_pair[1][0]))
                                        matching_result.missing_rfs.append(
                                            (
                                                template_pair[0],
                                                role_name,
                                                mention_pair[1]
                                            )
                                        )

                                elif mention_pair[1] is None:
                                    matching_result.stats[role_name]["p_den"] += 1
                                    matching_result.stats["total"]["p_den"] += 1
                                    matching_result.error_score += 1
                                    if mode in ["MUC_Errors", "Errors"]:
                                        matching_result.errors[
                                            "General_Spurious_Role_Filler"
                                        ] += 1
                                        matching_result.spurious_rfs.append(
                                            (
                                                template_pair[0],
                                                role_name,
                                                mention_pair[0],
                                            )
                                        )
                                        matching_result.transformations.append(("", "", ["Remove_Spurious_Role_Filler"], ""))
                                else:
                                    matching_result.stats[role_name]["num"] += int(
                                        mention_pair[2] == 0
                                    )
                                    matching_result.stats[role_name]["p_den"] += 1
                                    matching_result.stats[role_name]["r_den"] += 1

                                    matching_result.stats["total"]["num"] += int(
                                        mention_pair[2] == 0
                                    )
                                    matching_result.stats["total"]["p_den"] += 1
                                    matching_result.stats["total"]["r_den"] += 1

                                    matching_result.error_score += mention_pair[2]
                                    if (
                                        mode in ["MUC_Errors", "Errors"]
                                        and 0 < mention_pair[2] < 1
                                    ):
                                        matching_result.errors["Span_Error"] += 1
                                        matching_result.transformations.append((role_name, mention_pair[0], ["Alter_Span"], mention_pair[3]))
                                        
                                    if (
                                        mode in ["MUC_Errors", "Errors"]
                                        and mention_pair[2] == 1
                                    ):
                                        matching_result.errors[
                                            "Missing_Role_Filler"
                                        ] += 1
                                        matching_result.transformations.append((role_name, mention_pair[0], ["Introduce_Missing_Role_Filler"], mention_pair[1][0]))
                                        matching_result.missing_rfs.append(
                                            (
                                                template_pair[0],
                                                role_name,
                                                mention_pair[1]
                                            )
                                        )

                                        matching_result.errors[
                                            "General_Spurious_Role_Filler"
                                        ] += 1
                                        matching_result.spurious_rfs.append(
                                            (
                                                template_pair[0],
                                                role_name,
                                                mention_pair[0],
                                            )
                                        )
                                        matching_result.transformations.append(("", "", ["Remove_Spurious_Role_Filler"], ""))
                            if matching_result.valid and (
                                rolewise_result is None
                                or matching_result > rolewise_result
                            ):
                                rolewise_result = matching_result
                            
                    pairwise_result = Result.combine(pairwise_result, rolewise_result)
            result = Result.combine(result, pairwise_result)
        
        return result

    best_result = None
    best_matching = None
    for template_matching in template_matches(predicted_templates, gold_templates):
        result = analyze_template_matching(template_matching)
        if result.valid and (best_result is None or result > best_result):
            best_result = result
            best_matching = template_matching

    def handle_spurious_rfs(best_result, best_matching):

        remove_sprfs = []

        for ridx, sprf in enumerate(best_result.spurious_rfs):

            pred_template, pred_role_name, pred_mention = sprf

            transform_idx = best_result.transformations.index(("", "", ["Remove_Spurious_Role_Filler"], ""))

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
                                best_result.transformations[transform_idx] = (pred_role_name, pred_mention, ["Alter_Role"], role_name)
                                best_result.errors["Within_Template_Incorrect_Role"] += 1
                                remove_sprfs.append(ridx)
                                error_found = True
                                break
                            else:
                                best_result.transformations[transform_idx] = (pred_role_name, pred_mention, ["Remove_Duplicate_Role_Filler"], None)
                                best_result.errors["Duplicate_Role_Filler"] += 1
                                error_found = True
                                break

                    break

            if error_found:
                continue
            else:

                for role_name in role_names:
                    gold_mention_lst = []
                    gold_template_idxs = []
                    i = 0
                    for gold_template in gold_templates:
                        if (
                            matched_gold_template != None
                            and gold_template != matched_gold_template
                        ):
                            temp_lst = [
                                mention
                                for corefs in gold_template[role_name]
                                for mention in corefs
                            ]
                            gold_mention_lst += temp_lst
                            gold_template_idxs += [i]*len(temp_lst)
                        i += 1
                    try:
                        idx = gold_mention_lst.index(pred_mention)
                        gold_idx = gold_template_idxs[idx]
                        if pred_role_name != role_name:
                            best_result.transformations[transform_idx] = (pred_role_name, pred_mention, ["Alter_Role", "Remove_Cross_Template_Spurious_Role_Filler"], role_name, gold_templates[gold_idx])
                            best_result.errors["Wrong_Template + Wrong_Role"] += 1
                        else:
                            best_result.transformations[transform_idx] = (pred_role_name, pred_mention, ["Remove_Cross_Template_Spurious_Role_Filler"], None, gold_templates[gold_idx])
                            best_result.errors["Wrong_Template_For_Role_Filler"] += 1
                        error_found = True
                        break
                    except:
                        continue

                if not error_found:
                    best_result.transformations[transform_idx] = (pred_role_name, pred_mention, ["Remove_Unrelated_Spurious_Role_Filler"], None)
                    best_result.errors["Spurious_Role_Filler"] += 1

        
        for r in remove_sprfs:
            best_result.spurious_rfs[r] = "Removed"

        best_result.spurious_rfs = [sprf for sprf in best_result.spurious_rfs if sprf != "Removed"]

    handle_spurious_rfs(best_result, best_matching)
    best_result.compute(best_matching, docid)

    return best_result, best_matching


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

        data.append((docid, (pred_templates, gold_templates)))

    return data, documents

# instantiation of decorator function
@profile
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

    output_file.write("\nANALYZING DATA AND APPLYING TRANSFORMATIONS ...")

    total_result_before = Result()

    for docid, pair in tqdm(data, desc="Analyzing Data and Applying Transformations: "):
        output_file.write("\n\n\t------------------\n\n")
        output_file.write("DOCID: " + str(docid) + "\n\n")
        output_file.write("Comparing:")
        output_file.write(
            "\n"
            + summary_to_str(pair[0], mode)
            + "\n -to- \n"
            + summary_to_str(pair[1], mode)
        )
        result, matching = analyze(docid, *pair, mode, scoring_mode, verbose)

        for idx, template_pair in enumerate(matching):
            output_file.write("\n\n\t---\n\n")
            output_file.write("Template Pair " + str(idx + 1) + ": \nMatching:")
            str_t_0 = summary_to_str([template_pair[0]], mode)[11: ] if template_pair[0] != None else summary_to_str([template_pair[0]], mode)[9: ]
            str_t_1 = summary_to_str([template_pair[1]], mode)[11: ] if template_pair[1] != None else summary_to_str([template_pair[1]], mode)[9: ]
            output_file.write(
                "\n"
                + str_t_0
                + "\n -to- \n"
                + str_t_1
        )

        output_file.write("\n\n" + result.__str__(verbosity=2))
        total_result_before = Result.combine(total_result_before, result)

    total_result_before.update_stats()
    output_file.write(
        "\n\n************************************\nTotal Result Before Transformation : \n************************************"
    )
    output_file.write("\n\n" + total_result_before.__str__(verbosity=3))

    if analyze_transformed:
        output_file.write("\n\nANALYZING TRANSFORMED DATA ...")

        total_result_after = Result()

        for docid, pair in tqdm(total_result_before.transformed_data, desc="Analyzing Transformed Data: "):
            output_file.write("\n\n\t------------------\n\n")
            output_file.write("DOCID: " + str(docid) + "\n\n")
            output_file.write("Comparing:")
            output_file.write(
                "\n"
                + summary_to_str(pair[0], mode)
                + "\n -to- \n"
                + summary_to_str(pair[1], mode)
            )
            result, matching = analyze(docid, *pair, mode, scoring_mode, verbose)

            for idx, template_pair in enumerate(matching):
                output_file.write("\n\n\t---\n\n")
                output_file.write("Template Pair " + str(idx + 1) + ": \nMatching:")
                str_t_0 = summary_to_str([template_pair[0]], mode)[11: ] if template_pair[0] != None else summary_to_str([template_pair[0]], mode)[9: ]
                str_t_1 = summary_to_str([template_pair[1]], mode)[11: ] if template_pair[1] != None else summary_to_str([template_pair[1]], mode)[9: ]
                output_file.write(
                    "\n"
                    + str_t_0
                    + "\n -to- \n"
                    + str_t_1
            )

            output_file.write("\n\n" + result.__str__(verbosity=2))
            total_result_after = Result.combine(total_result_after, result)

        total_result_after.update_stats()
        output_file.write(
            "\n\n************************************\nTotal Result After Transformation : \n************************************"
        )
        output_file.write("\n\n" + total_result_after.__str__(verbosity=3))

    output_file.close()
    print("Time: " + str(time.time() - start))




if __name__ == "__main__":
    main()
