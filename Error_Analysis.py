import copy

def all_matchings(a, b):
    """returns a list of all matchings, where each matching is a dictionary with keys "pairs," "unmatched_gold," "unmatched_predicted." 
    A matching is a set of pairs (i,j) where i in range(a), j is in range(b), unmatched_predicted is a subset of range(a), 
    unmatched_gold is a subset of range(b), every element of range(a) occurs exactly once in unmatched_predicted 
    or in the first position of a pair, and every element of range(b) occurs exactly once in unmatched_gold 
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


def extract_span(predicted_mention, best_gold_mention):
    # extracting missing/extra parts of the spans that cause span errors
    # m - missing, e - extra
    spans = []
    if best_gold_mention != None:
        diff_1 = predicted_mention.span[0] - best_gold_mention.span[0]
        diff_2 = predicted_mention.span[1] - best_gold_mention.span[1]
        docid_str = "Doc ID: " + predicted_mention.doc_id
        if diff_1 > 0:
            chars = extract_span_diff(best_gold_mention.literal, diff_1, True)
            spans.append((docid_str, chars, "m"))
        elif diff_1 < 0:
            chars = extract_span_diff(predicted_mention.literal, -diff_1, True)
            spans.append((docid_str, chars, "e"))
        else:
            pass
        if diff_2 > 0:
            chars = extract_span_diff(predicted_mention.literal, diff_2, False)
            spans.append((docid_str, chars, "e"))
        elif diff_2 < 0:
            chars = extract_span_diff(best_gold_mention.literal, -diff_2, False)
            spans.append((docid_str, chars, "m"))
        else:
            pass
    return spans

# A single mention
class Mention:
    def __init__(self, doc_id, span, literal, result_type):
        self.doc_id = doc_id
        self.span = span  # a pair of indices delimiting a string in the doc
        self.literal = literal
        self.result_type = result_type

    @staticmethod
    def compare(predicted_mention, gold_mentions, role_name):  
        assert predicted_mention is None or gold_mentions is None or predicted_mention.result_type == gold_mentions.result_type, "only mention with the same result type can be compared"

        result = predicted_mention.result_type() if predicted_mention is not None else gold_mentions.result_type()
        if predicted_mention is not None and predicted_mention.doc_id == "30003" and gold_mentions is None:
            print("ABBBA")
        if gold_mentions is None:
            result.update("Spurious_Role_Filler", {"predicted_mention": predicted_mention, "role_name": role_name})
            return result

        if predicted_mention is None:
            result.update("Missing_Role_Filler", {"gold_mentions": gold_mentions, "role_name": role_name})
            return result

        result.update("Matched_Role_Filler", {"predicted_mention": predicted_mention, "gold_mentions": gold_mentions, "role_name": role_name})
        
        return result

    def from_doc(self):
        # doc_tokens = docs[self.doc_id]  # global variable docs - tokenized documents
        # start, end = self.span
        #return ' '.join(doc_tokens[start:end + 1])
        return self.literal

    def __str__(self):
        return str(self.span) + " - " + self.from_doc()

# An entity in a gold template, with a list of coref mentions
class Mentions:
    def __init__(self, doc_id, mentions, result_type):
        self.doc_id = doc_id
        self.mentions = mentions  # a list of coref mentions
        self.result_type = result_type
        assert all(
            mention.doc_id == self.doc_id for mention in mentions), "only mentions for the same doc can form a list"
        assert all(
            mention.result_type == self.result_type for mention in mentions), "only mentions with the same result type can form a list"

    def __str__(self):
        return "[" + ", ".join([str(mention) for mention in self.mentions]) + "]"

    @staticmethod
    def str_from_doc(role):
        result = ""
        for mention_or_mentions in role.mentions:
            if isinstance(mention_or_mentions, Mention):
                result+='('
                mention = mention_or_mentions
                result += mention.from_doc()
                if mention != role.mentions[-1]:
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
    def __init__(self, doc_id, mentions, gold, result_type):
        self.doc_id = doc_id
        self.mentions = mentions  # a list of Mention (not Gold) or Mentions (gold)
        self.gold = gold
        self.result_type = result_type
        assert all(
            mention.doc_id == self.doc_id for mention in self.mentions), "mentions must be for the same doc as its role"
        assert all(
            mention.result_type == self.result_type for mention in self.mentions), "mentions must have the same result type as its role"

    def __str__(self, outer=True):
        return "[" + ", ".join([str(mention) for mention in self.mentions]) + "]"

    @staticmethod
    def compare(predicted_role, gold_role, role, verbose=False):
        assert predicted_role is None or gold_role is None or predicted_role.result_type == gold_role.result_type, \
            "only roles with the same result type can be compared"

        best_result = predicted_role.result_type() if predicted_role is not None else gold_role.result_type()

        if gold_role is None:
            for mention in predicted_role.mentions:
                best_result = predicted_role.result_type.combine(best_result, Mention.compare(mention, None, role))
            return best_result

        if predicted_role is None:
            for mentions in gold_role.mentions:
                best_result = predicted_role.result_type.combine(best_result, Mention.compare(None, mentions, role))
            return best_result

        for matching in all_matchings(len(predicted_role.mentions), len(gold_role.mentions)):
            result = Role.compare_matching(matching, predicted_role, gold_role, role, verbose)
            if result > best_result: best_result = result
        return best_result

    @staticmethod
    def compare_matching(matching, predicted_role, gold_role, role, verbose=False):
        assert predicted_role is None or gold_role is None or predicted_role.result_type == gold_role.result_type, \
            "only roles with the same result type can be compared"
        result = predicted_role.result_type() if predicted_role is not None else gold_role.result_type()
        for i, j in matching["pairs"]:
           result = predicted_role.result_type.combine(result, Mention.compare(predicted_role.mentions[i], gold_role.mentions[j], role))
        for i in matching["unmatched_predicted"]:
           result = predicted_role.result_type.combine(result, Mention.compare(predicted_role.mentions[i], None, role))
        for i in matching["unmatched_gold"]:
            result = predicted_role.result_type.combine(result, Mention.compare(None, gold_role.mentions[i], role))
        return result

# A data structure containing structured information about an event in a document
class Template:
    def __init__(self, doc_id, roles, gold, result_type):
        self.doc_id = doc_id
        self.roles = roles  # a dictionary, indexed by strings, with Role values
        self.gold = gold
        self.result_type = result_type
        assert all(role.gold == self.gold for _, role in
                   self.roles.items()), "roles must be in the same format as its template"
        assert all(role.doc_id == self.doc_id for _, role in
                   self.roles.items()), "roles must be for the same doc as its template"
        assert all(role.result_type == self.result_type for _, role in
                   self.roles.items()), "roles must have the same result type as its templates"

    def __str__(self, outer=True):
        re = "Template" + ((" (gold)" if self.gold else " (predicted)") if outer else "") + ":"
        if outer: re += "\n - Doc ID: " + str(self.doc_id)
        for role_name, role in self.roles.items():
            re += "\n - " + role_name + ": " + role.__str__(False)
        return re

    @staticmethod
    def compare(predicted_template, gold_template, verbose=False):
        assert predicted_template is not None or gold_template is not None, "cannot compare None to None"

        if gold_template is None:
            result = predicted_template.result_type()
            result.update("Spurious_Template", {"predicted_template": predicted_template})
            return result

        if predicted_template is None:
            result = gold_template.result_type()
            result.update("Missing_Template",  {"gold_template": gold_template})
            return result

        assert predicted_template.result_type == gold_template.result_type, "only templates with the same result type can be compared"
        assert predicted_template.roles.keys() == gold_template.roles.keys(), "only templates with the same roles can be compared"

        result = predicted_template.result_type()
        result.update("Matched_Template", {"predicted_template": predicted_template, "gold_template": gold_template})
        for role_name in predicted_template.roles:
            comparison = Role.compare(predicted_template.roles[role_name], gold_template.roles[role_name], role_name, verbose)
            result = predicted_template.result_type.combine(result, comparison)
        return result

# Represents a list of templates extracted from a single document
class Summary:
    def __init__(self, doc_id, templates, gold, result_type):
        self.doc_id = doc_id
        self.templates = templates  # a list of Template
        self.gold = gold
        self.result_type = result_type
        assert all(template.gold == self.gold for template in 
                    self.templates), "summary must be in the same format as its templates"
        assert all(template.doc_id == self.doc_id for template in
                   self.templates), "summary must be for the same doc as its templates"
        assert all(template.result_type == self.result_type for template in
                   self.templates), "summary must have the same result type as its templates"

    def __str__(self, outer=True):
        re = "Summary" + ((" (gold)" if self.gold else " (predicted)") if outer else "") + ":"
        if outer: re += "\n - Doc ID: " + str(self.doc_id)
        for template in self.templates:
            re += "\n" + template.__str__(False)
        return re

    @staticmethod
    def compare(predicted_summary, gold_summary, verbose=False):
        """returns a Result object representing the comparision  
        between Summaries [predicted_summary] and [gold_summary]"""
        assert predicted_summary is not None or gold_summary is not None, "cannot compare None to None"
        assert predicted_summary is None or gold_summary is None or predicted_summary.result_type == gold_summary.result_type, "only summaries with the same result type can be compared"

        best_result = predicted_summary.result_type() if predicted_summary is not None else gold_summary.result_type()
        for matching in all_matchings(len(predicted_summary.templates), len(gold_summary.templates)):
            result = Summary.compare_matching(matching, predicted_summary, gold_summary, verbose)
            if result > best_result: best_result = result
        return best_result

    @staticmethod
    def compare_matching(matching, predicted_summary, gold_summary, verbose=False):
        assert predicted_summary is not None or gold_summary is not None, "cannot compare None to None"
        assert predicted_summary is None or gold_summary is None or predicted_summary.result_type == gold_summary.result_type, \
            "only summaries with the same result type can be compared"

        result = predicted_summary.result_type() if predicted_summary is not None else gold_summary.result_type()
        for i, j in matching["pairs"]:
            pair_result = Template.compare(predicted_summary.templates[i], gold_summary.templates[j], verbose)
            result = predicted_summary.result_type.combine(result, pair_result)
        for i in matching["unmatched_predicted"]:
            result = predicted_summary.result_type.combine(result, Template.compare(predicted_summary.templates[i], None))
        for i in matching["unmatched_gold"]:
            result = predicted_summary.result_type.combine(result, Template.compare(None, gold_summary.templates[i]))
        return result

class Result:
    """An object representing all data associated with a part or all 
    of the difference between some predicted data and some gold data."""
    def __init__(self): 
        pass
    
    def __str__(self): 
        pass

    def __gt__(self,other):
        pass

    @staticmethod
    def combine(result1, result2):
        """returns a Result representing the simultaneous occurrence of [result1] and [result2]"""
        pass

    def update(self, comparison_event):
        """updates [self] to include a [comparision_event]"""
        pass
