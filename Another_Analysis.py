# Modes: MUC score only, MUC + errors, errors only

def from_file():
    pass


def analyze(predicted_templates, gold_templates, mode = "MUC_errors"):

    def template_matches(predicted_templates, gold_templates):
        if len(predicted_templates) == 0: yield []
        else:
            for matching in template_matches(predicted_templates[1:], gold_templates):
                yield [(predicted_templates[0], None)]+matching
            for i in range(len(gold_templates)):
                for matching in template_matches(predicted_templates[1:], gold_templates[:i]+gold_templates[i+1:]):
                    yield [(predicted_templates[0], gold_templates[i])] + matching
                

    class Matching:
        def __init__(self):
            self.valid = True
            self.stats = {}
            self.error_score = 0

        def __gt__(self, other):
            if not other.valid: return True
            if self.stats["total"]["f1"] != other.stats["total"]["f1"]: 
                return self.stats["total"]["f1"] > other.stats["total"]["f1"]
            return self.error_score < other.error_score

        def compute():
            """Generate the log and transformations for this matching"""
            pass


    def analyze_template_matching(template_matching):

        for pair in template_matching:
            if mode in ["MUC_only", "MUC_errors"] and pair[0]["incident_type"] != pair[1]["incident_type"]: return Matching
            

    #best_matching = Matching()
    for template_matching in template_matches(predicted_templates, gold_templates):
        print(template_matching)
        #matching = analyze_template_matching(template_matching)
        #if matching > best_matching: best_matching = matching

    #best_matching.compute()
    #return best_matching


analyze(list(range(4)), list(range(4)))

