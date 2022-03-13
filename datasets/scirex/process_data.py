import json
import re
import sys

import numpy as np
import matplotlib

CASE_SENSITIVE = False
TO_LOWER_CASE = True
# REPLACE_UNDERSCORE = True
IGNORE_SCORE = True
# IGNORE_TASK = False
TOTAL_TYPES = 5 - (1 if IGNORE_SCORE else 0)  # - (1 if IGNORE_TASK else 0)
REMOVE_LONGER_THAN = 25

# Additional preprocess: removed all sub_templates, removed examples without templates.
# see {@code article_restore} function

files = [open("./release/dev.jsonl"), open("./release/test.jsonl"), open("./release/train.jsonl")]

# Stats on how many entities are not present in the document, by their annotation
no_show = {}
# Stats on how many entities are not present in the document, for each relation
removal = {}
# Total dev/verification/training examples
example_counter = 0
# Total relations all examples have
relation_counter = 0
# Counters on how many relations are completely removed
emptied_relations = 0
# Average earliest mention
earlist = []
# words
n_words = []
# char
n_char = []
n_char500 = []
# Sub template counter
n_sub = 0
# Actual template counter
n_template = 0


def update_progress(progress):
    barLength = 10  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


# def remove_underscore(s):
#     """Replace _ with space and avoid repeating space"""
#     return " ".join(" ".join(str(s).split('_')).split())


def new_file_addr(name, pretty=False):
    """
    Convert release data address to new addresses
    """
    name = re.sub(r"release/", "processed/" if not pretty else "processed/pretty_", name)
    name = re.sub(r".jsonl", ".json", name)
    return name


def article_restore(words, suffix=''):
    words = filter(lambda x: len(x) <= REMOVE_LONGER_THAN, words)
    text = " ".join(words)
    text = re.sub(r" ([^a-zA-Z0-9\[{(\-‘])", r"\1", text)
    text = re.sub(r"\[ ", r"[", text)
    text = re.sub(r"([{(‘]) ", r"\1", text)
    text = re.sub(r"\) \.", r").", text)
    text = re.sub(r" - ", r"-", text)
    text = re.sub(r"–", r"-", text)
    text = re.sub(r"[‘’]", r"'", text)
    text = re.sub(r"[“”]", r'"', text)
    if TO_LOWER_CASE:
        text = text.lower()
    return text + suffix


def is_sub_relation(relation1, relation2):
    for key in relation1.keys():
        for item in np.array(relation1[key]).flatten():
            if item not in np.array(relation2[key]).flatten():
                return False
    return True


def non_repeating_add(current_list, new_relation):
    global n_sub
    for old_relation in current_list[:]:
        if is_sub_relation(old_relation, new_relation):
            current_list.remove(old_relation)
            current_list.append(new_relation)
            n_sub += 1
            return current_list
        if is_sub_relation(new_relation, old_relation):
            n_sub += 1
            return current_list
    current_list.append(new_relation)
    return current_list


def sorted_by_second(data):
    return sorted(data, key=lambda tup: tup[1])


def find_all_coref(words, coref_list):
    result = {}
    for coref in coref_list:
        start, end = coref[0], coref[1]
        mention = words[start:end]
        if len(mention[-1]) == 1 and not mention[-1].isalnum():
            mention = mention[:-1]
        while len(mention[0]) == 1 and not mention[0].isalnum():
            mention = mention[1:]
        if len(mention) == 0:
            continue
        pos = len(article_restore(words[:start])) + 1
        if start == 0:
            pos -= 1
        if article_restore(mention) not in result:
            result[article_restore(mention)] = pos
    return sorted_by_second([[i, result[i]] for i in list(result.keys())])


if __name__ == '__main__':
    for f in files:
        numbering = 0
        new_f = open(new_file_addr(f.name), 'w')
        print('\nProcessing', f.name)
        new_f_pretty = open(new_file_addr(f.name, True), 'w')
        # JsonLine file: each line contains a json.
        lines = f.readlines()
        for i, line in enumerate(lines):
            update_progress(i / len(lines))

            example = json.loads(line)
            example_counter += 1
            new_example = {}
            new_relation_list = []

            # Stats
            n_words.append(len(example['words']))
            n_char.append(len(article_restore(example['words']).lower()))
            n_char500.append((len(article_restore(example['words'][:500]).lower())))

            for relation in list(example['n_ary_relations']):
                # Stats
                relation_counter += 1
                no_show_slot_count = 0

                if IGNORE_SCORE and 'score' in relation:
                    del relation['score']
                # if IGNORE_TASK and 'Task' in relation:
                #         del relation['Task']

                for entity_type, entity_name in list(relation.items()):
                    if entity_type.lower() == 'score' and IGNORE_SCORE:
                        continue
                    # if entity_type == 'Task' and IGNORE_TASK:
                    #     continue
                    # entity_name_new = remove_underscore(entity_name) if REPLACE_UNDERSCORE else entity_name

                    # if (not CASE_SENSITIVE and str(entity_name_new).lower() not in article_restore(
                    #         example['words']).lower()) or \
                    #         (CASE_SENSITIVE and str(entity_name_new) not in article_restore(example['words'])):
                    if not example['coref'][entity_name]:
                        relation[entity_type] = []
                        no_show_slot_count += 1
                        if entity_type in no_show:
                            no_show[entity_type] += 1
                        else:
                            no_show[entity_type] = 1
                    else:
                        relation[entity_type] = [[]]
                        mentions = find_all_coref(example['words'], example['coref'][entity_name])
                        assert mentions != []
                        for mention_pair in mentions:
                            earlist.append(mention_pair[1]

                                           # re.search(re.escape(mention_pair.lower()) + "[^a-z]",
                                           #           article_restore(
                                           #               example['words'], ' ').lower()).start()
                                           # if not CASE_SENSITIVE else re.search(
                                           #     re.escape(mention_pair) + "[^a-z]",
                                           #     article_restore(example['words'], ' ')).start()
                                           )
                            # if REPLACE_UNDERSCORE:
                            #     relation[entity_type] = entity_name_new
                            if TO_LOWER_CASE:
                                mention_pair[0] = mention_pair[0].lower()
                            relation[entity_type][0].append(mention_pair)

                if no_show_slot_count < TOTAL_TYPES:
                    new_relation_list = non_repeating_add(new_relation_list, relation)

                if no_show_slot_count > 0 and no_show_slot_count in removal:
                    removal[no_show_slot_count] += 1
                elif no_show_slot_count > 0:
                    removal[no_show_slot_count] = 1
            if not new_relation_list:
                emptied_relations += 1
            else:
                n_template += len(new_relation_list)
                new_example['docid'] = example['doc_id'] + '-' + str(numbering)
                numbering += 1
                new_example['doctext'] = article_restore(example['words'])
                new_example['templates'] = new_relation_list
                new_f.write(json.dumps(new_example) + '\n')
                new_f_pretty.write(json.dumps(new_example, indent=4, sort_keys=True) + "\n")

        new_f.flush()
        new_f_pretty.flush()
        new_f.close()
        new_f_pretty.close()

    print('[CASE_SENSITIVE]', CASE_SENSITIVE, '[IGNORE_SCORE]'
          , IGNORE_SCORE)
    print()
    print('Total relations:', relation_counter)
    print('Gold entities no show per relation:', removal)
    print('In other words, entities actually present per relation and their ratio = :',
          {(TOTAL_TYPES - k): v / relation_counter for k, v in removal.items()})
    print('Gold entities no show by their type', no_show)
    print('Gold entities no show by their type, percentage', {k: v / relation_counter for k, v in no_show.items()})
    print('Global missing entities', sum([v for k, v in no_show.items()]), '/', relation_counter * TOTAL_TYPES)
    print('After processing,', emptied_relations, 'example of', example_counter,
          'examples will have no relations')
    print('On average, each document has', str(np.average(n_words)), 'words, or ', str(np.average(n_char)), 'chars')
    print('The first 500 words on average have', str(np.average(n_char500)), 'characters')
    print('Removed', n_sub, 'relations because they are subset of another relation')
    print('There are total', n_template, 'templates/relations for all', example_counter - emptied_relations, 'examples')
