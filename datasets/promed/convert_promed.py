import os, json, re

# Used for converting the raw ProMED data to JSON format

def preprocess_string(doc_str):
  return re.sub(r' +', ' ', re.sub(r'(\-\-+|\.\.+)', "", doc_str.lower()))

def convert_data(text_dir, ans_dir, mode, percent_dev=0):

    slot_names = ['Story', 'ID', 'Date', 'Event', 'Status', 'Containment', 'Country', 'Diseases', 'Victims']
    role_names = ['Status', 'Country', 'Disease', 'Victims']

    docs = os.listdir(text_dir)
    
    if mode == "train":
      out_f = open("train.json", "w")
      pout_f = open("pretty_train.json", "w")

    if mode == "test":
      if percent_dev == 0:
        raise Exception("You need to decide percentage of test set to be used as dev set.")
      out_f = open("test.json", "w")
      pout_f = open("pretty_test.json", "w")
      dev_f = open("dev.json", "w")
      pdev_f = open("pretty_dev.json", "w")
      missing_country_dev = []
    
    missing_country = []
    dev_count = int((percent_dev/100)*len(docs))

    for doc in docs:

      new_doc = {}
      parts = doc.split(".")
      new_doc["docid"] = "0-PROMED-" + "".join(parts[:2])

      text_f = open(text_dir + "/" + doc, "r")

      new_doc["doctext"] = preprocess_string("".join(line[:-1] for line in text_f))[1:]

      temp_f = open(ans_dir + "/" + doc + ".annot", "r")

      templates = []
      
      template = {}
      newline_count = 0
      new_template = False
      current_role = ""

      for line in temp_f:
        if newline_count == 2 or new_template:
          if template != {}:
            templates.append(template)
          template = {}
          newline_count = 0
          new_template = False
          current_role = ""

        l = line[:-1]
        if line == "\n":
          newline_count += 1
          continue
        elif l.split(" ")[0] == "Bytespans":
          break
        else:
          data_line = [word.strip() for word in l.split(":")]
          
          if data_line[0] == 'Event' and data_line[1] == "not an outbreak":
            new_template = True

          if data_line[0] in role_names:
            current_role = data_line[0]
            if data_line[1] == "-----":
              template[current_role] = []
            else:
              if current_role == "Status":
                template[current_role] = data_line[1].strip()
              else:
                template[current_role] = [[[preprocess_string(mention.strip())] for mention in data_line[1].split("/")]]
                if current_role == "Country" and preprocess_string(data_line[1].strip()) not in new_doc["doctext"]:
                  if mode == "test" and dev_count > 0:
                    missing_country_dev.append(doc)
                  else:
                    missing_country.append(doc)
                  
          else:
              if data_line[0] in slot_names:
                current_role = data_line[0]
              else:
                if current_role in role_names:
                  template[current_role].append([[preprocess_string(mention.strip())] for mention in data_line[0].split("/")])

      new_doc["templates"] = templates

      if mode == "test" and dev_count > 0:
        dev_f.write(json.dumps(new_doc) + "\n")
        pdev_f.write(json.dumps(new_doc, indent=4) + "\n")
        dev_count -= 1
      else:
        out_f.write(json.dumps(new_doc) + "\n")
        pout_f.write(json.dumps(new_doc, indent=4) + "\n")

    if mode == "train":
      print("\ntrain.json - No. of templates in which the Country is not present in the document: " + str(len(missing_country)) + "\n")
      for d in missing_country:
        print(d)
    if mode == "test":
      print("\ndev.json - No. of templates in which the Country is not present in the document: " + str(len(missing_country_dev)) + "\n")
      for d in missing_country_dev:
        print(d)
      print("\ntest.json - No. of templates in which the Country is not present in the document: " + str(len(missing_country)) + "\n")
      for d in missing_country:
        print(d)
      print("\n")
    

text_dir = "tuning-zoned"
ans_dir = "tuning-anskey"
convert_data(text_dir, ans_dir, "train")

text_dir = "test-zoned"
ans_dir = "test-anskey"
convert_data(text_dir, ans_dir, "test", 10)