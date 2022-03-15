# ProMED dataset

### Available on request from Professor Ellen Riloff, University of Utah
    .
    ├── convert_promed.py                           # script used for processing the raw ProMED data into JSON form        
    ├── promed_docids_event_n.json                  # Optionally used for calculating micro-averaged F1 in promed_eval.py
    ├── promed_eval.py                              # Used for creating the file used as input of the error analysis tool
    └── promed_span_to_temp.py                      # Used for creating the heuristic for clustering extracted entities 
                                                    # for DyGIE++ into templates
