# SciREX dataset
    .
    ├── processed                        # preprocessed dataset
    │   ├── .          
    │   └── .           
    ├── release                          # Original dataset
    │   └── .         
    │   └── . 
    ├── process_data.py                  # Script used for processing the dataset
    ├── scirex_docids_event_n.json       # Optionally used for calculating micro-averaged F1 in scirex_eval.py
    ├── scirex_eval.py                   # Used for creating the file used as input of the error analysis tool
    └── span_to_temp.py                  # Used for creating the heuristic for clustering extracted entities 
                                         # for DyGIE++ into templates
