# Automatic Error Analysis for Document-level Information Extraction

Software for an ACL 2022 Main paper (Link to be posted) 

We build on the work of Kummerfeld and Klein (2013) to propose a transformation-based framework for automating error analysis in document-level event and (N-ary) relation extraction. From the output, one can further generate graph like below:

![An example image](./image/image1.png?raw=true)

## INSTALLATION

Requires the use of Python 3.6 and above. Please install the following packages:

json

re

argparse

textwrap

copy

numpy

tqdm

psutil

os


## USAGE 

Error_Analysis.py script command line arguments:

  -h, --help            

    show this help message and exit

  -i INPUT_FILE, --input_file INPUT_FILE            

    The path to the input file given to the system

  -v, --verbose         

    Increase output verbosity

  -at, --analyze_transformed            

    Analyze transformed data

  -s {all,msp,mmi,mat}, --scoring_mode {all,msp,mmi,mat}

    Choose scoring mode according to MUC:
    
    1. all - All Templates
    
    2. msp - Matched/Spurious
    
    3. mmi - Matched/Missing
    
    4. mat - Matched Only

  -m {MUC_Errors,Errors}, --mode {MUC_Errors,Errors}

    Choose evaluation mode:
    
    1. MUC_Errors - MUC evaluation with added constraint of incident_types of templates needing to match
    
    2. Errors - General evaluation with no added constraints

  -o OUTPUT_FILE, --output_file OUTPUT_FILE            

    The path to the output file the system writes to

  -j OUTPUT_JSON, output_json OUTPUT_JSON     

    The path to the output file the system writes to as JSON


​                        
## EXAMPLE:

- For MUC data:

``python3 Error_Analysis.py -i "model_preds.out" -o "err_file.out" --verbose -s all -m "MUC_Errors" -at``

- For other datasets

``python3 Error_Analysis.py -i "model_preds.out" -o "err_file.out" --verbose -s all -m "Errors" -at``

Remember to change the global variable **role_names** in the Error_Analysis.py script to match the roles associated with your dataset.
