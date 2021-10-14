# auto-err-template-fill

Requires the use of Python 3.6 and above. Please install the following packages:

1.json
2.re
3.argparse
4.textwrap
5.copy
6.numpy
7.tqdm
8.psutil
9.os

Error_Analysis.py script command line arguments:

  -h, --help            show this help message and exit
  
  -i INPUT_FILE, --input_file INPUT_FILE
  
                        The path to the input file given to the system
                        
  -v, --verbose         Increase output verbosity
  
  -at, --analyze_transformed
  
                        Analyze transformed data
                        
  -s {all,msp,mmi,mat}, --scoring_mode {all,msp,mmi,mat}
  
                        Choose scoring mode according to MUC:
                        
                        all - All Templates
                        
                        msp - Matched/Spurious
                        
                        mmi - Matched/Missing
                        
                        mat - Matched Only
                        
  -m {MUC_Errors,Errors}, --mode {MUC_Errors,Errors}
  
                        Choose evaluation mode:
                        
                        MUC_Errors - MUC evaluation with added constraint of incident_types of templates needing to match
                        
                        Errors - General evaluation with no added constraints
                        
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
  
                        The path to the output file the system writes to
                        
Usage:

- For MUC data:

``python3 Error_Analysis.py -i "model_preds.out" -o "err_file.out" --verbose -s all -m "MUC_Errors" -at``

- For other datasets

``python3 Error_Analysis.py -i "model_preds.out" -o "err_file.out" --verbose -s all -m "Errors" -at``

Remember to change the global variable **role_names** in the Error_Analysis.py script to match the roles associated with your dataset.
