import pandas as pd
import numpy as np

import jiwer
from copy import deepcopy

import argparse

def calculate_metrics(data_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    '''
    Calculates word error rate (WER) per sentence 
    
    WER = (S + D + I)/N
    
    where 
    S = substitutions
    D = deletions
    I = insertions
    N = number of words
    
    Here S,D,I are calculated from an optimal string alignment
    between the reference and hypothesis
        
    @param data_df: pandas dataframe with columns ["reference", "hypothesis"]
    @param verbose: if True, prints alignment 
    '''
    
    results_df = deepcopy(data_df)

    sentence_wer = []
    
    for index, row in results_df.iterrows():
        output = jiwer.process_words(reference = row['reference'],
                                     hypothesis = row['hypothesis'])
        sentence_wer.append(output.wer)
        
        if verbose:
            output_string = jiwer.visualize_alignment(output, 
                                                      show_measures = False)
            start_idx = output_string.find("REF")

            print("sentence pair: {}".format(index))
            print(output_string[start_idx:])
            print("SUB={} DEL={} INSERT={}, WER:{:.3f} \n".format(output.substitutions, 
                                                      output.deletions,
                                                      output.insertions,
                                                      output.wer))
    
    results_df["WER"] = sentence_wer
    
    return results_df


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file", help = "input csv file", type = str)
    parser.add_argument("-output_file", help = "output csv file", type = str)
    parser.add_argument("--verbose", default = False, help = "verbose output", type = bool)
    args = parser.parse_args()
    
    data = pd.read_csv(args.input_file, index_col = 0)
    
    results = calculate_metrics(data, verbose = args.verbose)
    
    results.to_csv(args.output_file)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
