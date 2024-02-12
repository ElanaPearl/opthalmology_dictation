import pandas as pd
import argparse

from copy import deepcopy
import jiwer


def calculate_keyword_error_rate(df: pd.DataFrame, keyword_list: list, verbose: bool = False) -> pd.DataFrame:
    """
    Compute the keyword error rate per sentence and overall for the given dictation data.

    Keyword Error Rate (KER) is defined as (F + M) / N
     where F = number of falsely recognized keywords
           M = number of missing keywords
           N = total number of keywords

    @pandas df: dataframe with columns ["reference", "hypothesis"], giving ground-truth sentences and their dictations
    @param keyword_list: list of keywords to track the error of in the data
    @param verbose: if True, prints further details
    @return: dataframe with F, M, and N values as well as the KER for each sentence and overall

    @note: KER equation based on paper - Park, Youngja, et al. "An empirical analysis of word error rate
    and keyword error rate." Interspeech. 2008.
    """
    data_df = deepcopy(df)
    keyword_set = set(keyword_list)

    keyword_counts = []
    # for each row (typically per sentence), calculate number of missing/incorrect keywords
    for index, row in data_df.iterrows():
        output = jiwer.process_words(reference=row['reference'],
                                     hypothesis=row['hypothesis'])

        references = output.references
        hypothesis = output.hypotheses
        alignment = output.alignments

        total_count = sum([w in keyword_set for w in references[0]])  # todo check list of list
        missing_count = 0
        incorrect_count = 0

        # for each set of chunks - if only parsing sentences, there should only be one set
        for gt, hp, chunks in zip(references, hypothesis, alignment):
            if len(chunks) == 1 and chunks[0].type == "equal":  # fully correct dictation
                continue

            # for each chunk of sentence
            for alignment_chunk in chunks:
                if alignment_chunk.type == 'equal':  # skip parts of sentence that are equal
                    continue

                # get corresponding ground truth words
                alignment_gt = gt[alignment_chunk.ref_start_idx:alignment_chunk.ref_end_idx]

                for gt_word in alignment_gt:
                    # check if missing
                    if gt_word in keyword_set and alignment_chunk.type == 'delete':
                        missing_count += 1
                    elif gt_word in keyword_set and alignment_chunk.type == 'substitute':  # or transcribed incorrectly
                        incorrect_count += 1

        keyword_counts.append([missing_count, incorrect_count, total_count])

    # add counts to dataframe and calculate KER score
    count_columns = ['num_missing', 'num_incorrect', 'num_total']
    keyword_counts_df = pd.DataFrame(keyword_counts, columns=count_columns, dtype=int)
    keyword_counts_df.loc['total', :] = keyword_counts_df.sum()
    keyword_counts_df['KER'] = (keyword_counts_df['num_missing'] +
                                keyword_counts_df['num_incorrect']) / keyword_counts_df['num_total']

    keyword_counts_df[count_columns] = keyword_counts_df[count_columns].astype(int)

    # print KER scores for each sentence
    if verbose:
        print(data_df.merge(keyword_counts_df.iloc[:-1, -1], left_index=True, right_index=True))
        print('\n')

    print(f'Overall KER: {keyword_counts_df.loc['total', 'ker']}')

    return keyword_counts_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="input csv file", type=str)
    parser.add_argument("-o", "--output_file", default="./output.csv", help="output csv file", type=str)
    parser.add_argument("--verbose", default=False, help="verbose output", action='store_true')
    args = parser.parse_args()

    input_df = pd.read_csv(args.input_file, index_col=0)

    results = calculate_keyword_error_rate(input_df, ['sentence', 'deletion', 'substitution'], verbose=args.verbose)
    results.to_csv(args.output_file, sep='\t')

