import pandas as pd

from copy import deepcopy
import jiwer


def calculate_keyword_error_rate(data_df: pd.DataFrame, keyword_list: list, verbose: bool = False) -> pd.DataFrame:
    """
    Compute the keyword error rate per sentence and overall for the given dictation data.

    Keyword Error Rate (KER) is defined as (F + M) * 100 / N
     where F = number of falsely recognized keywords
           M = number of missing keywords
           N = total number of keywords

    @pandas: dataframe with columns ["reference", "hypothesis"], giving ground-truth sentences and their dictations
    @param keyword_list: list of keywords to track the error of in the data
    @param verbose: if True, prints further details
    @return:

    @note: KER equation based on paper - Park, Youngja, et al. "An empirical analysis of word error rate
    and keyword error rate." Interspeech. Vol. 2008. 2008.
    """
    results_df = deepcopy(data_df)
    keyword_set = set(keyword_list)

    keyword_counts = []

    # for each row (typically per sentence), calculate number of missing/incorrect keywords
    for index, row in results_df.iterrows():
        output = jiwer.process_words(reference=row['reference'],
                                     hypothesis=row['hypothesis'])

        references = output.references
        hypothesis = output.hypotheses
        alignment = output.alignments

        total_count = sum([w in keyword_set for w in references[0]])  # todo check list of list
        missing_count = 0
        incorrect_count = 0

        for gt, hp, chunks in zip(references, hypothesis, alignment):
            if len(chunks) == 1 and chunks[0].type == "equal":
                continue

            # for each non-equal chunk
            for alignment_chunk in chunks:
                if alignment_chunk.type == 'equal':
                    continue
                alignment_gt = gt[alignment_chunk.ref_start_idx:alignment_chunk.ref_end_idx]

                for gt_word in alignment_gt:
                    # check if missing
                    if gt_word in keyword_set and alignment_chunk.type == 'delete':
                        missing_count += 1
                    elif gt_word in keyword_set and alignment_chunk.type == 'substitute': # or transcribed incorrectly
                        incorrect_count += 1

        keyword_counts.append([missing_count, incorrect_count, total_count])

    keyword_counts_df = pd.DataFrame(keyword_counts, columns=['num_missing', 'num_incorrect', 'num_total'])
    keyword_counts_df.loc['total', :] = keyword_counts_df.sum()
    keyword_counts_df['ker'] = (keyword_counts_df['num_missing'] +
                                keyword_counts_df['num_incorrect']) / keyword_counts_df['num_total']

    return keyword_counts_df


if __name__ == "__main__":
    input_df = pd.read_csv("example_data.csv", index_col = 0)

    keyword_counts_df = calculate_keyword_error_rate(input_df, ['sentence', 'deletion', 'substitution'])