from evaluate import load

# make 2 dictionaries to hold references and predictions
def compute_wer(file_path):
    # initialize dictionaries to hold references and predictions
    references = {}
    predictions = {}

    with open(file_path, "r") as file:
        lines = file.readlines()
        
        for line in lines:
            #split line into label and text using ":" as the 
            #separator, and only split 1 time
            label, text = line.split('=', 1)

            # extract the numbers
            number = int(label.split("_")[-1])

            # strip leading and trailing white spaces
            text = text.strip().strip('"')
            
            # store the text in references or predictions dictionary 
            if label.startswith("reference"):
                references[number] = text
            if label.startswith("prediction"):
                predictions[number] = text



    wer_metric = load("wer")
    wer_scores = []

    for num in sorted(references.keys()):
        if num in predictions:
            wer = wer_metric.compute(references=[references[num]], predictions=[predictions[num]])
            wer_scores.append(wer)
            print(f'WER score for prediction {num}: {wer}')

    # compute average wer score
    ave_wer_score = sum(wer_scores)/len(wer_scores)
    print(f'average word error rate is: {ave_wer_score}')

compute_wer("references_predictions.txt")



