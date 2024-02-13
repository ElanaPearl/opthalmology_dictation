from evaluate import load
import re

# make 2 dictionaries to hold references and predictions
def compute_cer(file_path):
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
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # store the text in references or predictions dictionary 
            if label.startswith("reference"):
                references[number] = text
            if label.startswith("prediction"):
                predictions[number] = text



    cer_metric = load("cer")
    cer_scores = []

    for num in sorted(references.keys()):
        if num in predictions:
            cer = cer_metric.compute(references=[references[num]], predictions=[predictions[num]])
            cer_scores.append(cer)
            # print(f'CER score for prediction {num}: {cer}')
            print(f'Character accuracy {num}: {1-cer}')

    # compute average wer score
    ave_cer_score = sum(cer_scores)/len(cer_scores)
    print(f'average character error rate is: {ave_cer_score}')

compute_cer("./Character_error_rate/references_predictions.txt")



