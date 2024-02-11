# opthalmology_dictation
Dictation of physician notes of Optos fundus photography images into formal text to be used in EHR

## To create conda environment
```bash
conda env create -f env.yml
conda activate optho
```

## Calculate Optos Fundus key-words using TF-IDF
1) Download [MeDAL](https://arxiv.org/abs/2012.13978) data set from [HuggingFace](https://huggingface.co/datasets/medal/blob/main/data/pretrain_subset.zip). This dataset is a large medical text used for pre-training LLMs on medical data. This serves as the background distribution for medical terms that are not necessarily specific to Optos fundus. Note, this download requires 4G of memory.

First download a 2G zip file of all the pretraining data for MeDAL
```bash
wget --directory-prefix data https://huggingface.co/datasets/medal/resolve/main/data/pretrain_subset.zip
```

Then unzip only the test subset of this dataset (it is 1M examples and ~2G which is sufficient for our purposes)
```bash
unzip -p data/pretrain_subset.zip test.csv > data/background_tf_idf.csv
```

Now, run the TF-IDF detection on the long-form dictation text with MeDAL as the background
```bash
python identify_keywords_in_text.py
```

Finally, delete the original zip file to save memory. Feel free to also delete the background_tf_idf.csv to save memory
```bash
rm data/pretrain_subset.zip
rm data/background_tf_idf.csv
```