from setup import *
from preprocess import *

subtitle_lookup = pd.read_table("../data/knnw_en_sub_edit.csv", sep = ";", header=0)
length = len(subtitle_lookup)
for i in range(length):
    subtitle_item = subtitle_lookup.iloc[i, 3]
    subtitle_lookup.iloc[i, 3] = '"' + knnw_process_string(subtitle_item) + '"'

newpath = "../data/processed"    
subtitle_lookup.to_csv(newpath + ".csv", index=False, sep=";", quoting=csv.QUOTE_NONE)
subtitle_lookup.to_csv(newpath + "_comma.csv", index=False, quoting=csv.QUOTE_NONE)

# kenlm corpus generation code
kenlm_data = subtitle_lookup
sentences = []
for sentence in kenlm_data['Text']:
    clean_sentence = sentence.replace('"', '') + " "
    print(clean_sentence)
    sentences.append(clean_sentence)

corpus_path = os.path.expanduser('~/autosubs/data/kenlm_knnw.txt')
with open(corpus_path, "w") as outfile:
        outfile.writelines(sentences)
