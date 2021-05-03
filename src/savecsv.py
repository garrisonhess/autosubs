from setup import *
from preprocess import *
import csv
def remove_chars(text):
    text = text.lower()

    null = 'null'
    text = re.sub(r'.*""', null, text)
    text = text.replace('?', '')
    text = text.replace('!', '')
    text = text.replace(',', '')
    text = text.replace('-', ' ')
    text = text.replace('"', '')
    text = text.replace("“", '')
    text = text.replace("”", '')
    text = text.replace('...', '')
    text = text.replace('é', 'e')
    text = text.replace('21', 'twenty one')
    text = text.replace('1200', 'twelve hundred')
    text = text.replace('20th', 'twentieth')
    text = text.replace('7:40', 'seven fourty')
    text = text.replace('8:42', 'eight fourty two')
    text = text.replace('1994', 'nineteen ninety four')
    text = text.replace('9', 'nine')
    text = text.replace('500', 'five hundred')
    text = text.replace('.', '')
    text = re.sub(r'\(.*\)', '', text)
    text = re.sub(r'[\w ]+: ', ' ', text)
    text = re.sub(r' +', ' ', text)
    if text[0] == ' ':
        text = text[1:]
    text = re.sub(r'\[.*\] *', ' ', text)
    text = text.strip()
    if text == '':
        text = null
    return text


subtitle_lookup = pd.read_table("../data/knnw_en_sub_edit.csv", sep = ";", header=0)
length = len(subtitle_lookup)
for i in range(length):
    subtitle_item = subtitle_lookup.iloc[i, 3]
    subtitle_lookup.iloc[i, 3] = '"' + remove_chars(subtitle_item) + '"'
newpath = "../data/processed"    
subtitle_lookup.to_csv(newpath + ".csv", index=False, sep=";", quoting=csv.QUOTE_NONE)
subtitle_lookup.to_csv(newpath + "_comma.csv", index=False, quoting=csv.QUOTE_NONE)
   