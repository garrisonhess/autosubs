from setup import *
import attention
from attention import *

# Load the training, validation and testing data
train_data = np.load(train_path, allow_pickle=True, encoding='bytes')
val_data = np.load(val_path, allow_pickle=True, encoding='bytes')
test_data = np.load(test_path, allow_pickle=True, encoding='bytes')
debug_train_data = np.load(debug_train_path, allow_pickle=True, encoding='bytes')
debug_val_data = np.load(debug_val_path, allow_pickle=True, encoding='bytes')


# Load the training, validation raw text transcripts
raw_train_transcript = np.load(train_transcripts_path, allow_pickle=True, encoding='bytes')
raw_val_transcript = np.load(val_transcripts_path, allow_pickle=True, encoding='bytes')
raw_debug_train_transcript = np.load(debug_train_transcripts_path, allow_pickle=True, encoding='bytes')
raw_debug_val_transcript = np.load(debug_val_transcripts_path, allow_pickle=True, encoding='bytes')





#########################################################  DEBUG DATA
# print(index2letter.items())
# print("=================================")
# print(letter2index.items())


print(f"debug_train_data utterances: {debug_train_data.shape}")
print(f"debug_train_data[0].shape {debug_train_data[0].shape}")
print(f"raw_debug_train_transcript: {raw_debug_train_transcript.shape}")
print(f"raw_debug_train_transcript[0]: {raw_debug_train_transcript[0].shape}")
print(f"raw_debug_train_transcript[0]: {raw_debug_train_transcript[0]}")


debug_train_transcript = newtransform(raw_debug_train_transcript, debug=True)
debug_val_transcript = newtransform(raw_debug_val_transcript, debug=True)

print(f"debug_train_transcript[0] {debug_train_transcript[0]}")

debug_train_sentence = ""
for char_val in debug_train_transcript[0]:
    debug_train_sentence += index2letter[char_val]

debug_val_sentence = ""
for char_val in debug_val_transcript[0]:
    debug_val_sentence += index2letter[char_val]

print(f"debug_train_sentence: {debug_train_sentence}")
print(f"debug_val_sentence: {debug_val_sentence}")





##################### ACTUAL DATA

# print(f"training utterances: {train_data.shape}")
# print(f"train_data[0].shape {train_data[0].shape}")
# print(f"train_data[1].shape {train_data[1].shape}")
# print(f"raw_train_transcript: {raw_train_transcript.shape}")
# print(f"raw_train_transcript[0]: {raw_train_transcript[0].shape}")
# print(f"raw_train_transcript[1]: {raw_train_transcript[1].shape}")
# print(f"raw_train_transcript[0]: {raw_train_transcript[0]}")
# print(f"raw_val_transcript[0]: {raw_val_transcript[0]}")

# train_transcript = transform_letter_to_index(raw_train_transcript)
# val_transcript = transform_letter_to_index(raw_val_transcript)

# print(f"train_transcript[0] {train_transcript[0]}")
# print(f"val_transcript[0] {val_transcript[0]}")

# train_sentence = ""
# for char_val in train_transcript[0]:
#     train_sentence += index2letter[char_val]

# val_sentence = ""
# for char_val in val_transcript[0]:
#     val_sentence += index2letter[char_val]

# print(f"train_sentence: {train_sentence}")
# print(f"val_sentence: {val_sentence}")