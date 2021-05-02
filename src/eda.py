from setup import *
from datasets import *


dataset = KnnwAudioDataset(knnw_audio_path
                        , knnw_subtitle_path
                        , KNNW_TOTAL_FRAMES
                        , KNNW_TOTAL_DURATION
                        , spec_aug=cfg['spec_augment']
                        , freq=cfg['freq_mask']
                        , time=cfg['time_mask']
                        )

split_idx = int(cfg['train_test_split']*len(dataset))
train_dataset = Subset(dataset, [idx for idx in range(split_idx)])
val_dataset = Subset(dataset, [idx for idx in range(split_idx, len(dataset))])





train_loader = DataLoader(dataset=train_dataset
                        , batch_size=8
                        , num_workers=cfg['num_workers']
                        , pin_memory=cfg['pin_memory']
                        , collate_fn=pad_collate_fn)
val_loader = DataLoader(dataset=val_dataset
                        , batch_size=cfg['val_batch_size']
                        , num_workers=cfg['val_workers']
                        , pin_memory=cfg['pin_memory']
                        , collate_fn=pad_collate_fn)


print(len(train_dataset))
print(len(val_dataset))



# for inputs, targets, input_lengths, target_lengths in train_loader:
#     assert(targets.size(0) == len(input_lengths))
#     assert(len(input_lengths) == len(target_lengths))
#     inputs = inputs.to(device, non_blocking=True)
#     targets = targets.to(device, non_blocking=True)
#     batch_size = inputs.size(0)
#     max_seq_len = inputs.size(1)
#     max_target_len = targets.size(1)
#     assert(max_seq_len == max(input_lengths))
#     assert(max_target_len == max(target_lengths))





# for inputs, targets, input_lengths, target_lengths in val_loader:
#     assert(targets.size(0) == len(input_lengths))
#     assert(len(input_lengths) == len(target_lengths))
#     inputs = inputs.to(device, non_blocking=True)
#     targets = targets.to(device, non_blocking=True)
#     input_lengths, sorted_idxs = input_lengths.sort(dim=-1)
#     inputs = inputs[sorted_idxs]
#     targets = targets[sorted_idxs]
#     target_lengths = target_lengths[sorted_idxs]
#     batch_size = inputs.size(0)
#     max_seq_len = inputs.size(1)
#     max_target_len = targets.size(1)
#     assert(max_seq_len == max(input_lengths))
#     assert(max_target_len == max(target_lengths))
