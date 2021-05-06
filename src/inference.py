from setup import *
from seq2seq import Seq2Seq
from preprocess import *
from datasets import *
from train import *

# Load Config
inf_cfg_path = "./inference_config.yaml"
with open(inf_cfg_path) as file:
    inf_cfg = yaml.load(file, Loader=Loader)

model1_ckpt_path = os.path.expanduser(inf_cfg['knnw_ckpt_path'])
input_dim = 129
model1 = Seq2Seq(input_dim=input_dim
                , vocab_size=len(LETTER_LIST)
                , encoder_hidden_dim=inf_cfg['enc_h'][0]
                , decoder_hidden_dim=inf_cfg['dec_h'][0]
                , embed_dim=inf_cfg['embed_dim'][0]
                , key_value_size=inf_cfg['attn_dim'][0]
                , enc_dropout=inf_cfg['enc_dropout'][0]
                , dec_dropout=inf_cfg['dec_dropout'][0]
                , encoder_arch=inf_cfg['encoder_arch'][0]
                , use_multihead=inf_cfg['use_multihead'][0]
                , nheads=inf_cfg['nheads'][0]
                )


model1.load_state_dict(torch.load(model1_ckpt_path))
device = torch.device(cfg['device'])
model1 = model1.to(device)
model1.eval()
torch.set_grad_enabled(False)

val_dataset = KnnwAudioDataset(knnw_audio_path
                        , knnw_subtitle_path
                        , KNNW_TOTAL_FRAMES
                        , KNNW_TOTAL_DURATION
                        , spec_aug=False
                        , freq=0
                        , time=0
                        )
split_idx = int(cfg['train_test_split']*len(val_dataset))
idxs = np.arange(len(val_dataset))
np.random.shuffle(idxs)
train_idxs, val_idxs = idxs[:split_idx], idxs[split_idx:]
val_dataset = Subset(val_dataset, val_idxs)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=False, collate_fn=collate)




outfile = open("../final_data/inference_results.csv", "w+")
eval_lev_dist = inference(model1, val_loader, criterion=nn.CrossEntropyLoss(), epoch=0, device=device, outfile=outfile)
print(f"Evaluation Levenshtein Distance: {eval_lev_dist}")
outfile.close()