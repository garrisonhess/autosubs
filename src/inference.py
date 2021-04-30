from setup import *
from seq2seq import Seq2Seq
from preprocess import *
from datasets import *
from train import *

# Load Config
inf_cfg_path = "./inference_config.yaml"
with open(inf_cfg_path) as file:
    inf_cfg = yaml.load(file, Loader=Loader)

model1_ckpt_path = os.path.expanduser(inf_cfg['full_ckpt_path'])

model1 = Seq2Seq(input_dim=cfg['input_dim']
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


# load datasets
val_dataset = ASRDataset(val_path, val_transcripts_path)
test_dataset = ASRTestDataset(test_path)

if inf_cfg['DEBUG']:
    print(f"debug mode, using subsets of size {cfg['threshold']}")
    val_dataset = torch.utils.data.Subset(val_dataset, [x for x in range(cfg['threshold'])])
    test_dataset = torch.utils.data.Subset(test_dataset, [x for x in range(cfg['threshold'])])


val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=False, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=False, collate_fn=collate_test)
# validate model before inference
eval_lev_dist = eval(model1, val_loader, criterion=nn.CrossEntropyLoss(), epoch=0, device=device, peek=False)
print(f"Evaluation Levenshtein Distance: {eval_lev_dist}")
# eval_lev_dist, eval_beam_lev_dist = eval_beam(model1, val_loader, criterion=nn.CrossEntropyLoss(), epoch=0, device=device, peek=False)
# print(f"Evaluation Levenshtein Distance: {eval_lev_dist}, Beam Lev Distance: {eval_beam_lev_dist}")




preds = []

# perform inference
for inputs, input_lengths in test_loader:
    inputs = inputs.to(device, non_blocking=True)
    batch_size = inputs.size(0)
    max_seq_len = inputs.size(1)
    assert(max_seq_len == max(input_lengths))

    # outputs come out as (batch_size, max_target_length, classes)
    outputs, _ = model1(inputs=inputs, input_lengths=input_lengths, teacher_forcing=0.0, device=device, targets=None, mode='val')
    
    output_paths = []
    for batch_idx, output in enumerate(outputs):
        seq = ''
        for seq_idx, char_probs in enumerate(output):
            char_idx = int(torch.argmax(char_probs))
            next_letter = index2letter[char_idx]

            if next_letter == '<EOS>' or next_letter == '<PAD>':
                break

            seq += next_letter
        output_paths.append(seq)
    
    preds += output_paths


indexed_preds = []

for i, pred in enumerate(preds):
    indexed_preds.append([i, pred])

out_path = f"{results_path}predictions.csv"
predictions = pd.DataFrame(indexed_preds, columns=[
                           'id', 'label'], dtype=np.int)
predictions.to_csv(out_path, index=False)
print(f"Wrote predictions to {out_path}")
