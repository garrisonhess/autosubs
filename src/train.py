from setup import *
from preprocess import *
from our_datasets import *
from seq2seq import Seq2Seq
from decoder import Decoder
from encoder import Encoder
from layers import *
from plots import plot_attention
from ctcdecode import CTCBeamDecoder

def beam_to_string(path_tokens, letter_list, seq_len):
    return ''.join([letter_list[x] for x in path_tokens[:seq_len]])



def train_model(config, **kwargs):


    if cfg['use_wav2vec']:
        input_dim = 512
        train_dataset = Wav2vecProcessed(vec2wav_npy, knnw_subtitle_processed_path)
        val_dataset = Wav2vecProcessed(vec2wav_npy, knnw_subtitle_processed_path)
    else:
        input_dim = 129
        train_dataset =  KnnwAudioDataset(knnw_audio_path
                                , knnw_subtitle_path
                                , KNNW_TOTAL_FRAMES
                                , KNNW_TOTAL_DURATION
                                , spec_aug=cfg['spec_augment']
                                , freq=cfg['freq_mask']
                                , time=cfg['time_mask']
                                )

        val_dataset = KnnwAudioDataset(knnw_audio_path
                                , knnw_subtitle_path
                                , KNNW_TOTAL_FRAMES
                                , KNNW_TOTAL_DURATION
                                , spec_aug=False
                                , freq=0
                                , time=0
                                )
    


    split_idx = int(cfg['train_test_split']*len(train_dataset))
    idxs = np.arange(len(train_dataset))
    
    if cfg['random_sampling']:
        np.random.shuffle(idxs)
    
    train_idxs, val_idxs = idxs[:split_idx], idxs[split_idx:]
    train_dataset = Subset(train_dataset, train_idxs)
    val_dataset = Subset(val_dataset, val_idxs)

    if cfg['DEBUG']:
        train_dataset = val_dataset

    train_loader = DataLoader(dataset=train_dataset
                            , batch_size=config['batch_size']
                            , num_workers=cfg['num_workers']
                            , pin_memory=cfg['pin_memory']
                            , shuffle=cfg['train_shuffle']
                            , collate_fn=pad_collate_fn)
    val_loader = DataLoader(dataset=val_dataset
                            , batch_size=cfg['val_batch_size']
                            , num_workers=cfg['val_workers']
                            , pin_memory=cfg['pin_memory']
                            , shuffle=False
                            , collate_fn=pad_collate_fn)

    model = Seq2Seq(input_dim=input_dim
                    , vocab_size=len(LETTER_LIST)
                    , encoder_hidden_dim=config['enc_h']
                    , decoder_hidden_dim=config['dec_h']
                    , embed_dim=config['embed_dim']
                    , key_value_size=config['attn_dim']
                    , enc_dropout=config['enc_dropout']
                    , dec_dropout=config['dec_dropout']
                    , encoder_arch=config['encoder_arch']
                    , use_multihead=config['use_multihead']
                    , nheads=config['nheads']
                    )
    print(model)

    # handle transfer learning

    if cfg['transfer_knnw']:
        print("Transfer Learning from KNNW to KNNW")
        if cfg['use_wav2vec']:
            # when transferring from knnw dim 129 to wav2vec, ignore first layer
            checkpoint = torch.load(knnw_ckpt_path)
            transfer_keys = []
            for key in checkpoint.keys():
                if not key.startswith("encoder.encoder.0"):
                    transfer_keys.append(key)
            
            transfer_state = dict()
            for key in transfer_keys:
                transfer_state[key] = checkpoint[key]
            
            model.load_state_dict(transfer_state, strict=False)
        else:
            knnw_ckpt = torch.load(knnw_ckpt_path)
            model.load_state_dict(knnw_ckpt, strict=True)
    elif cfg['transfer_wsj']:
        if cfg['transfer_decoder']:
            print("Starting with decoder pretrained weights")
            checkpoint = torch.load(decoder_ckpt_path)
            decoder_keys = []
            for key in checkpoint.keys():
                if key.startswith('decoder'):
                    decoder_keys.append(key)
            decoder_state = dict()
            for key in decoder_keys:
                decoder_state[key] = checkpoint[key]
            model.load_state_dict(decoder_state, strict=False)
        elif cfg['transfer_full']:
            print("Transfer Learning from WSJ to KNNW")
            checkpoint = torch.load(wsj_ckpt_path)
            transfer_keys = []
            for key in checkpoint.keys():
                if not key.startswith("encoder.encoder.0"):
                    transfer_keys.append(key)
            
            transfer_state = dict()
            for key in transfer_keys:
                transfer_state[key] = checkpoint[key]
            
            model.load_state_dict(transfer_state, strict=False)
        
        # freeze proper subnetworks for transfer learning specification
        if cfg['freeze_decoder']:
            for i, dec_child in enumerate(model.decoder.children()):
                print(f"Freezing: {i, dec_child}")
                dec_child.requires_grad = False
        if cfg['freeze_encoder']:
            for i, enc_child in enumerate(model.encoder.children()):
                if i == 0:
                    for layer_idx, lstm_layer in enumerate(enc_child.children()):
                        if layer_idx > 0:
                            print(f"Freezing: {layer_idx, lstm_layer}")
                            lstm_layer.requires_grad = False
                else:
                    print(f"Freezing: {i, enc_child}")
                    enc_child.requires_grad = False

    params = model.parameters()
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    model = model.to(device=device)
    lr_scheduler = StepLR(optimizer=optimizer, step_size=config['lr_step'], gamma=config['gamma'])

    decoder = CTCBeamDecoder(
        LETTER_LIST,
        model_path=kenlm_path,
        alpha=0.1,
        beta=0.1,
        cutoff_top_n=5,
        cutoff_prob=1.0,
        beam_width=10,
        num_processes=4,
        blank_id=letter2index["<EOS>"],
        log_probs_input=True
    )

    # Training Variables
    best_lev_dist = 1000000000
    last_ckpt_time = 0.0
    best_model = copy.deepcopy(model.state_dict())
    teacher_forcing = cfg['tf_init']
    mode = 'train'

    for epoch in range(1, cfg['epochs'] + 1):
        model, train_loss, teacher_forcing = train(model
                                                    , mode
                                                    , config
                                                    , train_loader
                                                    , optimizer
                                                    , criterion
                                                    , lr_scheduler
                                                    , epoch
                                                    , device
                                                    , cfg['update_freq']
                                                    , cfg['amp']
                                                    , cfg['DEBUG']
                                                    , teacher_forcing
                                                    , scaler=None)
        
        eval_gr_dist, eval_beam_dist = eval(model, val_loader, criterion, epoch, device, decoder)

        # Update Ray Tune
        tune.report(train_loss=train_loss, eval_gr_dist=eval_gr_dist, eval_beam_dist=eval_beam_dist)

        if min(eval_beam_dist, eval_gr_dist) <= best_lev_dist:
            best_lev_dist = min(eval_gr_dist, eval_beam_dist)
            best_model = copy.deepcopy(model.state_dict())
            curr_time = time.strftime("%Y-%m-%d-%H-%M%S")
            model_type = 'knnw'

            # only checkpoint after N minutes
            N = 10
            if not cfg['DEBUG'] and time.time() > last_ckpt_time + 60*N:
                torch.save(model.state_dict(),
                           f"{checkpoints_path}/{model_type}-{curr_time}-epoch{epoch}-dist{int(best_lev_dist)}-{tune.get_trial_name()}.pth")


def train(model, mode, config, train_loader, optimizer, criterion, lr_scheduler, epoch, device, update_freq, amp, debug, teacher_forcing, scaler):

    # training mode
    model.train()
    torch.set_grad_enabled(True)

    new_pct = 0
    training_loss = 0.
    batch_index = 0
    last_pct = 0
    num_train_batches = (len(train_loader.dataset)/train_loader.batch_size) + 1
    running_loss = 0.
    running_corrects = 0.
    start_time = time.time()
    epoch_attentions = []

    if epoch % cfg['tf_drop_every'] == 0:
        teacher_forcing = max(teacher_forcing - cfg['tf_drop'], cfg['min_tf'])
    print(f"mode: {mode}, epoch: {epoch}, teacher forcing: {teacher_forcing}")

    for inputs, targets, input_lengths, target_lengths in train_loader:
        assert(targets.size(0) == len(input_lengths))
        assert(len(input_lengths) == len(target_lengths))
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = inputs.size(0)
        max_seq_len = inputs.size(1)
        max_target_len = targets.size(1)
        assert(max_seq_len == max(input_lengths))
        assert(max_target_len == max(target_lengths))

        optimizer.zero_grad()

        outputs, batch_attentions, _ = model(inputs=inputs
                                        , input_lengths=input_lengths
                                        , teacher_forcing=teacher_forcing
                                        , device=device
                                        , targets=targets
                                        , mode='train')
        
        epoch_attentions.append(batch_attentions)
        outputs = pack_padded_sequence(outputs, target_lengths, batch_first=True, enforce_sorted=False)
        targets = pack_padded_sequence(targets, target_lengths, batch_first=True, enforce_sorted=False)

        # Calculate the loss and mask it to remove the padding part
        loss = criterion(outputs.data, targets.data).sum()
        loss /= batch_size
        loss.backward()

        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        if math.isnan(grad):
            print(f"exploding gradient: {grad}")
        else:
            optimizer.step()


        # log progress
        batch_index += 1
        new_pct = (int)((batch_index/num_train_batches)*100)

        if new_pct > last_pct + update_freq:
            time_remaining = (100. - new_pct) * \
                ((time.time() - start_time)/float(new_pct))
            min_remaining = int(time_remaining // 60)
            sec_remaining = int(time_remaining % 60)
            print(f"batch index: {batch_index}, percent complete: {new_pct}, approx {min_remaining} mins and {sec_remaining} secs remaining")
            last_pct = new_pct

        # update statistics
        running_loss += float(loss.item())
    
    plot_attention(epoch_attentions, epoch)
    train_loss = running_loss / len(train_loader.dataset)
    lr_scheduler.step()

    return model, train_loss, teacher_forcing


def eval(model, val_loader, criterion, epoch, device, decoder):

    model.eval()
    torch.set_grad_enabled(False)
    running_loss = 0.
    running_lev_dist = 0.
    running_beam_dist = 0.
    ctr = 0
    mode = 'val'

    for inputs, targets, input_lengths, target_lengths in val_loader:
        assert(targets.size(0) == len(input_lengths))
        assert(len(input_lengths) == len(target_lengths))
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = inputs.size(0)
        max_seq_len = inputs.size(1)
        max_target_len = targets.size(1)
        assert(max_seq_len == max(input_lengths))
        assert(max_target_len == max(target_lengths))

        # outputs come out as (batch_size, max_target_length, classes)
        outputs, _, encoded_seq_lens = model(inputs=inputs, input_lengths=input_lengths, teacher_forcing=0.0, device=device, targets=targets, mode=mode)

        # beam search
        beam_results, beam_scores, timesteps, out_seq_len = decoder.decode(outputs, seq_lens=encoded_seq_lens)
        beam_output_paths = []

        for i, _ in enumerate(beam_results):
            result = beam_to_string(beam_results[i][0], LETTER_LIST, out_seq_len[i][0])
            beam_output_paths.append(result)
        

        #  greedy search
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
        

        # build target string
        target_paths = []
        for batch_idx, target in enumerate(targets):
            target_path = ""
            curr_target_len = int(target_lengths[batch_idx])
            curr_target = target[:curr_target_len]

            for target_char_idx in curr_target:
                next_letter = index2letter[int(target_char_idx)]
                if next_letter == '<EOS>' or next_letter == '<PAD>':
                    break
                target_path += next_letter
            
            target_paths.append(target_path)
        
        # accumulate levenshtein distance between each output and target
        dist = 0.0
        for out_path, targ_path in zip(output_paths, target_paths):
            dist += Levenshtein.distance(out_path, targ_path)

        # accumulate levenshtein distance between each output and target
        beam_dist = 0.0
        for beam_path, targ_path in zip(beam_output_paths, target_paths):
            beam_dist += Levenshtein.distance(beam_path, targ_path)

        assert(len(output_paths) == len(target_paths))
        if ctr < 3:
            for i in range(min(10, len(output_paths))):
                print(f"TARGET {i}: {target_paths[i]}")
                print(f"GREEDY {i}: {output_paths[i]}")
                # print(f"BEAM {i}: {beam_output_paths[i]}")
        
        ctr += 1

        # update statistics
        running_lev_dist += dist
        running_beam_dist += beam_dist

    # report statistics
    lev_dist = running_lev_dist / len(val_loader.dataset)
    beam_lev_dist = running_beam_dist / len(val_loader.dataset)

    return lev_dist, beam_lev_dist

