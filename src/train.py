from setup import *
from preprocess import *
from datasets import *
from attention import Attention
from seq2seq import Seq2Seq
from decoder import Decoder
from encoder import Encoder
from layers import *
from plots import plot_attention

def train_model(config, **kwargs):

    transforms = None
    if cfg['spec_augment']:
        transforms = spec_augment
    

    #####  SET THE KNNW VARIABLES HERE
    #####  CREATE NEW CONFIG FIELDS FOR PATHS
    if cfg['dataset'] == "KNNW":
        train_dataset = KnnwAudioDataset(audio_path=None
                                        , subtitle_lookup_path=None
                                        , total_frames=None
                                        , total_duration=None)
        val_dataset = KnnwAudioDataset(audio_path=None
                                    , subtitle_lookup_path=None
                                    , total_frames=None
                                    , total_duration=None)
    else:
        train_dataset = ASRDataset(train_path, train_transcripts_path, transforms=None)
        val_dataset = ASRDataset(val_path, val_transcripts_path)



    if cfg['minival']:
        val_dataset = torch.utils.data.Subset(val_dataset, [x for x in range(cfg['threshold'])])

    if cfg['DEBUG']:
        print(f"debug mode, using subsets of size {cfg['threshold']}")
        train_dataset = torch.utils.data.Subset(train_dataset, [x for x in range(cfg['threshold'])])
        val_dataset = torch.utils.data.Subset(train_dataset, [x for x in range(cfg['threshold'])])
    
    train_loader = DataLoader(train_dataset
                            , batch_size=config['batch_size']
                            , shuffle=cfg['train_shuffle']
                            , num_workers=cfg['num_workers']
                            , pin_memory=cfg['pin_memory']
                            , collate_fn=collate)
    val_loader = DataLoader(val_dataset
                        , batch_size=cfg['val_batch_size']
                        , shuffle=False
                        , num_workers=cfg['val_workers']
                        , pin_memory=False
                        , collate_fn=collate)

    model = Seq2Seq(input_dim=cfg['input_dim']
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
    if cfg['pretrained_decoder'] and cfg['pretrained_full']:
        print("Both pretrained_decoder and pretrained_full are set.")
        exit()
    elif cfg['pretrained_decoder']:
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
    elif cfg['pretrained_full']:
        print("Starting with full pretrained weights")
        checkpoint = torch.load(full_ckpt_path)
        model.load_state_dict(checkpoint)
    
    params = model.parameters()
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    model = model.to(device=device)
    lr_scheduler = StepLR(optimizer=optimizer, step_size=config['lr_step'], gamma=config['gamma'])

    # Training Variables
    best_lev_dist = 1000000000
    last_ckpt_time = 0.0
    best_model = copy.deepcopy(model.state_dict())
    eval_lev_dist = 1000.
    teacher_forcing = cfg['tf_init']
    mode = 'warmup'
    warmup_epochs = cfg['warmup_epochs']
    # disable encoder training during warmup
    model.encoder.requires_grad_(requires_grad=False)

    for epoch in range(1, cfg['epochs'] + 1):
        if epoch > warmup_epochs and mode == 'warmup':
            mode = 'train'
            # enable encoder training after warmup
            print(f"enabling encoder gradient at epoch: {epoch}")
            model.encoder.requires_grad_(requires_grad=True)
            best_lev_dist = 10000
            eval_lev_dist = 10000
            last_ckpt_time = 0.
                
        model, train_loss, teacher_forcing = train(model
                                                    , mode
                                                    , warmup_epochs
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
        
        eval_lev_dist = eval(model, val_loader, criterion, epoch, device, warmup=(mode == 'warmup'))

        # Update Ray Tune
        tune.report(train_loss=train_loss, eval_lev_dist=eval_lev_dist)

        if mode == 'warmup' or eval_lev_dist <= best_lev_dist:
            best_lev_dist = eval_lev_dist
            best_model = copy.deepcopy(model.state_dict())
            curr_time = time.strftime("%Y-%m-%d-%H-%M%S")
            model_type = 'decoder' if mode == 'warmup' else 'full'

            # only checkpoint after N minutes
            N = 1
            if not cfg['DEBUG'] and time.time() > last_ckpt_time + 60*N:
                torch.save(model.state_dict(),
                           f"{checkpoints_path}/{model_type}-{curr_time}-epoch{epoch}-dist{int(eval_lev_dist)}-{tune.get_trial_name()}.pth")


def train(model, mode, warmup_epochs, config, train_loader, optimizer, criterion, lr_scheduler, epoch, device, update_freq, amp, debug, teacher_forcing, scaler):

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

    if cfg['DEBUG'] and cfg['TOY_DATA']:
        teacher_forcing = 0.0
    else:
        if epoch % cfg['tf_drop_every'] == 0:
            teacher_forcing = max(teacher_forcing - cfg['tf_drop'], cfg['min_tf'])


    print(f"mode: {mode}, epoch: {epoch}, teacher forcing: {teacher_forcing}")

    for inputs, targets, input_lengths, target_lengths in train_loader:
        assert(targets.size(0) == len(input_lengths))
        assert(len(input_lengths) == len(target_lengths))
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        input_lengths, sorted_idxs = input_lengths.sort(dim=-1)
        inputs = inputs[sorted_idxs]
        targets = targets[sorted_idxs]
        target_lengths = target_lengths[sorted_idxs]
        batch_size = inputs.size(0)
        max_seq_len = inputs.size(1)
        max_target_len = targets.size(1)
        assert(max_seq_len == max(input_lengths))
        assert(max_target_len == max(target_lengths))

        optimizer.zero_grad()


        # outputs come out as (batch_size, max_target_length, classes)
        if mode == 'warmup':
            # kv dims: (B, T, key_value_size)
            time_downsample = 8
            key = torch.zeros(size=(batch_size, max_seq_len // time_downsample, cfg['attn_dim'][0]), dtype=torch.float32, device=device)
            value = torch.zeros(size=(batch_size, max_seq_len // time_downsample, cfg['attn_dim'][0]), dtype=torch.float32, device=device)
            encoded_seq_lens = torch.tensor([z // time_downsample for z in input_lengths], dtype=torch.long)
            outputs, batch_attentions = model.decoder(key=key
                                                    , value=value
                                                    , encoded_seq_lens=encoded_seq_lens
                                                    , device=device
                                                    , teacher_forcing=teacher_forcing
                                                    , targets=targets
                                                    , mode='warmup'
                                                    )
        else:
            outputs, batch_attentions = model(inputs=inputs
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
    if mode != 'warmup':
        lr_scheduler.step()

    return model, train_loss, teacher_forcing




def eval(model, val_loader, criterion, epoch, device, peek=False, warmup=False):

    model.eval()
    torch.set_grad_enabled(False)
    running_loss = 0.
    running_lev_dist = 0.
    ctr = 0
    mode = 'val'
    if warmup:
        mode = 'warmup'

    for inputs, targets, input_lengths, target_lengths in val_loader:
        assert(targets.size(0) == len(input_lengths))
        assert(len(input_lengths) == len(target_lengths))
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        input_lengths, sorted_idxs = input_lengths.sort(dim=-1)
        inputs = inputs[sorted_idxs]
        targets = targets[sorted_idxs]
        target_lengths = target_lengths[sorted_idxs]
        batch_size = inputs.size(0)
        max_seq_len = inputs.size(1)
        max_target_len = targets.size(1)
        assert(max_seq_len == max(input_lengths))
        assert(max_target_len == max(target_lengths))

        # outputs come out as (batch_size, max_target_length, classes)
        outputs, _ = model(inputs=inputs, input_lengths=input_lengths, teacher_forcing=0.0, device=device, targets=targets, mode=mode)
        output_paths = []
        for batch_idx, output in enumerate(outputs):
            seq = ''
            for seq_idx, char_probs in enumerate(output):
                char_idx = int(torch.argmax(char_probs))
                next_letter = index2letter[char_idx]
                seq += next_letter
                if next_letter == '<EOS>' or next_letter == '<PAD>':
                    break
            output_paths.append(seq)
        
        # build target string
        target_paths = []
        for batch_idx, target in enumerate(targets):
            target_path = ""
            curr_target_len = int(target_lengths[batch_idx])
            curr_target = target[:curr_target_len]

            for target_char_idx in curr_target:
                target_path += index2letter[int(target_char_idx)]
            
            target_paths.append(target_path)
        
        # accumulate levenshtein distance between each output and target
        dist = 0.0
        for out_path, targ_path in zip(output_paths, target_paths):
            dist += Levenshtein.distance(out_path, targ_path)

        if ctr < 3:
            print(f"TARGET1: {target_paths[10]}, TARGET2: {target_paths[20]}")
            print(f"BEAM1: {output_paths[10]}, BEAM2: {output_paths[20]}")
            if peek:
                print(f"only running 1 validation batch")
        
        ctr += 1

        # update statistics
        running_lev_dist += dist

        if peek:
            return running_lev_dist / batch_size

    # report statistics
    lev_dist = running_lev_dist / len(val_loader.dataset)

    return lev_dist
