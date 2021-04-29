from setup import *
from train import *




search_space = {
    "num_epochs": cfg['epochs'],
    "lr": tune.grid_search(cfg['lr']),
    "weight_decay": tune.grid_search(cfg['weight_decay']),
    "gamma": tune.grid_search(cfg['gamma']),
    "batch_size": tune.grid_search(cfg['batch_size']),
    "lr_step": tune.grid_search(cfg['lr_step']),
    "conv_channels": tune.grid_search(cfg["conv_channels"]),
    "enc_dropout": tune.grid_search(cfg["enc_dropout"]),
    "enc_h": tune.grid_search(cfg["enc_h"]),
    "dec_h": tune.grid_search(cfg["dec_h"]),
    "embed_dim": tune.grid_search(cfg["embed_dim"]),
    "attn_dim": tune.grid_search(cfg["attn_dim"]),
    "dec_dropout": tune.grid_search(cfg["dec_dropout"]),
    "lock_drop": tune.grid_search(cfg["lock_drop"]),
    "use_multihead": tune.grid_search(cfg["use_multihead"]),
    "nheads": tune.grid_search(cfg["nheads"]),
    "encoder_arch": tune.grid_search(cfg["encoder_arch"]),
}




scheduler = tune.schedulers.ASHAScheduler(
    max_t=cfg['epochs'],
    grace_period=cfg['grace_period'],
    reduction_factor=cfg['reduction_factor'],
    brackets=cfg['brackets'])





result = tune.run(run_or_experiment=tune.with_parameters(train_model, **cfg), name=cfg['experiment_name'], num_samples=cfg['num_samples'], resources_per_trial={
                  'gpu': 1}, config=search_space, metric="eval_lev_dist", mode="min", scheduler=scheduler, checkpoint_at_end=False, search_alg=None, verbose=cfg['verbosity'], local_dir=cfg['ray_results_dir'], log_to_file=True)
