from setup import *



def plot_attention(epoch_attentions, epoch):
    attn_plot_size = 16
    attention = np.array(epoch_attentions[-1])
    attention = attention[:attn_plot_size, :attn_plot_size]
    plt.clf()
    sns_plot = sns.heatmap(attention, cmap="GnBu")
    plt.title('Hidden State Activation vs. Decoder Time Step', fontsize=12)
    plt.xlabel('Decoder Time Step', fontsize=12)
    plt.ylabel('Hidden Activation', fontsize=12)
    curr_plot_dir = f"{plot_path}/{start_time}-{tune.get_trial_name()}/"
    os.makedirs(curr_plot_dir, exist_ok=True)
    plt.savefig(f"{curr_plot_dir}/attn-epoch{epoch}-.png")
    plt.savefig(f"{tune.get_trial_dir()}/attn-epoch{epoch}-.png")

