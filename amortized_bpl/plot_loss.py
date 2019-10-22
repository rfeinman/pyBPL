import matplotlib.pyplot as plt
import model


if __name__ == '__main__':
    bpl = model.BPL()
    bpl.load_inference_network('save/bpl_inference_network')

    trace_per_second = bpl._inference_network._total_train_traces / \
        bpl._inference_network._total_train_seconds
    print('Average traces/second: {:.2f}'.format(trace_per_second))
    fig, ax = plt.subplots(1, 1, dpi=200)
    ax.plot(bpl._inference_network._history_train_loss_trace,
            bpl._inference_network._history_train_loss)
    ax.set_xlabel('traces')
    ax.set_ylabel('loss')

    fig.tight_layout(pad=0)

    filename = 'plots/loss.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))
