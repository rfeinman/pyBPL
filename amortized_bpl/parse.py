import sys
sys.path.insert(1, '/Users/tuananhle/Documents/research/projects/amortized-bpl')

import data
import model
import pyprob
import matplotlib.pyplot as plt


def main():
    bpl = model.BPL()
    bpl.load_inference_network('save/bpl_inference_network')
    omniglot_test_dataset = data.OmniglotDataset(data.test_img_dir, data.test_motor_dir)
    test_image = omniglot_test_dataset[0][0]

    posterior = bpl.posterior_results(
        num_traces=10,
        inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK,
        observe={'image': test_image}
    )
    char_type_mode, char_token_mode = posterior.mode
    reconstructed_image = bpl.model.sample_image(char_token_mode)
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=100)
    axs[0].imshow(1 - test_image, cmap='gray')
    axs[0].set_title('test image')
    axs[1].imshow(1 - reconstructed_image, cmap='gray')
    axs[1].set_title('reconstructed test image')
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.1)
    filename = 'plots/parse.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()