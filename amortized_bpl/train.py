import model
import pyprob


def main():
    num_traces = 10
    batch_size = 1

    bpl = model.BPL()
    bpl.learn_inference_network(
        num_traces=num_traces,
        batch_size=batch_size,
        observe_embeddings={'image': {'dim': 32,
                            'embedding': pyprob.ObserveEmbedding.CNN2D5C,
                            'reshape': (1, 105, 105)}},
        inference_network=pyprob.InferenceNetwork.LSTM)

    bpl.save_inference_network('save/bpl_inference_network')


if __name__ == '__main__':
    main()