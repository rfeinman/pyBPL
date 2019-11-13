import model


def main(args):
    print(args)

    if args.large_lib:
        lib_dir = '../lib_data'
        dataset_dir = '{}'.format(args.dataset_dir_base)
    else:
        lib_dir = '../lib_data250'
        dataset_dir = '{}_250'.format(args.dataset_dir_base)
    bpl = model.BPL(lib_dir=lib_dir)
    bpl.save_dataset(dataset_dir, args.num_traces, args.num_traces_per_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-dir-base', default='./offline_data/data')
    parser.add_argument('--num-traces', type=int, default=10)
    parser.add_argument('--num-traces-per-file', type=int, default=2)
    parser.add_argument('--large-lib', action='store_true',
                        help='use 1250 primitives')
    args = parser.parse_args()
    main(args)
