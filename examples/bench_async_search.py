import pickle
import uuid

import click

# utilities
from examples.bench_search import bench_search


@click.command()
@click.option('--version', default=0, help='Code version, 0=sklearn, 1=dasksearchcv,'
                                           ' 2=async, 3=async-with-cacheplugin')
@click.option('--lparam', default=4, help='number of parameters used in grid')
@click.option('--rstate', default=0, help='random state for SGD classifier')
@click.option('--refit', default=1, help='do refit')
@click.option('--occupancy', default=2, help='occupancy factor')
@click.option('--outfile', default=None, help='output file name')
def main(version, lparam, rstate, refit, occupancy, outfile):
    results = bench_search(version, lparam, rstate, refit, occupancy)
    results['params'] = {
        'version': version,
        'lparam': lparam,
        'rstate': rstate,
        'refit': refit,
        'occupancy': occupancy
    }
    # save the results to pickle
    if outfile is None:
        outfile = uuid.uuid4()

    with open('/data/{}.pkl'.format(outfile), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()

