import logging
import subprocess as sp
from itertools import product

import numpy as np

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    param_grid = {
        'version': [0, 1, 3],
        'lparam': [2, 3, 4, 5, 6, 7],
        'rstate': [20],
        'refit': [1],
        'occupancy': [2],
        # 'cpus': [4, 8, 16],
    }

    nparams = np.product([len(v) for v in param_grid.values()])

    def my_product(dicts):
        return (dict(zip(dicts, x)) for x in product(*dicts.values()))

    for i, params in enumerate(my_product(param_grid)):
        cpus = 2
        ntrials = 4

        args = ' '.join(['--{}={}'.format(k, v) for k, v in params.items()])
        cmd = ['make', 'bench', "ARGS={}".format(args),
               "CPU_COUNT={}".format(cpus)]
        logger.info('running {} of {}: {}'.format(i+1, nparams, ' '.join(cmd)))

        # todo: find a way to open multiple subprocesses / farm out tasks
        for _ in range(ntrials):
            sp.Popen(['make',
                      'bench',
                      "ARGS={}".format(args),
                      "CPU_COUNT={}".format(cpus)]).wait()
