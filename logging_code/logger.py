
import logging
import copy
import os
import logging, logging.config
from pathlib import Path

logger = None
def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    #args_copy.iters = 1
    # args_copy.verbose = False
    # args_copy.log_interval = 1
    # args_copy.seed = 0

    log_path = args.result_dir + '/{0}.log'.format(args.seed)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg) 

# def init():
#     global log
#     SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', 4711)
#     JOBTMPDIR = (Path("/tmp") / f"job-{SLURM_JOB_ID}")
#     logdir = JOBTMPDIR / "logs"
#     for d in [JOBTMPDIR, logdir]:
#         d.mkdir(exist_ok=True, parents=True)

#     lname = "CIFAR10"

#     logging.config.dictConfig({
#         'version': 1,
#         'formatters': {
#             'short': {
#                 'class': 'logging.Formatter',
#                 'format': '%(asctime)s %(name)-9s %(levelname)-4.4s: %(message)s'
#             },
#             'detail': {
#                 'class': 'logging.Formatter',
#                 'format': '%(asctime)s %(name)-9s %(levelname)-4.4s %(filename)s:%(lineno)d %(funcName)s: %(message)s'
#             }
#         },
#         'handlers': {
#             'console': {
#                 'class': 'logging.StreamHandler',
#                 'level': 'INFO',
#                 'formatter': 'short',
#             },
#             'file': {
#                 'class': 'logging.FileHandler',
#                 'filename': f'{logdir / "debug.log"}',
#                 'level': 'DEBUG',
#                 'formatter': 'detail',
#             }
#         },
#         'loggers' : {
#             '': {
#                 'level': 'INFO',
#                 'handlers': ['console', 'file'],
#             },
#             lname : {
#                 'level': 'DEBUG',
#                 'handlers': ['console', 'file'],
#                 'propagate': False,
#             }
#         }
#     })
#     logging.captureWarnings(True)
#     log = logging.getLogger(lname)
#     log.setLevel(logging.DEBUG)
#     return log


