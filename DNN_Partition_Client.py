import thriftpy2 as thriftpy
import numpy as np
from thriftpy2.rpc import make_client
from Branchy_Alexnet_Infer import infer
from utils import load_regression_data
from config import *
from Optimize import Optimize
import time
from typing import Dict
from FI_loader import get_loaders, SewageDataset
from torch.utils import data

BATCH_SIZE = 1
NUM_WORKER = 2
FILENAME = 'intermediate.npy'


def client_start():
    partition_thrift = thriftpy.load('partition.thrift', module_name='partition_thrift')
    return make_client(partition_thrift.Partition, '202.199.117.66', 6000, timeout=100000)


def file_info(filename: str):
    with open(filename, 'rb') as file:
        file_content = file.read()
    return {filename: file_content}


if __name__ == '__main__':
    # get time threshold

    # threshold = float(input('Please input latency threshold: '))
    threshold = 1.0
    # get test data

    # client init
    client = client_start()

    # info = file_info(FILENAME)
    # print('Predict answer is: ' + client.partition(info, 3, 3))
    test_dataset = SewageDataset(TEST_DATASET, mode="test")
    test_loader = data.DataLoader(test_dataset,
                                  2,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=NUM_WORKER)
    images, labels = next(iter(test_loader))

    # time start include optimize and translate server regression data
    start = time.time()
    server_regression_data = client.load_server_regression_result()
    client_regression_data = load_regression_data(REGRESSION_RESULT_DIR)

    # get partition point and exit point
    ep, pp = Optimize(threshold, server_regression_data, client_regression_data)
    print('Branch is %d, and partition point is %d' %(ep, pp))

    # infer left part
    out = infer(CLIENT, ep, pp, images)

    print('Left part of model inference complete.')

    # save intermediate for RPC process
    intermediate = out.detach().numpy()
    np.save('intermediate.npy', intermediate)

    info = file_info(FILENAME)
    print('Predict answer is: ' + client.partition(info, ep, pp))
    print('True label is: %d' %labels[0])

    end = time.time()
    print('Total time: %f' %(end-start))
