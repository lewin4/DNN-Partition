import thriftpy2 as thriftpy
import numpy as np
from thriftpy2.rpc import make_client
from Branchy_Alexnet_Infer import infer
from utils import load_regression_data
from config import *

import time
from typing import Dict
from FI_loader import get_loaders, SewageDataset
from torch.utils import data

BATCH_SIZE = 1
NUM_WORKER = 2
FILENAME = 'intermediate.npy'


def client_start():
    partition_thrift = thriftpy.load('partition.thrift', module_name='partition_thrift')
    return make_client(partition_thrift.Partition, '127.0.0.1', 6000, timeout=3000)


def file_info(filename: str):
    with open(filename, 'rb') as file:
        file_content = file.read()
    return {filename: file_content}


if __name__ == '__main__':
    from Optimize import Optimize
    # get time threshold

    # threshold = float(input('Please input latency threshold: '))

    # get test data

    # info = file_info(FILENAME)
    # print('Predict answer is: ' + client.partition(info, 3, 3))
    test_dataset = SewageDataset(TEST_DATASET, mode="test")
    test_loader = data.DataLoader(test_dataset,
                                  1,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=NUM_WORKER)

#    for images, labels in test_loader:
#        break

    images, labels = test_dataset[5]
    if len(images.size()) == 3:
        images = images.unsqueeze(0)

    # client init
    client = client_start()

    # time start include optimize and translate server regression data
    start = time.time()
    # print("Try to request server device data.")
    # server_regression_data = client.load_server_regression_result()
    # client.close()
    # print("Get server device data successfully.")
    #
    # client_regression_data = load_regression_data(regression_result_dir)

    # get partition point and exit point
    # ep, pp = Optimize(threshold, B, server_regression_data, client_regression_data)
    ep_pp = client.optimize_pp_ep()
    client.close()
    ep = ep_pp[0]
    pp = ep_pp[1]
    # ep, pp = 1, 1
    print('Branch is %d, and partition point is %d' %(ep, pp))

    # infer left part
    inference_start = time.time()
    out = infer(CLIENT, ep, pp, images)

    print('Left part of model inference complete.')

    # save intermediate for RPC process
    intermediate = out.detach().numpy()
    np.save('intermediate.npy', intermediate)

    # client init
    client = client_start()

    info = file_info(FILENAME)
    print('Predict answer is: ' + client.partition(info, ep, pp))
    print('True label is: %d' % labels[0])

    end = time.time()
    print('Inference time: %f' % (end - inference_start))
    print('Total time: %f' %(end-start))
