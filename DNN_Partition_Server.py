import os
from utils import load_regression_data
import thriftpy2 as thriftpy
import numpy as np
import torch
from thriftpy2.rpc import make_server
from Branchy_Alexnet_Infer import infer
from config import *


def server_start():
    partition_thrift = thriftpy.load('partition.thrift', module_name='partition_thrift')
    server = make_server(partition_thrift.Partition, Dispacher(), '202.199.116.243', 6000, client_timeout=3000)
    print('Thriftpy server is listening...')
    server.serve()


class Dispacher(object):
    @staticmethod
    def partition(file, ep, pp):
        for filename, content in file.items():
            with open('recv_' + filename, 'wb') as f:
                f.write(content)

        readed = np.load('recv_intermediate.npy')
        input = torch.from_numpy(readed)
        out = infer(SERVER, ep, pp, input)
        prob = torch.exp(out).detach().numpy().tolist()[0]
        pred = str((prob.index(max(prob)), max(prob)))
        print("Result have been returned.")
        return pred

    @staticmethod
    def load_server_regression_result(server_regression_result_dir: str = REGRESSION_RESULT_DIR):
        data = load_regression_data(server_regression_result_dir)
        return data


if __name__ == '__main__':
    server_start()
    # D = Dispacher()
    # data = D.load_server_regression_result()
    # print(data)
