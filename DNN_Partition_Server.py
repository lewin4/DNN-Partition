import os
from utils import load_regression_data
import thriftpy2 as thriftpy
import numpy as np
import torch
from thriftpy2.rpc import make_server
from Branchy_Alexnet_Infer import infer
from config import *
from Optimize import Optimize


def server_start():
    partition_thrift = thriftpy.load('partition.thrift', module_name='partition_thrift')
    server = make_server(partition_thrift.Partition, Dispacher(), '127.0.0.1', 6000, client_timeout=3000)
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
    def load_server_regression_result(server_regression_result_dir: str = regression_result_dir):
        data = load_regression_data(server_regression_result_dir)
        return data

    @staticmethod
    def optimize_pp_ep():
        if os.path.isdir(regression_result_dir) and os.path.isdir(device_regression_result_dir):
            server_regression_data = load_regression_data(regression_result_dir)
            client_regression_data = load_regression_data(device_regression_result_dir)
        else:
            raise Exception("not find device or serve regression result.\n{}\n{}" \
                            .format(device_regression_result_dir, regression_result_dir))
        ep, pp = Optimize(threshold, B, server_regression_data, client_regression_data)
        ep_pp = [ep, pp]
        return ep_pp


if __name__ == '__main__':
    server_start()
    # D = Dispacher()
    # data = D.load_server_regression_result()
    # print(data)
