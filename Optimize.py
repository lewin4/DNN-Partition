import numpy as np
import torch

from Time_Prediction import ServerTime, DeviceTime, partition_point_number, OutputSizeofPartitionLayer
from Resnet_Model_Pair import *
from torchsummary import summary
from utils import load_regression_data

# TODO： B？ 500KB/s for test
B = 4096000


def get_all_model_size():
    params_size_dict = {}
    for exit_branch in range(BRANCH_NUMBER - 1, -1, -1):
        for partition_point in range(partition_point_number[exit_branch]):
            L_model_name = "NetExit" + str(exit_branch + 1) + "Part" + str(partition_point + 1) + 'L'
            R_model_name = "NetExit" + str(exit_branch + 1) + "Part" + str(partition_point + 1) + 'R'
            net_L = eval(L_model_name)()
            summarydict, summ = summary(net_L, INPUT_SIZE, device="cuda" if torch.cuda.is_available() else "cpu")
            output_shape = next(reversed(summ.items()))[1]["output_shape"]
            img_shape = tuple(output_shape[1:])

            params_size_dict[L_model_name] = summarydict["Total params"]
            del net_L
            net_R = eval(R_model_name)()
            summarydict, _ = summary(net_R, img_shape, device="cuda" if torch.cuda.is_available() else "cpu")
            params_size_dict[R_model_name] = summarydict["Total params"]
    torch.save(params_size_dict, MODEL_DIR + "model_size.pth")
    return params_size_dict


def Optimize(latency_threshold, server_regression_data, client_regression_data):
    server_time_predictor = ServerTime(server_regression_data)
    device_time_predictor = DeviceTime(client_regression_data)

    params_size_dict = torch.load(MODEL_DIR + "model_size.pth")

    # 无穷大
    min_time = np.float64("inf")
    minep, minpp = 1, 1
    for exit_branch in range(BRANCH_NUMBER - 1, -1, -1):    #iter[2, 1, 0]
        times = []
        for partition_point in range(partition_point_number[exit_branch]):
            # model load time
            L_model_name = "NetExit" + str(exit_branch + 1) + "Part" + str(partition_point + 1) + 'L'
            R_model_name = "NetExit" + str(exit_branch + 1) + "Part" + str(partition_point + 1) + 'R'

            net_l = eval(L_model_name)()
            net_r = eval(R_model_name)()

            # immediate data size(bits)
            left_model_size = params_size_dict[L_model_name]
            right_model_size = params_size_dict[R_model_name]

            device_time, output_shape = device_time_predictor.predict_time(net_l, (INFER_BATCH_SIZE, *INPUT_SIZE))
            output_size = np.prod(tuple(output_shape)) * 32

            server_time = server_time_predictor.predict_time(net_r, output_shape)


            model_load_time = device_time_predictor.device_model_load(left_model_size) + \
                              server_time_predictor.server_model_load(right_model_size)

            total_time = device_time + server_time + model_load_time + \
                         output_size / B
            print("Time of ep {} and pp {}: {}".format(exit_branch, partition_point, total_time))
            times.append(total_time)
            if total_time <= min_time:
                min_time = total_time
                minep, minpp = exit_branch, partition_point

        # find min latency in this branch
        partition_point = times.index(min(times))

        if times[partition_point] < latency_threshold:
            return exit_branch + 1, partition_point + 1
    # if no ep and pp can satisfy latency required then return min-time's ep and pp
    print("No ep and pp can satisfy latency required ({:.5f}) then return min-time's ep and pp.".format(latency_threshold))
    return minep + 1, minpp + 1


if __name__ == '__main__':
    # net_l = NetExit4Part3L()
    # summarydict, summ = summary(net_l, INPUT_SIZE, device="cuda" if torch.cuda.is_available() else "cpu")
    client_regression_data = load_regression_data(REGRESSION_RESULT_DIR)
    # print(Optimize(1.0))
    Optimize(1.0, client_regression_data, client_regression_data)
