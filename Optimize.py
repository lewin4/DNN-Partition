import numpy as np
import os

from Time_Prediction import ServerTime, DeviceTime, partition_point_number, OutputSizeofPartitionLayer
from Resnet_Model_Pair import *
import time
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from utils import load_regression_data
from DNN_Partition_Client import client_start

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_all_model_size():
    params_size_dict = {}
    for exit_branch in range(BRANCH_NUMBER - 1, -1, -1):
        for partition_point in range(partition_point_number[exit_branch]):
            L_model_name = "NetExit" + str(exit_branch + 1) + "Part" + str(partition_point + 1) + 'L'
            R_model_name = "NetExit" + str(exit_branch + 1) + "Part" + str(partition_point + 1) + 'R'

            net_L = eval(L_model_name)().to(torch.device(device))
            summarydict, summ = summary(net_L, INPUT_SIZE, device=device)
            output_shape = next(reversed(summ.items()))[1]["output_shape"]
            img_shape = tuple(output_shape[1:])

            params_size_dict[L_model_name] = summarydict["Total params"]
            del net_L
            net_R = eval(R_model_name)().to(torch.device(device))
            summarydict, _ = summary(net_R, img_shape, device=device)
            params_size_dict[R_model_name] = summarydict["Total params"]
    torch.save(params_size_dict, MODEL_DIR + "model_size.pth")
    return params_size_dict


def Optimize(latency_threshold, B, server_regression_data, client_regression_data):
    server_time_predictor = ServerTime(server_regression_data)
    device_time_predictor = DeviceTime(client_regression_data)
    print("BW: {}".format(B))

    if os.path.exists(MODEL_DIR + "model_size.pth"):
        params_size_dict = torch.load(MODEL_DIR + "model_size.pth")
        print("Find " + MODEL_DIR + "model_size.pth")
    else:
        params_size_dict = get_all_model_size()

    # 无穷大
    min_time = np.float64("inf")
    minep, minpp = 1, 1
    for exit_branch in range(BRANCH_NUMBER - 1, -1, -1):    #iter[2, 1, 0]
        times = []
        for partition_point in range(partition_point_number[exit_branch]):
            # model load time
            L_model_name = "NetExit" + str(exit_branch + 1) + "Part" + str(partition_point + 1) + 'L'
            R_model_name = "NetExit" + str(exit_branch + 1) + "Part" + str(partition_point + 1) + 'R'

            net_l = eval(L_model_name)().to(torch.device(device))
            net_r = eval(R_model_name)().to(torch.device(device))

            # immediate data size(bits)
            left_model_size = params_size_dict[L_model_name]
            right_model_size = params_size_dict[R_model_name]

            device_time, output_shape = device_time_predictor.predict_time(net_l,
                                                                           (INFER_BATCH_SIZE, *INPUT_SIZE),
                                                                           device)
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

    # client init
    client = client_start()

    print("Try to request server device data.")
    server_regression_data = client.load_server_regression_result()
    client.close()
    print("Get server device data successfully.")

    # 创建Summary-writer
    time_local = time.localtime()
    time_str = str(time_local[1]) + "m" + str(time_local[2]) + "d" + str(time_local[3]) + "h" + str(
        time_local[4]) + "m" + str(time_local[5]) + "s"
    writer_dir = "./logs/optimize/" + time_str + "/"
    summary_writer = SummaryWriter(writer_dir)

    for B in range(6000000, 10000000, 1000):
        ep, pp = Optimize(1.0, B, server_regression_data, client_regression_data)
        summary_writer.add_scalar("ep", ep, B)
        summary_writer.add_scalar("pp", pp, B)
        print("Ep: {}, Pp: {}".format(ep, pp))
        import time
        l_net = NetExit4Part1L().to(torch.device(device)).eval()
        r_net = NetExit4Part1R().to(torch.device(device)).eval()
        x = torch.rand(1, 3, 192, 256).to(torch.device(device))
        start_time = time.time()
        out = l_net(x)
        out = r_net(out)
        end_time = time.time()
        print("time: {}".format(end_time-start_time))
        print("Out: {}".format(out))