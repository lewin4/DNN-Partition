import socket

CLIENT, SERVER = range(2)
IMAGE_DIM = 32
NUM_CLASSES = 6
BRANCH_NUMBER = 4
INFER_BATCH_SIZE = 1
BATCH_SIZE = 64
REGRESSION_RESULT_DIR = "regression_output/regression_result/"
regression_type = ["conv", "relu", "pool", "bn", "fc", "load"]
INPUT_SIZE = (3, 192, 256)
OUTPUT_DIR = './resnet18_data_out/'
MODEL_DIR = OUTPUT_DIR + 'models/'  # model checkpoints
if socket.gethostname() == 'LAPTOP-5G1BF2CK':
    TRAIN_DATASET = r"D:\Code\data\sewage\classification_aug"
    TEST_DATASET = r"D:\Code\data\sewage\test_dataset"
elif socket.gethostname() == 'DESKTOP-D6L914M':
    TRAIN_DATASET = r"E:\LY\data\classification_aug"
    TEST_DATASET = r"E:\LY\data\test_dataset"
elif socket.gethostname() == "raspberrypi":
    TRAIN_DATASET = "/home/e303/Code/data/classification_aug"
    TEST_DATASET = "/home/e303/Code/data/test_dataset"
if socket.gethostname() == 'wlj':
    TRAIN_DATASET = r"D:\Code\data\sewage\classification_aug"
    TEST_DATASET = r"D:\Code\data\sewage\test_dataset"
if socket.gethostname() == 'ubuntu':
    TRAIN_DATASET = "/home/e303/ly/data/classification_aug"
    TEST_DATASET = "/home/e303/ly/data/test_dataset"
