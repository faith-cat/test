import os
class DefaultConfigs(object):
    # hardware
    gpu = "0"

    # numerical configs
    NB_IV3_LAYERS_TO_FREEZE = 172
    image_size = 224
    batch_size = 32
    # nb_classes = 2
    RGB = True
    lr = 0.0001
    # epochs = 1

    FC_SIZE = 512
    dropout = 0.3
    val_size = 0.2

    # path var
    # train_dir = "/home/yangchuan/imageseg/errors/1"
    # # train_dir = os.environ['IMAGE_TRAIN_INPUT_PATH']
    # save_model_url = './saved_model/test.h5'
    # model_url = ''
    # save_model_pb = './saved_model'
    # # save_model_pb = os.environ['MODEL_INFERENCE_PATH']

    # test modify
    epochs = 8
    nb_classes = 100
    # train_dir = "/home/yangchuan/imageseg/errors/1"
    train_dir = os.environ['IMAGE_TRAIN_INPUT_PATH']
    save_model_url = '/saved_model/test.h5'
    model_url = ''
    # save_model_pb = './saved_model'
    save_model_pb = os.environ['MODEL_INFERENCE_PATH']


    # 1. string configs
    # train_data = "../data/train/"
    # val_data = "../data/val/"  # if exists else use train_test_split to generate val dataset
    # test_data = "../data/all/traffic-sign/test/00003/"  # for competitions
    # model_name = "NASNetMobile"
    # weights_path = "./checkpoints/model.h5"  # save weights for predict
    #
    # # 2. numerical configs
    # lr = 0.001
    # epochs = 50
    # num_classes = 2
    # image_size = 512
    # batch_size = 16
    # channels = 3
    # gpu = "0"


config = DefaultConfigs()
