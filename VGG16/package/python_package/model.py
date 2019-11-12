from __future__ import print_function
import keras
# from keras.datasets import cifar10
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D
from tensorflow.python.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,Flatten,Dropout, MaxPooling2D,Conv2D,Input,GlobalAveragePooling2D
from tensorflow.python.keras.models import load_model
import cv2
import numpy as np
from config import config
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tqdm import tqdm

from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.applications.vgg16 import VGG16


#load dataset
# (x_train,y_train),(x_test,y_test)= cifar10.load_data()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = ""
image_size = config.image_size

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



# # image normalize
def data_normalize():
    x_train_norm = x_train.astype('float32')/255.0
    x_test_norm = x_test.astype('float32')/255.0
    # one-hot encoding:
    y_train_oneHot = np_utils.to_categorical(y_train)
    y_test_oneHot  = np_utils.to_categorical(y_test)
    print(y_train_oneHot.shape)
    return x_train_norm,x_test_norm,y_train_oneHot,y_test_oneHot

def data_nomalize2():
    datagen = ImageDataGenerator(rescale=1. /255)

    return datagen

def add_new_classifier(base_model):
    FC_SIZE = config.FC_SIZE
    # FC_SIZE = 512
    n_classes = config.nb_classes
    x = base_model.output

    # x = Conv2D(filters=512,kernel_size=(5,5), strides=(2,2), activation="relu")(x)
    # x = Flatten()(x)
    # x = Dense(FC_SIZE, activation="relu")(x)
    # predictions = Dense(n_classes, activation="softmax")(x)
    # model = Model(inputs=base_model.input, outputs=predictions)

    x = GlobalMaxPooling2D()(x)
    x = Dropout(config.dropout)(x)
    # x = Dense(FC_SIZE, activation='relu')(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Pooling_size = 2
    # # x = GlobalAveragePooling2D()(x)
    # # x = Flatten(name='flatten')(x)
    # # x = Dropout(0.7)(x)
    # # x = Dense(FC_SIZE, activation='relu')(x)  #
    # # # predictions = Dense(n_classes, activation='sigmoid')(x)
    # # predictions = Dense(n_classes, activation='softmax',name='mydence')(x)
    # # x = AveragePooling2D((Pooling_size, Pooling_size), strides=(Pooling_size, Pooling_size), name='avg_pool')(x)
    # inputs = Input(shape = (image_size, image_size, 3))
    # x = base_model.output
    # x = Conv2D(256,kernel_size = (3,3),strides = (2,2),padding = 'same',activation= 'relu')(x)
    # x = MaxPooling2D((Pooling_size, Pooling_size), strides=(Pooling_size, Pooling_size))(x)
    # x = Flatten(name='flatten')(x)
    # #x = Dense(256,activation='relu')(x)
    # x = Dense(512, activation='relu')(x)
    # #x = Dense(256,activation= 'relu')(x)
    # predictions = Dense(n_classes, name='dence100',activation='softmax')(x)
    # model = Model(base_model.input,predictions)
    return model

def set_up_model(base_model,x_train,y_train,x_test,y_test):
    model = add_new_classifier(base_model)

    for layer in base_model.layers:
        layer.trainable = False
    print(model.summary())
    sgd = SGD(lr=0.001,decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001,decay=1e-6),metrics = ['accuracy'])

    tensorboard = TensorBoard(log_dir='./log')
    callback_lists = [tensorboard]

    # model.fit_generator(
    #     x_train,
    #     steps_per_epoch=train_size / self.batch_size,
    #     # steps_per_epoch=train_size / self.batch_size * 4,
    #     epochs=self.epochs,
    #     validation_data=validation_generator,
    #     validation_steps=val_size / self.batch_size,
    #     # verbose=1,
    #     # initial_epoch=11,
    #     callbacks=checkpoint
    # )

    model.fit(
        x = x_train,
        y = y_train,
        batch_size=10,
        epochs=10,
        validation_data=(x_test,y_test),
        shuffle='True',
        callbacks=callback_lists
    )

    model.save('resnet.h5')

    return model

def get_files(root,mode, label_path):
    # pwd = os.path.dirname(os.path.abspath('__file__'))
    # f = open(pwd + "/class_index.txt", "rb")
    f = open(label_path, "rb")
    chinese_labels = []
    num_labels = []

    lines = f.readlines()
    for line in lines:
        curLine = line.split(b' ')
        chinese_labels.append(curLine[0])
        num_labels.append(int(curLine[1]))

    #for test
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename":files})
        return files
    elif mode != "test":
        #for train and val
        all_data_path,labels = [],[]
        # # lambda x: root + x, os.listdir(root)
        # # image_folders = list(map(lambda x:root+x,os.listdir(root)))
        # image_folders = list(map(lambda x: root + '/'+ x, os.listdir(root)))
        # # print(image_folders)
        # all_images = list(chain.from_iterable(list(map(lambda x:glob(x+"/*"),image_folders))))
        # # print(all_images)
        # print("loading train dataset")
        # imgs_list = []
        # for file in tqdm(all_images):
        train_imgs_list = []
        test_imgs_list = []
        image_folders = list(map(lambda x: root + '/'+ x, os.listdir(root)))
        for it in image_folders:
            if(os.path.isdir(it)):
                cur_dir_imgs = list(map(lambda x: glob(x + "/*"),  os.listdir(it)))
                for file in cur_dir_imgs:
                    print(file)
                    cur_img = cv2.imread(imagePath)
                    imgs_list.append(cur_img)
                    all_data_path.append(file)
                    # labels.append(int(file.split("/")[-2]))

                    cur_chinese_label = file.split("/")[-2]
                    index_cur = chinese_labels.index(cur_chinese_label)
                    print(index_cur)
                    labels.append(int(num_labels[index_cur]))

        all_files = pd.DataFrame({"filename":all_data_path,"label":labels})
        return all_files
    else:
        print("check the mode please!")

if __name__ == '__main__':
    batch_size = config.batch_size
    num_classes = config.nb_classes
    epochs = config.epochs
    data_augmentation = True
    pwd = os.path.dirname(os.path.abspath(__file__))

    base_model = VGG16(include_top=False,
                                   # weights= pwd + '/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
                          weights=pwd + '/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                       # weights="imagenet",
                                   input_shape=(image_size, image_size, 3))
    # base_model = Model(inputs=base_model.input, outputs=base_model.get_layer(index=125).output)
    model = add_new_classifier(base_model)
    # for layer in base_model.layers:
    #     layer.trainable = False
    print("VGG16!!!")
    print(model.summary())
    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=config.lr, decay=1e-6), metrics=['accuracy'])

    # img_rows = image_size
    # img_cols = image_size
    # x_train = np.array([cv2.resize(img, (img_rows, img_cols)) for img in x_train[:, :, :, :]])
    # x_test = np.array([cv2.resize(img, (img_rows, img_cols)) for img in x_test[:, :, :, :]])
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    # model = LSUVinit(model,x_train[:batch_size,:,:,:])
    # tbCallBack = TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True)
    save_model_url = pwd + config.save_model_url
    checkpoint = ModelCheckpoint(save_model_url, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    datagen = ImageDataGenerator(
        rescale=1/255.0,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,    # randomly flip images
        validation_split=0.2
        )
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    train_dir = config.train_dir
    # for f in os.listdir(train_dir):
    #     print(f.decode(encoding='gbk'))
    print(os.listdir(train_dir))
    train_generator = datagen.flow_from_directory(train_dir,
                                                  target_size=(image_size, image_size),
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  seed=666,
                                                  # classes=dirs,
                                                  class_mode='categorical',
                                                  subset='training')
    val_generator = datagen.flow_from_directory(train_dir,
                                                target_size=(image_size, image_size),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                seed=666,
                                                # classes=dirs,
                                                class_mode='categorical',
                                                subset='validation')
    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples // batch_size,
                        epochs=epochs,
                        validation_data=val_generator,
                        # callbacks=[checkpoint]
                        )
    # best_model = load_model(save_model_url)
    # MODEL_PATH = config.save_model_pb
    # tf.keras.experimental.export_saved_model(best_model, MODEL_PATH + '/SavedModel')

    # best_model.summary()
    # MODEL_PATH = os.environ['MODEL_INFERENCE_PATH']
    

    # batch_size = config.batch_size
    # num_classes = config.nb_classes
    # epochs = config.epochs
    # data_augmentation = True
    # pwd = os.path.dirname(os.path.abspath(__file__))
    #
    # # The data, shuffled and split between train and test sets:
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # print('x_train shape:', x_train.shape)
    # print('y_train shape:', y_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')
    # # Convert class vectors to binary class matrices.
    # train_size = 50000
    # test_size = 10000
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    #
    # base_model = InceptionResNetV2(include_top=False,
    #                                weights=pwd + '/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                                input_shape=(image_size, image_size, 3))
    # # base_model = Model(inputs=base_model.input, outputs=base_model.output)
    # model = add_new_classifier(base_model)
    # for layer in base_model.layers:
    #     layer.trainable = False
    # # print(model.summary())
    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=config.lr, decay=1e-6), metrics=['accuracy'])
    #
    # img_rows = image_size
    # img_cols = image_size
    # x_train = np.array([cv2.resize(img, (img_rows, img_cols)) for img in x_train[:, :, :, :]])
    # x_test = np.array([cv2.resize(img, (img_rows, img_cols)) for img in x_test[:, :, :, :]])
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    # # model = LSUVinit(model,x_train[:batch_size,:,:,:])
    # # tbCallBack = TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True)
    # save_model_url = pwd + config.save_model_url
    # # checkpoint = ModelCheckpoint(save_model_url, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #
    # datagen = ImageDataGenerator(
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    #     width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images
    # # Compute quantities required for feature-wise normalization
    # # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(x_train)
    # # Fit the model on the batches generated by datagen.flow().
    # model.fit_generator(datagen.flow(x_train, y_train,
    #                                  batch_size=batch_size),
    #                     steps_per_epoch=x_train.shape[0] // batch_size,
    #                     epochs=epochs,
    #                     validation_data=(x_test, y_test),
    #                     # callbacks=[checkpoint]
    #                     )
    # best_model = load_model(save_model_url)
    # # best_model.summary()
    # # MODEL_PATH = os.environ['MODEL_INFERENCE_PATH']
    # MODEL_PATH = config.save_model_pb
    # tf.keras.experimental.export_saved_model(best_model, MODEL_PATH + '/SavedModel')





