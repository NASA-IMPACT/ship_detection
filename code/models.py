import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

from keras import models, layers
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.models import load_model
from config import (
    UPSAMPLE_MODE,
    NET_SCALING,
    GAUSSIAN_NOISE,
    EDGE_CROP,
    NET_SCALING,
    MAX_TRAIN_STEPS,
    BATCH_SIZE,
    VALID_IMG_COUNT,
    NB_EPOCHS,
    IMG_SIZE
)

from keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    EarlyStopping,
    ReduceLROnPlateau
)


weight_path="../models/{}_iou_weights.orig.hdf5".format('seg_model')
def build_callbacks():
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=50,
                                       verbose=1, mode='min', epsilon=0.0001, cooldown=2, min_lr=1e-6)
    early = EarlyStopping(monitor="val_dice_coef",
                          mode="min",
                          patience=30)
    # return [checkpoint, early, reduceLROnPlat]
    return [checkpoint, reduceLROnPlat]


# Build U-Net model
def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)


def make_model_rcnn():
    from mrcnn.config import Config
    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize
    from mrcnn.model import log

    model_path = '../mask_rcnn_airbus_0022.h5'
    class DetectorConfig(Config):
        # Give the configuration a recognizable name
        NAME = 'airbus'

        GPU_COUNT = 1
        IMAGES_PER_GPU = 9

        BACKBONE = 'resnet50'

        NUM_CLASSES = 2  # background and ship classes

        IMAGE_MIN_DIM = 384
        IMAGE_MAX_DIM = 384
        RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        TRAIN_ROIS_PER_IMAGE = 64
        MAX_GT_INSTANCES = 14
        DETECTION_MAX_INSTANCES = 10
        DETECTION_MIN_CONFIDENCE = 0.95
        DETECTION_NMS_THRESHOLD = 0.0

        STEPS_PER_EPOCH = 15
        VALIDATION_STEPS = 10

        ## balance out losses
        LOSS_WEIGHTS = {
            "rpn_class_loss": 30.0,
            "rpn_bbox_loss": 0.8,
            "mrcnn_class_loss": 6.0,
            "mrcnn_bbox_loss": 1.0,
            "mrcnn_mask_loss": 1.2
        }

    class InferenceConfig(DetectorConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 9

    import tensorflow as tf
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode='inference',
                              config=inference_config,
                              model_dir='../data/')
    model.load_weights(model_path, by_name=True)

    return model


def predict_rcnn(model, img):
    predictions = model.detect(img)
    masks =  [pred['masks'] for pred in predictions]
    appended_masks = []
    for mask in masks:
        if mask.shape[2] == 0:
            appended_masks.append(np.zeros((mask.shape[0], mask.shape[1])))
        else:
            appended_masks.append((np.sum(mask, axis=-1) > 0) * 255)
    return appended_masks

def make_model(input_shape):
    if UPSAMPLE_MODE == 'DECONV':
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    input_img = layers.Input(input_shape[1:], name='RGB_Input')
    pp_in_layer = input_img
    if NET_SCALING is not None:
        pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)

    pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)

    c1 = layers.Conv2D(8, (3, 3), activation='relu',
                       padding='same')(pp_in_layer)
    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = upsample(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = upsample(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u8 = upsample(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u9 = upsample(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
    d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
    if NET_SCALING is not None:
        d = layers.UpSampling2D(NET_SCALING)(d)

    seg_model = models.Model(inputs=[input_img], outputs=[d])
    seg_model.summary()
    return seg_model


## intersection over union
def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    dice_p = binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
    return 1e-3 * dice_p


def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / K.sum(y_true)


def main():

    from data_process import (
        gen_data,
        create_aug_gen,
        make_image_gen
    )
    train_data, val_data = gen_data()
    train_gen = make_image_gen(train_data)
    t_x, t_y = next(train_gen)
    valid_x, valid_y = next(make_image_gen(val_data, VALID_IMG_COUNT))
    print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
    print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
    # seg_model = load_model(
    #     "../models/{}_weights.orig.hdf5".format('seg_model2'),
    #     custom_objects={
    #         'IoU': IoU,
    #     }
    # )
    seg_model = make_model(t_x.shape)
    seg_model.compile(
        optimizer=Adam(1e-2, decay=1e-6),
        loss=IoU,
        metrics=[dice_p_bce, dice_coef, 'binary_accuracy', true_positive_rate]
    )
    # seg_model = load_model(
    #     weight_path,
    #     custom_objects={
    #         'dice_p_bce': dice_p_bce,
    #         'dice_coef': dice_coef,
    #         'true_positive_rate': true_positive_rate
    #     }
    # )
    step_count = min(MAX_TRAIN_STEPS, t_x.shape[0] // BATCH_SIZE)
    loss_history = [seg_model.fit_generator(
        train_gen,
        steps_per_epoch=step_count,
        epochs=NB_EPOCHS,
        validation_data=(valid_x, valid_y),
        callbacks=build_callbacks(),
        workers=1  # the generator is not very thread safe
    )]
    print(loss_history)

    pred_y = model.predict(valid_x)
    # import ipdb; ipdb.set_trace()
    for i in range(valid_x):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 30))
        ax1.imshow(valid_x[i], cmap='gray')
        ax1.set_title('image')
        ax2.imshow(valid_y[i], cmap='gray_r')
        ax2.set_title('mask')
        ax2.imshow(pred_y[i], cmap='gray_r')
        ax2.set_title('prediction')
        plt.savefig(f'{i}.png')


def load_from_path(weight_file_path):
    seg_model = make_model((1, IMG_SIZE, IMG_SIZE, 3))
    seg_model.load_weights(weight_file_path)
    return seg_model


def predict(weight_file_path):
    from data_process import (
        gen_data,
        create_aug_gen,
        make_image_gen
    )
    train_data, val_data = gen_data()
    valid_x, valid_y = next(make_image_gen(train_data, VALID_IMG_COUNT))


    pred_y = seg_model.predict(valid_x)
    import ipdb; ipdb.set_trace()
    for i in range(400):

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
        ax1.imshow(valid_x[i], cmap='gray')
        ax1.set_title('image')
        ax2.imshow(valid_y[i,:,:,0].astype('uint8') * 255)
        ax2.set_title('mask')
        ax3.imshow((pred_y[i,:,:,0] * 255.0).astype('uint8'))
        ax3.set_title('prediction')
        plt.savefig(f'../data/predictions/{i}.png')


# if __name__ == '__main__':
#     # main()
#     # predict()
