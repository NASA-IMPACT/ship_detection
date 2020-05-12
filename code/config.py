BATCH_SIZE = 16
EDGE_CROP = 16
NB_EPOCHS = 500
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
# downsampling inside the network
NET_SCALING = None
# downsampling in preprocessing
IMG_SCALING = (1, 1)
# number of validation images to use
VALID_IMG_COUNT = 900
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 200
AUGMENT_BRIGHTNESS = False

PATH = '../data/airbus-ship-detection/'
TRAIN_PATH = PATH + 'train_v2/'
TEST_PATH = PATH + 'test_v2/'
SEGMENTATION_PATH = PATH + 'train_ship_segmentations_v2.csv'
PRETRAINED = PATH + 'fine-tuning-resnet34-on-ship-detection/models/Resnet34_lable_256_1.h5'
EXCLUDE_LIST = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images
