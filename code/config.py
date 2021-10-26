
AUGMENT_BRIGHTNESS = False
BATCH_SIZE = 9
EDGE_CROP = 16
EXTENTS = {
    'san_fran': [-123.43, 37.71, -123.30, 37.85]
}
GAUSSIAN_NOISE = 0.1
# downsampling in preprocessing
IMG_SCALING = (1, 1)
IMG_SIZE = 768

# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 200

NB_EPOCHS = 500

# downsampling inside the network
NET_SCALING = None
PATH = '../../ship_detection_2/data/airbus-ship-detection/'
PRETRAINED = PATH + 'fine-tuning-resnet34-on-ship-detection/models/Resnet34_lable_256_1.h5'
SEGMENTATION_PATH = PATH + 'train_ship_segmentations_v2.csv'
TEST_PATH = PATH + 'test_v2/'
THRESHOLD = 0.5
TRAIN_PATH = PATH + 'train_v2/'
UPSAMPLE_MODE = 'SIMPLE'
WATERBODY_JSON = '../../ship_detection_2/data/stanford-sf-geojson.geojson'

IMG_DIM = 768
THRESHOLD = 0.5
# number of validation images to use
VALID_IMG_COUNT = 900
ZOOM_LEVEL = 14  # 16 is in pixel resolution == 2.4m


SHIP_AREA_MIN = 4000
SHIP_AREA_MAX = 50000
