import os
import os.path
import os
from itertools import product
import rasterio as rio
import cv2
import numpy as np
import fiona
import rasterio
import matplotlib.pyplot as plt

from rasterio import windows
from tensorflow.keras.models import load_model
from glob import glob
from osgeo import gdal, osr, ogr
from rasterio.warp import calculate_default_transform, reproject, Resampling
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from copy import deepcopy
from shapely.geometry import (
    shape, mapping, Point, Polygon, MultiPolygon, MultiPoint
)
from models import (
    dice_p_bce,
    dice_coef,
    true_positive_rate,
    IoU,
    make_model
)
from config import (
    WATERBODY_JSON,
    IMG_DIM,
    THRESHOLD
)


def predict(tif_path, model):

    predict_json = {
        "TYPE": "MultiPoint",
        "coordinates": [],
    }
    ds = gdal.Open(tif_path)
    img = ds.ReadAsArray()
    xoff, a, b, yoff, d, e = ds.GetGeoTransform()
    ds_proj = ds.GetProjectionRef()
    ds_srs = osr.SpatialReference(ds_proj)
    geogcs = ds_srs.CloneGeogCS()
    transform = osr.CoordinateTransformation(ds_srs, geogcs)
    reshaped_img = np.moveaxis(img[:-1, :, :], 0, -1)
    if reshaped_img.shape == (IMG_DIM, IMG_DIM, 3):
        segments = model.predict(
            np.expand_dims(reshaped_img, 0)
        )[0, :, :, 0]
        # plt.imshow(segments)
        # plt.savefig(tif_path[:-4] + '.png')
        segments = (segments > THRESHOLD).astype('uint8')

        for idx, ship in enumerate(regionprops(segments)):

            #if (ship.area >= SHIP_AREA_MIN and ship.area <= SHIP_AREA_MAX):
            x, y = (int(np.average([ship.bbox[0], ship.bbox[2]])),
                    int(np.average([ship.bbox[1], ship.bbox[3]])))

            # Get global coordinates from pixel x, y coords
            projected_x = a * y + b * x + xoff
            projected_y = d * y + e * x + yoff

            # Transform from projected x, y to geographic lat, lng
            (lat, lng, elev) = transform.TransformPoint(
                projected_x, projected_y
            )

            # Add ship to results cluster
            predict_json['coordinates'].append([lng, lat])

    return filter_predictions(predict_json)


def filter_predictions(predict_json):
    predictions = MultiPoint(predict_json['coordinates'])
    new_predictions = deepcopy(predict_json)
    new_predictions['coordinates'] = []
    with fiona.open(WATERBODY_JSON) as water_geoms:
        for pol in water_geoms:
            water_features = shape(pol['geometry'])
            for point in predictions:
                if point.within(water_features):
                    new_predictions['coordinates'].append([point.x, point.y])
    return new_predictions


def get_predictions(tif_path, seg_model):
    predict_json_list = []
    tile_path = create_tiles(tif_path)
    for tile in glob(tile_path + '/*.tif'):
        print(tile)
        predict_json_list.append(predict(tile, seg_model))
    return predict_json_list


def create_tiles(tif_image):

    out_path = f'../data/temp/{os.path.basename(tif_image)}_tile/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    output_filename = 'tile_{}-{}.tif'
    with rio.open(tif_image) as src:
        tile_width, tile_height = IMG_DIM, IMG_DIM

        meta = src.meta.copy()

        for window, transform in get_tiles(src, tile_width, tile_height):
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            outpath = os.path.join(out_path, output_filename.format(
                int(window.col_off), int(window.row_off)))
            with rio.open(outpath, 'w', **meta) as outds:
                outds.write(src.read(window=window))
    return out_path


def get_tiles(ds, width, height):
    ncols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, ncols, width), range(0, nrows, height))
    big_window = windows.Window(
        col_off=0, row_off=0, width=ncols, height=nrows)
    for col_off, row_off in offsets:
        window = windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=width,
            height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


def join_geojsons(json_list):
    predict_json = {
        "TYPE": "MultiPoint",
        "coordinates": [],
    }
    for item in json_list:
        if item['coordinates'] is not []:
            for points in item['coordinates']:
                predict_json['coordinates'].append(points)
    return predict_json


def scrape_planet_folder(data_path, weight_path):
    seg_model = make_model((1, IMG_DIM, IMG_DIM, 3))
    seg_model.load_weights(weight_path)
    predict_folder_name = data_path + '_predictions'
    if not os.path.exists(predict_folder_name):
        os.makedirs(predict_folder_name)
    for dirpath, dirnames, filenames in os.walk(data_path):
        for filename in [f for f in filenames if f.endswith("Visual.tif")]:
            jsons = get_predictions(
                os.path.join(dirpath, filename), seg_model,
            )
            combined_jsons = join_geojsons(jsons)
            visualize_prediction(
                os.path.join(dirpath, filename),
                combined_jsons,
                os.path.join(predict_folder_name, filename)
            )


def crs_to_4326(tif_image, dest_image):
    dst_crs = 'EPSG:4326'

    with rasterio.open(tif_image) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dest_image, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


def visualize_prediction(tif_image, predict_json, save_path):
    latlon = predict_json['coordinates']
    rc = []
    temp_file = '../data/RGB.byte.wgs84.tif'
    crs_to_4326(tif_image, temp_file)
    src = rasterio.open(temp_file)
    bounds = src.bounds
    plt.imshow(np.moveaxis(src.read(), 0, -1))
    for x, y in latlon:

        if (x >= bounds[0] and x <= bounds[2]) and\
           y >= bounds[1] and y <= bounds[3]:
            rc = rasterio.transform.rowcol(src.transform, x, y)
            plt.scatter(rc[1], rc[0], marker='o', alpha=.3, c='blue', linewidth=1)
    plt.savefig(save_path, dpi=1000)


if __name__ == '__main__':
    seg_model = make_model((1, IMG_DIM, IMG_DIM, 3))
    seg_model.load_weights('../models/seg_model_weights.best.hdf5')
    jsons = get_predictions('../data/sf_2.tif', seg_model)
    jsons = join_geojsons(jsons)
    visualize_prediction('../data/sf_2.tif', jsons, 'test2.png')
    # print(jsons)
