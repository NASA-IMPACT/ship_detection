import cv2
import fiona
import fiona.transform
import gc
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
import rasterio as rio
import skimage.transform as st
import subprocess

from area import area
from copy import deepcopy
from glob import glob
from itertools import product
from datetime import datetime
from rasterio import windows
from rasterio.crs import CRS
from rasterio.windows import Window
from skimage.io import ImageCollection
from skimage.measure import regionprops
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from PIL import Image
from tensorflow.keras.models import load_model
from zipfile import ZipFile
from keras.preprocessing.image import ImageDataGenerator
from multiprocessing import Pool

from rasterio.warp import (
    aligned_target,
    calculate_default_transform,
    reproject,
    Resampling,
)
from shapely.geometry import (
    mapping,
    mapping,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    shape,
)

from models import (
    dice_coef,
    dice_p_bce,
    IoU,
    make_model_rcnn,
    predict_rcnn,
    true_positive_rate,
)
from config import (
    IMG_DIM,
    THRESHOLD,
    WATERBODY_JSON,
    BATCH_SIZE,
    SHIP_AREA_MIN,
    SHIP_AREA_MAX,
    MAX_ASPECT_RATIO,
)


def read_resize_img(tif_path, dim=(768, 768)):
    """
    tif_path: location of image to read and resize
    dim: dimension to resize to
    returns: resized image
    """
    ds = rasterio.open(tif_path, 'r')
    img = ds.read([1, 2, 3]) # just read first three bands
    img = np.moveaxis(img, 0, -1)
    ds.close()
    meta = ds.meta
    del ds
    gc.collect()
    return (cv2.resize(img, dim), meta)


def predict_batch(tifs_path, model) -> list:
    """
    tifs_path: path to tifs to batch predict
    model: model to use for prediction
    returns: list of predictions in geojson format
    """
    image_collection = ImageCollection(
        tifs_path + '/*.tif',
        load_func=read_resize_img
    )
    ic_batches = np.array_split(
        image_collection,
        int(len(image_collection) / BATCH_SIZE)
    )
    #  it returns l % n sub-arrays of size l//n + 1 and the rest of size l//n.

    segments_list = []
    meta_list = []
    geojsons = []
    for idx, batch in enumerate(ic_batches):
        if len(batch) is not BATCH_SIZE:
            batch = batch[:BATCH_SIZE]
        images, metas = zip(*batch)
        batch_segments = predict_rcnn(
            model, images
        )
        segments_list += batch_segments
        meta_list += metas
        del batch_segments
    with Pool(processes=4) as pool:
        geojsons = pool.starmap(
            mask_to_geojson,
            zip(segments_list, meta_list)
        )
        pool.close()
        gc.collect()

    if not os.path.exists(f'./shapes/{os.path.basename(tifs_path)}'):
        os.mkdir(f'./shapes/{os.path.basename(tifs_path)}')

    for idx, geojson in enumerate(geojsons):
        if geojson['features'] != []:

            with open(
                f'./shapes/{os.path.basename(tifs_path)}/{idx}.geojson', 'w'
                ) as shapefile:
                json.dump(geojson, shapefile)


def mask_to_geojson(segment, meta):
    """
    batch_segments: list of segmentation masks
    num_threads: number of threads to use for creating geojsons
    """
    predict_json = {
        "TYPE": "MultiPolygon",
        "coordinates": [],
    }

    segment = (segment > 125).astype('uint8')
    w_0, h_0 = meta['width'], meta['height']
    segment = cv2.resize(segment, (w_0, h_0))  # reshape to original
    contours = cv2.findContours(
        segment,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours[0] if len(contours) == 2 else contours[1]

    for idx, contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        sides = rect[1]
        # check for aspect ratio
        if min(sides) / max(sides) > MAX_ASPECT_RATIO:
            continue
        box = cv2.boxPoints(rect)
        box = [[y, x] for x, y in box]
        boxpoints = zip(*box)
        xs, ys = rasterio.transform.xy(meta['transform'], *boxpoints)
        lats, lons = fiona.transform.transform(
            meta['crs'].to_proj4(),
            '+init=epsg:4326',
            xs,
            ys
        )
        zipped_coords = [[lat, lon] for lat, lon in zip(lats, lons)]
        zipped_coords.append([lats[0], lons[0]])
        predict_json['coordinates'].append(zipped_coords)
    return gen_geojson(predict_json)


def predict(tif_path, model):

    predict_json = {
        "TYPE": "MultiPolygon",
        "coordinates": [],
    }
    ds = rasterio.open(tif_path, 'r')
    img = ds.read([1, 2, 3]) # just read first three bands
    w_0, h_0 = img.shape[1:] # preserve original width, height
    img = np.moveaxis(img, 0, -1)
    reshaped_img = cv2.resize(img,(IMG_DIM, IMG_DIM))

    if reshaped_img.shape == (IMG_DIM, IMG_DIM, 3):

        segments = predict_rcnn(
            model, np.expand_dims(reshaped_img, 0)
        )
        segments = (segments > 125).astype('uint8')
        segments = cv2.resize(segments, (w_0, h_0))  # reshape to original
        contours = cv2.findContours(segments,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for idx, contour in enumerate(contours):
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = [[y, x] for x, y in box]
            boxpoints = zip(*box)
            xs, ys = rasterio.transform.xy(ds.transform, *boxpoints)
            lats, lons = fiona.transform.transform(
                ds.crs.to_proj4(),
                '+init=epsg:4326',
                xs,
                ys
            )
            zipped_coords = [[lat, lon] for lat, lon in zip(lats, lons)]
            zipped_coords.append([lats[0], lons[0]])
            predict_json['coordinates'].append(zipped_coords)
    return gen_geojson(predict_json, tif_path)

def save_preds(img, preds):
    plt.imshow(img)
    plt.imshow(preds, alpha=0.5)
    plt.savefig('test.png')
    plt.close()

def gen_geojson(predict_json):

    geojson_schema = {
                  "type": "FeatureCollection",
                  "features": []
    }
    for idx, detection in enumerate(predict_json['coordinates']):
        polygon = {
                    "type":"Polygon",
                    "coordinates": [detection]
                }
        area_m2 = area(polygon)
        if not (area_m2 >= SHIP_AREA_MIN and area_m2 <= SHIP_AREA_MAX):
            # if the contour is too small or large to be a ship, ignore it
           continue
        geojson_schema['features'].append(
            {
                "type": "Feature",
                "properties": {},
                "geometry": polygon
            }
        )

    return geojson_schema

def filter_predictions(predict_json, tif_path):

    predictions = MultiPolygon([Polygon(vertices) for vertices in predict_json['coordinates']])
    new_predictions = deepcopy(predict_json)
    new_predictions['coordinates'] = []
    geojson_schema = {
                  "type": "FeatureCollection",
                  "features": []
    }

    tif_path = os.path.basename(tif_path)
    rowcol, date = tif_path.split('_')
    date = date.split('T')[0]
    from fiona.crs import from_epsg
    with fiona.open(WATERBODY_JSON) as water_geoms:
        for idx, point in enumerate(predictions):
            idx = f'{rowcol}i{idx}'

            shapefile_path = f'./shapes/{idx}_{date}'

            opts = {
                'schema': {'geometry': 'Polygon', 'properties': {}}
            }
            lng1, lat1, lng2, lat2 = point.bounds
            shape_json = mapping(point)
            poly = [
                        [
                        [
                            lng1,
                            lat1
                        ],
                        [
                            lng2,
                            lat1
                        ],
                        [
                            lng2,
                            lat2
                        ],
                        [
                            lng1,
                            lat2
                        ],
                        [
                            lng1,
                            lat1
                        ]
                        ]
                    ]
            print('poly', poly)
            geojson_schema['features'] = [{"type": "Feature",
                    "properties": {},'geometry': {'type':'Polygon', 'coordinates': predict_json['coordinates']}}]
            with open(f'{shapefile_path}.geojson', 'w') as shapefile:
                json.dump(geojson_schema, shapefile)
            args = ['ogr2ogr', '-f', 'ESRI Shapefile', f'{shapefile_path}.shp', f'{shapefile_path}.geojson']
            subprocess.call(args)
            file_list = set(glob(f'{shapefile_path}.*'))
            file_list -= set([f'{shapefile_path}.geojson'])
            file_list = list(file_list)
            zip_filename = f'{shapefile_path.replace(".tif", "")}.zip'
            zipObj = ZipFile(f'{shapefile_path}.zip', 'w')
            [zipObj.write(file, os.path.basename(file)) for file in file_list]
            zipObj.close()
            new_predictions['coordinates'].append(point)
    return new_predictions

def get_predictions(tif_path, seg_model):
    predict_json_list = []

    tile_path = create_tiles(tif_path)
    for tile in glob(tile_path + '/*.tif'):
        print(tile)
        predict_json_list.append(predict(tile, seg_model))
    return predict_json_list

def create_tiles(tif_image, scale=1.5):

    out_path = f'../data/temp/tiled_{os.path.basename(tif_image)}/'
    tif_path = os.path.basename(tif_image)
    tif_path = tif_path[:8]
    date = tif_path.split('_')[0]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    output_filename = '{}x{}_{}T000000.tif'
    with rio.open(tif_image) as src:
        tile_width, tile_height = int(IMG_DIM * scale), int(IMG_DIM * scale)

        meta = src.meta.copy()

        for window, transform in get_tiles(src, tile_width, tile_height):
            meta['transform'] = transform
            meta['width'], meta['height'] = tile_width, tile_height
            outpath = os.path.join(out_path, output_filename.format(
                int(window.col_off), int(window.row_off), date))
            with rio.open(outpath, 'w', **meta) as outds:
                outds.write(src.read(window=window))
    return out_path


def create_tiles_wgs84(args):
    tif_image = args[0]
    out_path = args[1]
    dst_crs = CRS({'init': 'EPSG:4326'})
    out_path = f'{out_path}/{os.path.basename(tif_image)}_tile/'
    out_path_4326 = out_path
    tif_path = os.path.basename(tif_image)
    tif_path = tif_path[:8]
    date = tif_path.split('_')[0]

    if not os.path.exists(out_path_4326):
        os.makedirs(out_path_4326)
    output_filename = '{}x{}_{}T000000.tif'

    with rio.open(tif_image) as src:
        meta_4326 = src.meta.copy()
        for window, transform in get_tiles(src.meta, IMG_DIM, IMG_DIM):
            dst_transform, width, height = calculate_default_transform(
            src.crs, dst_crs, window.width, window.height, *rasterio.windows.bounds(window, transform))

            meta_4326.update({
                'crs': src.crs,
                'transform': transform,
                'width': window.width,
                'height': window.height,
                'count': 3,
                'nodata': 0,
            })
            outpath_4326 = os.path.join(out_path_4326, output_filename.format(
                window.col_off, window.row_off, date))

            with rio.open(outpath_4326, 'w', **meta_4326) as outds:
                for i in range(1, 4):
                    reproject(
                        source=src.read(i, window=window),
                        destination=rasterio.band(outds, i),
                        src_transform=transform,
                        src_crs=src.crs,
                        resampling=Resampling.nearest
                    )
    return out_path_4326


def get_tiles(meta, width, height):
    ncols, nrows = meta['width'], meta['height']
    big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
    offsets = product(range(0, ncols, width), range(0, nrows, height))
    for col_off, row_off in offsets:
        window = windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=int(width),
            height=int(height)).intersection(big_window)

        transform = windows.transform(window, meta['transform'])
        yield window, transform


def join_geojsons(json_list):
    predict_json = {
        "TYPE": "MultiPolygon",
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
    rcnn_model = make_model_rcnn()
    jsons = get_predictions('files/PSScene3Band/20210412_083128_05_2307/visual/20210412_083128_05_2307_3B_Visual.tif', rcnn_model)
