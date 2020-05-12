import numpy as np
import skimage.io
import json
import fiona
import rasterio

from osgeo import gdal, osr, ogr
from rasterio.warp import calculate_default_transform, reproject, Resampling
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from copy import deepcopy
from shapely.geometry import (
    shape, mapping, Point, Polygon, MultiPolygon, MultiPoint
)

SHIP_AREA_MAX = 10000
SHIP_AREA_MIN = 600
WATERBODY_JSON = '../data/stanford-sf-geojson.geojson'


def predict(tif_path):

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
    red = np.moveaxis(img, 0, -1)[:, :, 0]
    transform = osr.CoordinateTransformation(ds_srs, geogcs)
    mask = red > 125
    segments_fz = felzenszwalb(np.dstack((mask, mask, mask)),
                               scale=5000,
                               sigma=3.1,
                               min_size=25)

    for idx, ship in enumerate(regionprops(segments_fz)):

        if (ship.area >= SHIP_AREA_MIN and ship.area <= SHIP_AREA_MAX):

            #predict_json['ship_count'] += 1

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


def get_heatmap(predict_json, tif_underlay):
    pass


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


def visualize_prediction(tif_image, predict_json):
    import matplotlib.pyplot as plt
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
            #rc.append(rasterio.transform.rowcol(src.transform, x, y))
            rc = rasterio.transform.rowcol(src.transform, x, y)
            plt.scatter(rc[1], rc[0], c='blue', s=4)
    plt.show()


def main():
    tif_image = '../data/sf_2.tif'
    predict_json = predict(tif_image)
    visualize_prediction(tif_image, predict_json)


if __name__ == '__main__':
    main()
