import mercantile
import requests
import numpy as np
import rasterio

from models import load_from_path
from io import BytesIO
from PIL import Image
from planet_downloader import PlanetDownloader
from skimage.measure import regionprops

EXTENTS = {
  'san_fran': [-123.43, 37.71, -123.30, 37.85]
}

THRESHOLD = 0.5
WEIGHT_FILE = '../weights/iou_model.hdf5'
ZOOM_LEVEL = 14
IMG_SIZE = 768
TILE_SIZE = 256
WMTS_URL = f"https://tiles1.planet.com/data/v1/PSScene3Band/{{}}/{ZOOM_LEVEL}/{{}}/{{}}.png?api_key={{}}"


class Infer:

    def __init__(self, date, weight_path=WEIGHT_FILE, credential=None):
        self.start_date_time, self.end_date_time = self.prepare_date(date)
        self.weight_path = weight_path
        self.model = self.prepare_model()
        self.credential = credential
        self.planet_downloader = PlanetDownloader(credential)


    def prepare_date(self, date):
        return [f"{date}T00:00:00Z", f"{date}T23:59:59Z"]


    def prepare_model(self):
        return load_from_path(self.weight_path)


    def infer(self):
        for location, extent  in EXTENTS.items():

          items = self.planet_downloader.search_ids(
            extent, self.start_date_time, self.end_date_time
          )
          for item in items:
            images = self.prepare_dataset(item['tiles'], item['id'])
            predictions = self.model.predict(images)
            columns, rows = [elem[1] - elem[0] for elem in item['tiles']]
            predictions = predictions.reshape((rows, columns, IMG_SIZE, IMG_SIZE))
            pred_list = self.xy_to_latlon(
              predictions, rows, columns, item['coordinates']
            )

    def prepare_dataset(self, tile_range, tile_id):
        x_indices, y_indices = tile_range
        images = list()
        for x_index in list(range(*x_indices)):
          for y_index in list(range(*y_indices)):
            response = requests.get(
              WMTS_URL.format(tile_id, x_index, y_index, self.credential)
            )

            response.raise_for_status()
            img = np.asarray(
              Image.open(BytesIO(response.content)).resize(
                  (IMG_SIZE, IMG_SIZE)
                ).convert('RGB')
              )
            images.append(img)
        return np.asarray(images)


    def prepare_geojson(self):
        pass

    def xy_to_latlon(self, grid_list, rows, cols, bounds):
        transform = rasterio.transform.from_bounds(
          *bounds, TILE_SIZE * cols, TILE_SIZE * rows
        )
        new_xs = list()
        new_ys = list()
        for row in range(rows):
            for col in range(cols):
                if grid_list[row][col].max() >= THRESHOLD:
                  segments = (grid_list[row][col] > THRESHOLD).astype('uint8')
                  for idx, ship in enumerate(regionprops(segments)):
                      x, y = (int(np.average([ship.bbox[0], ship.bbox[2]])),
                              int(np.average([ship.bbox[1], ship.bbox[3]])))
                      new_xs.append((col * 256) + x)
                      new_ys.append((row * 256) + y)
        latlon_grid_list = rasterio.transform.xy(transform, new_ys, new_xs)
        return latlon_grid_list
