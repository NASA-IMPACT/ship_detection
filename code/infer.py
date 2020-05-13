import mercantile
import requests
import numpy as np

from models import load_from_path
from io import BytesIO
from PIL import Image
from planet_downloader import PlanetDownloader

EXTENTS = {
  'san_fran': [-123.43, 37.71, -123.30, 37.85]
}

WEIGHT_FILE = '../weights/iou_model.hdf5'
ZOOM_LEVEL = 14
IMG_SIZE = 768
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
      print('running for location:', location)
      datasets = self.prepare_dataset(extent)
      for dataset in datasets:
        predictions = self.model.predict(dataset['images'])
        rows, columns = [elem[1] - elem[0] for elem in dataset['tiles']]
        predictions = predictions.reshape((rows, columns, IMG_SIZE, IMG_SIZE))
        # insert here


  def prepare_dataset(self, extent):
    items = self.planet_downloader.search_ids(
      extent, self.start_date_time, self.end_date_time
    )
    for item in items:
      x_indices, y_indices = item['tiles']
      print(x_indices[1] - x_indices[0])
      print(y_indices[1] - y_indices[0])
      for x_index in list(range(*x_indices)):
        for y_index in list(range(*y_indices)):
          response = requests.get(
            WMTS_URL.format(item['id'], x_index, y_index, self.credential)
          )
          response.raise_for_status()
          img = np.asarray(
            Image.open(BytesIO(response.content)).resize(
                (IMG_SIZE, IMG_SIZE)
              ).convert('RGB')
            )
          item['images'].append(img)
      item['images'] = np.asarray(item['images'])
    return items


  def prepare_geojson(self):
    pass
