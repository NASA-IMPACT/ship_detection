import json
import requests
import pyproj
import mercantile
import numpy as np

SEARCH_URL = 'https://api.planet.com/data/v1/quick-search'
BODY = {
    "filter": {
        "type": "AndFilter",
        "config": [ {
          "type": "GeometryFilter",
          "field_name": "geometry",
          "config": {
                "type": "Polygon",
                "coordinates": []
             }
        },
        {
          "type": "DateRangeFilter",
          "field_name":"acquired",
            "config":{
              "gt": "",
              "lte": ""
           }
        }]
    },
    "item_types": ["PSScene3Band"]
}

ZOOM_LEVEL = 14 # pixel resolution == 0.0032 m

class PlanetDownloader:

    def __init__(self, api_key):
        self.api_key = api_key


    def search_ids(self, extent, start_date_time, end_date_time):
        body = BODY
        body['filter']['config'][0]['config']['coordinates'] = self.prepare_coordinates(
          extent
        )
        body['filter']['config'][1]['config']['gt'] = start_date_time
        body['filter']['config'][1]['config']['lte'] = end_date_time
        response = requests.post(SEARCH_URL, auth=(self.api_key, ''), json=body)
        # if not 200 raise error
        response.raise_for_status()
        parsed_content = json.loads(response.text)

        return self.extract_data(parsed_content['features'])


    def extract_data(self, parsed_data):
        extracted_data = []
        for feature in parsed_data:
          current_data = { 'images': [] }
          current_data['id'] = feature['id']
          reverted_coordinates = self.revert_coordinates(
            feature['geometry']['coordinates']
          )
          current_data['coordinates'] = reverted_coordinates
          current_data['tiles'] = self.tile_indices(reverted_coordinates)
          extracted_data.append(current_data)
        return extracted_data


    def revert_coordinates(self, coordinates):
        coordinates = np.asarray(coordinates)
        lats = coordinates[:,:,1]
        lons = coordinates[:,:,0]
        return [lons.min(), lats.min(), lons.max(), lats.max()]


    def tile_indices(self, coordinates):
        start_x, start_y, _ = mercantile.tile(coordinates[0], coordinates[3], ZOOM_LEVEL)
        end_x, end_y, _ = mercantile.tile(coordinates[2], coordinates[1], ZOOM_LEVEL)
        return [[start_x, end_x], [start_y, end_y]]


    def prepare_coordinates(self, extent):
        return [
            [
                [extent[0], extent[1]],
                [extent[2], extent[1]],
                [extent[2], extent[3]],
                [extent[0], extent[3]],
                [extent[0], extent[1]]
            ]
        ]
