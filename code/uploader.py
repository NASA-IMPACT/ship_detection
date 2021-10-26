import base64
import json
import os
import rasterio
import requests
import subprocess

from glob import glob
from rasterio.io import MemoryFile
from rasterio.warp import reproject, calculate_default_transform, Resampling
from zipfile import ZipFile

BASE_URL = "https://api.labeler.nasa-impact.net"

CLIENT_ID = "TlrtevKVVQtSNBtrFrZtQiOSvVdjeUQzo2IsnYNv"
CLIENT_SECRET = "uKvfcTWRnS74L0DlawGIilP1Lgv8vloHopFR6jnxcWEPSg2SFYnEES6dRBw4ldP2taCoMeVsA15sl8bPbxoERSITC4CJrA84OrpWaaQHiYKmTLIQQa1B4TZqaIWPqW7m"

DEFAULT_CRS = 'EPSG:4326'

LOGIN_URL = f"{BASE_URL}/accounts/login/"

OGR_OGR = ['ogr2ogr', '-f', 'ESRI Shapefile']

SHAPEFILE_URL = f"{BASE_URL}/api/shapefiles"

IL_URL = {
    'shapefile': f"{BASE_URL}/api/shapefiles",
    'geotiff': f"{BASE_URL}/api/geotiffs"
}


class Uploader:

    def __init__(self, username, password):
        """
        Initializer
        Args:
            username (str): ImageLabeler Username
            password (str): ImageLabeler Password
        """
        self.request_token(username, password)
        Uploader.mkdir('updated')


    def upload_shapefile(self, shp_filename):
        """
        Upload shapes to imagelabeler
        Args:
            shp_filename (str): Filename of the shapefile to be uploaded
        """
        return self.upload_to_image_labeler(shp_filename, 'shapefile')


    def tag_shapefiles(
        self,
        event_id,
        shapefile_date,
        shapefile_id,
        bounding_box,
        tag_status='untagged'
    ):
        """
        Tag Shapefile to an event
        Args:
            event_id (int): Event identifier
            shapefile_date (str): date of the event in yyyy-mm-dd format
            shapefile_id (int): shapefile identifier
            bounding_box (str): string of bounding box in "lower, left, upper, right" format
            tag_status (str): string one of "yes", "no", "untagged"
        """

        data = {
            'bounding_box': bounding_box,
            'date_range': "",
            'event': event_id,
            'gibs_submit': "Extract",
            'layer': "MODIS_Terra_CorrectedReflectance_TrueColor",
            'product': "TrueColor",
            'query_date': shapefile_date,
            'resolution_in_km': 0.5,
            'sensor': "Terra",
            'shapefile': shapefile_id,
            'tag_status': 'no'
        }

        headers = {
            **self.headers,
            'Content-Type': 'application/json'
        }
        return requests.post(
            f"{BASE_URL}/api/map-tiles",
            headers=headers,
            json=data
        )


    def upload_geotiff(self, filename):
        """
        Reproject and upload geotiff into imagelabeler
        Args:
            filename (str): file name of the file to upload
        """
        tiff_file = rasterio.open(filename)
        updated_profile = self.calculate_updated_profile(tiff_file)
        with rasterio.open(filename, 'w', **updated_profile) as dst:
            for band in range(1, 4):
                reproject(
                    source=rasterio.band(tiff_file, band),
                    destination=rasterio.band(dst, band),
                    src_transform=tiff_file.transform,
                    src_crs=tiff_file.crs,
                    dst_transform=updated_profile['transform'],
                    dst_crs=DEFAULT_CRS,
                    resampling=Resampling.nearest)
        return self.upload_to_image_labeler(filename)


    def calculate_updated_profile(self, tiff_file):
        """
        Create updated profile for the provided tiff_file
        Args:
            tiff_file (rasterio.io.MemoryFile): rasterio memoryfile.
        Returns:
            dict: updated profile for new tiff file
        """
        profile = tiff_file.profile
        transform, width, height = calculate_default_transform(
            tiff_file.crs,
            DEFAULT_CRS,
            tiff_file.width,
            tiff_file.height,
            *tiff_file.bounds
        )
        profile.update(
            crs=DEFAULT_CRS,
            transform=transform,
            width=width,
            height=height,
            count=3,
            nodata=0,
            compress='lzw',
            dtype='uint8'
        )
        return profile


    def request_token(self, username, password):
        """
        this funtion will return an authentication token for users to use
        Args:
            username (string) : registered username of the user using the script
            password (string) : password associated with the user
        Exceptions:
            UserNotFound: Given user does not exist
        Returns:
            headers (dict): {
                "Authorization": "Bearer ..."
            }
        """

        payload = {
            "username": username,
            "password": password,
            "grant_type": "password"
        }

        response = requests.post(
            f"{BASE_URL}/authentication/token/",
            data=payload,
            auth=(CLIENT_ID, CLIENT_SECRET)
        )
        access_token = json.loads(response.text)['access_token']
        self.headers = {
            'Authorization': f"Bearer {access_token}",
        }


    def upload_to_image_labeler(self, file_name, file_type='geotiff'):
        """
        Uploads a single shapefile to the image labeler
        Args:
            file_name : name of zip file containing shapefiles
        Returns:
            response (tuple[string]): response text, response code
        """
        with open(file_name, 'rb') as upload_file_name:
            file_headers = {
                **self.headers,
            }
            files = {
                'file': (file_name, upload_file_name),
            }
            response = requests.post(
                IL_URL[file_type],
                files=files,
                headers=file_headers
            )
            print(f"{file_name} uploaded to imagelabeler with: {response.status_code}")
            return response

    @classmethod
    def mkdir(cls, dirname):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            print(f'directory created: {dirname}')