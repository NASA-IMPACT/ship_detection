''' Script to read, tile and batch predict planet scenes'''
import os

from planet_predict import (
    create_tiles_wgs84,
    predict_batch,
)

from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from glob import glob
from models import (
    make_model_rcnn,
    predict_rcnn
)

def planet_preprocess(
    planet_path: str,
    output_path: str,
    num_threads: int=4) -> None:
    """
    Preprocess the planet images.
    :param planet_path: the path of the planet images
    :param output_path: the path of the output folder
    :param num_threads: number of threads to use
    planet path is the extracted planet download path
    :return: None
    """

    image_paths = get_images_paths(planet_path, format_type='tif')
    pool = Pool(cpu_count())

    for _ in tqdm(pool.imap_unordered(
        create_tiles_wgs84,
        [(image_path, output_path) for image_path in image_paths])):
        # this is to make tqdm work with pool mapping to track progress
        pass
    pool.join()
    pool.close()

def get_images_paths(folder_path: str, format_type: str) -> list:
    """
    Get the images' paths.
    :param folder_path: the path of the folder
    :param format_type: the format type of the images
    :return: the images' paths
    """
    # Get the images' paths
    image_paths = []
    print(f'looking for {format_type} files in {folder_path}')
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(format_type):
                image_paths.append(os.path.join(root, file))

    return image_paths

def planet_predict(
    processed_images_path: str,
    output_path: str=None,
    num_threads: int=4) -> None:
    """
    Predict the planet images.
    :param processed_images_path: the path of the processed images (GeoTIFF)
    :param output_path: the path of the output folder (GeoJSON)
    :param num_threads: number of threads to use
    :return: None
    """
    scene_paths = glob(processed_images_path + '/*')

    for scene in tqdm(scene_paths):
        rcnn_model = make_model_rcnn()
        predict_batch(scene, rcnn_model)

if __name__ == '__main__':
    planet_preprocess(
        '/home/ubuntu/ship_detection/data/files/PSScene3Band',
        output_path='/home/ubuntu/ship_detection/data/la/temp'
    )
    planet_predict('/home/ubuntu/ship_detection/data/la/temp')