import glob
import geopandas as gpd
import itertools
import json
import shutil

from json import load, JSONEncoder
from optparse import OptionParser
from re import compile
from uploader import Uploader
from shapely.geometry import mapping, shape, Polygon


float_pat = compile(r'^-?\d+\.\d+(e-?\d+)?$')
charfloat_pat = compile(r'^[\[,\,]-?\d+\.\d+(e-?\d+)?$')
precision = 6

#env_vars = load_dotenv()

def merge_geojsons(infiles, outfile, remove_intersections=True):
    """
    Merge a list of GeoJSON files into a single GeoJSON file.
    :param infiles: list of GeoJSON files to merge
    :param outfile: output GeoJSON file
    :param remove_intersections: remove overlapping polygons
    """

    outjson = dict(type='FeatureCollection', features=[])

    if remove_intersections:
        polygons = []
        for infile in infiles:
            injson = load(open(infile))
            for feature in injson['features']:
                polygons.append(
                    Polygon(
                        feature['geometry']['coordinates'][0]
                    )
                )
        polygon_file_pair = zip(infiles, polygons)
        for pair1, pair2 in  itertools.combinations(polygon_file_pair, 2):
            file1, geom1 = pair1
            file2, geom2 = pair2
            if geom1.intersects(geom2):
                if file2 in infiles:
                    infiles.remove(file2)

    for infile in infiles:
        injson = load(open(infile))

        if injson.get('type', None) != 'FeatureCollection':
            raise Exception('Sorry, "%s" does not look like GeoJSON' % infile)

        if type(injson.get('features', None)) != list:
            raise Exception('Sorry, "%s" does not look like GeoJSON' % infile)

        outjson['features'] += injson['features']

    encoder = JSONEncoder(separators=(',', ':'))
    encoded = encoder.iterencode(outjson)

    format = '%.' + str(precision) + 'f'
    output = open(outfile, 'w')

    for token in encoded:
        if charfloat_pat.match(token):
            # in python 2.7, we see a character followed by a float literal
            output.write(token[0] + format % float(token[1:]))

        elif float_pat.match(token):
            # in python 2.6, we see a simple float literal
            output.write(format % float(token))

        else:
            output.write(token)


def upload_to_il():

    uploader = Uploader(env_vars['IL_USERNAME'], env_vars['IL_PASSWORD'])
    for geojson in glob.glob('combined_geojsons/la/*.geojson'):
        shapefile_name = geojson.replace('.geojson', '')
        gdf = gpd.read_file(geojson)
        if not gdf.empty:
            gdf.to_file(shapefile_name, driver='ESRI Shapefile')
            shutil.make_archive(shapefile_name, 'zip', root_dir=shapefile_name)
            shutil.rmtree(shapefile_name)
            uploader.upload_shapefile(shapefile_name+".zip")


def consolidate_geojsons_daywise(json_loc: str, output_path: str) -> None:
    """
    combine all the geojsons for a given day (present in the scene_id) into one file.
    :str json_loc: location of the geojsons
    :str output_path: path to the output file
    :return: None
    """
    json_locations = glob.glob(json_loc + '/*')
    timestamps = []
    for json_loc in json_locations:
        timestamps.append(json_loc.split('/')[-1].split('_')[0])
    unique_timestamps = list(set(timestamps))

    for unique_timestamp in unique_timestamps:
        geojson_filenames = []
        outfile = output_path + '/' + unique_timestamp + '.geojson'
        for json_loc in json_locations:
            if unique_timestamp in json_loc:
                geojson_filenames += glob.glob(json_loc + '/*.geojson')
        merge_geojsons(geojson_filenames, outfile)



if __name__ == '__main__':
    geojsons_location = 'shapes'
    combined_output_location = 'combined_geojsons'
    consolidate_geojsons_daywise(geojsons_location, combined_output_location)
