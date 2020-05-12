from osgeo import gdal, osr
import numpy
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
import skimage.io
from IPython.display import Image
import json
sample_data_file_name = '../data/download.tif'
# Read image into scimage package
img = skimage.io.imread(sample_data_file_name)
skimage.io.imsave('../data/original.png', img)

# Display original image
Image(filename='../data/original.png')

# Prepare result structure
result = {
    "ship_count": 0,
    "ships": []
}
# Open image with gdal
ds = gdal.Open(sample_data_file_name)
xoff, a, b, yoff, d, e = ds.GetGeoTransform()

# Get projection information from source image
ds_proj = ds.GetProjectionRef()
ds_srs = osr.SpatialReference(ds_proj)

# Get the source image's geographic coordinate system (the 'GEOGCS' node of ds_srs)
geogcs = ds_srs.CloneGeogCS()

# Set up a transformation between projected coordinates (x, y) & geographic coordinates (lat, lon)
transform = osr.CoordinateTransformation(ds_srs, geogcs)
import ipdb; ipdb.set_trace()

# Convert multi-channel image it into red, green and blueb[, alpha] channels
red, green, blue, alpha = numpy.rollaxis(numpy.array(img), axis=-1)

# Mask: threshold + stops canny detecting image boundary edges
mask = red > 75

# Create mask for edge detection
skimage.io.imsave('../data/mask.png', mask * 255)

# Use Felzenszwalb algo to find segements
segments_fz = felzenszwalb(numpy.dstack((mask, mask, mask)),
                           scale=5000,
                           sigma=3.1,
                           min_size=25)

# Build labeled mask to show where ships were dectected
segmented_img = mark_boundaries(mask, segments_fz)
skimage.io.imsave('../data/mask_labeled.png', segmented_img)

# Count ships and save image of each boat clipped from masked image
for idx, ship in enumerate(regionprops(segments_fz)):

    # If area matches that of a stanard ship, count it
    if (ship.area >= 300 and ship.area <= 10000):

        # Incrment count
        result['ship_count'] += 1

        # Create ship thumbnail
        x, y = (int(numpy.average([ship.bbox[0],
                                   ship.bbox[2]])),
                int(numpy.average([ship.bbox[1],
                                   ship.bbox[3]])))
        sx, ex = max(x - 35, 0), min(x + 35, img.shape[0] - 1)
        sy, ey = max(y - 35, 0), min(y + 35, img.shape[1] - 1)
        img_ship = img[sx:ex, sy:ey]
        skimage.io.imsave('../data/ship-%s.png' % str(idx + 1), img_ship)

        # Get global coordinates from pixel x, y coords
        projected_x = a * y + b * x + xoff
        projected_y = d * y + e * x + yoff

        # Transform from projected x, y to geographic lat, lng
        (lat, lng, elev) = transform.TransformPoint(projected_x, projected_y)

        # Add ship to results cluster
        result["ships"].append({
            "id": idx + 1,
            "lat": lat,
            "lng": lng
        })

# Display results
print(json.dumps(result, indent=2))

# Display mask used for ship detection.
display(Image(filename='../data/mask.png'))

# Display labled mask where we detected ships
display(Image(filename='../data/mask_labeled.png'))

# Display each individual ship cropped out of the original image
for idx, ship in enumerate(result['ships']):
    print("Ship " + str(idx + 1))
    display(Image(filename='../data/ship-' + str(idx + 1) + '.png'))

