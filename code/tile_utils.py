import rasterio
import numpy as np

from skimage.measure import regionprops

THRESHOLD = 0.5


def xy_to_latlon(grid_list, rows, cols, bounds, width, height):

    transform = rasterio.transform.from_bounds(*bounds, width, height)
    for row in range(rows):
        for col in range(cols):
            segments = (grid_list[row][col] > THRESHOLD).astype('uint8')
            for idx, ship in enumerate(regionprops(segments)):
                x, y = (int(np.average([ship.bbox[0], ship.bbox[2]])),
                        int(np.average([ship.bbox[1], ship.bbox[3]])))
                new_xs.append((row * 256) + x)
                new_xs.append((col * 256) + y)
    latlon_grid_list = rasterio.transform.rowcol(transform, new_xs, new_ys)
    return latlon_grid_list
