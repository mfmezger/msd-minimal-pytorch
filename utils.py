import numpy as np

from scipy.ndimage import zoom


# use this if you want to normalize or denormalize the images.
def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)




def resize_volume(vol, target_shape=[120, 512, 512]):
    if vol.shape == target_shape:
        return vol
    else:
        new_array = zoom(vol, (target_shape[0] / vol.shape[0], target_shape[1] / vol.shape[1], target_shape[2] / vol.shape[2]))
    return new_array