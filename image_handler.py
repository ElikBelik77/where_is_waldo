import numpy
from PIL import Image


def produce_crops(source_path, size=(64, 64)):
    source_image = Image.open(source_path)
    source_image = source_image.convert('RGB')
    for i in range(0, source_image.size[0] // size[0]):
        for j in range(0, source_image.size[1] // size[1]):
            area = (i * size[0], j * size[1],(i + 1) * size[0] , (j + 1) * size[1])
            crop = source_image.crop(area)
            crop = crop.convert('RGB')
            yield i, j, crop
