import time as t
import numpy as np
import random as rand
import libs.generators as gens
from libs.concepts import Concept


def ellipse_drawer(draw, dim, params, pen):
    assert len(params) == 2
    x, y = params
    draw.ellipse((x, y, dim - 1 - x, dim - 1 - y), pen)


def generate_thinning_pair(dim=27, show=False):
    x = rand.randint(0, dim // 2 - 1)
    y = rand.randint(0, dim // 2 - 1)
    outline = rand.randint(1, 4)

    print('generate_thinning_pair: x={}, y={}, outline={}'.format(x, y, outline))
    image_clear = gens.draw_image(dim, [x, y], ellipse_drawer, antialias=False, show=show)
    image_aa = gens.draw_image(dim, [x, y], ellipse_drawer, antialias=True, outline=outline, show=show)

    return image_clear, image_aa


def main(concept):
    dim = 27
    quadrant_width = 13

    images = []

    for x in range(quadrant_width):
        for y in range(quadrant_width):

            # introduce some randomness to angles,
            # so we cover as many different angles as possible
            shift = rand.randint(0, 29)

            for r in range(12):
                params = [x, y]
                image = gens.draw_image(dim, params, drawer=ellipse_drawer, rotate=shift + r * 30)
                assert np.sum(image) > 0

                images.append(image)

    images = np.array(images)
    print('shape:', images.shape)

    timestamp = int(t.time())
    filename = 'data/{}/{}-centered-{}-{}x{}x{}.npy'
    filename = filename.format(concept.code, concept.code, timestamp, len(images), dim, dim)
    np.save(filename, images)


if __name__ == '__main__':
    main(Concept.ELLIPSE)
    print('end')
