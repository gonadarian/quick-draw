import time as t
import numpy as np
import random as rand
import libs.generators as gens
from libs.concepts import Concept


def ellipse_drawer(draw, dim, params):
    assert len(params) == 2
    x, y = params
    draw.ellipse((x, y, dim - 1 - x, dim - 1 - y), outline=255)


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
