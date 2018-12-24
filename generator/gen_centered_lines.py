import time as t
import numpy as np
import random as rand
import libs.generators as gens
from libs.concepts import Concept


def line_drawer(draw, dim, params, pen):
    assert len(params) == 2
    x, y = params
    draw.line((x, y, dim - 1 - x, dim - 1 - y), pen)


def generate_image(dim=27, antialias=False, show=False):
    angle = rand.randint(0, 359)
    x = rand.randint(0, dim // 2)
    y = rand.randint(0, dim - 1)
    image = gens.draw_image(dim, [x, y], drawer=line_drawer, rotate=angle, antialias=antialias, show=show)

    if show:
        print(image)

    return image


def main(concept):
    dim = 27
    center = dim // 2

    images = []

    for x in range(center + 1):
        for y in range(dim):
            if x == center and y == center:
                break

            params = [x, y]
            image = gens.draw_image(dim, params, drawer=line_drawer, show=False)
            images.append(image)

    images = np.array(images)
    print('shape:', images.shape)

    timestamp = int(t.time())
    filename = 'data/{}/{}-centered-{}-{}x{}x{}.npy'
    filename = filename.format(concept.code, concept.code, timestamp, len(images), dim, dim)
    np.save(filename, images)


if __name__ == '__main__':
    main(Concept.LINE)
    print('end')
