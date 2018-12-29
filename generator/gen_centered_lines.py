import time as t
import numpy as np
import random as rand
import libs.generators as gens
from libs.concepts import Concept


def line_drawer(draw, dim, params, pen):
    assert len(params) == 2
    x, y = params
    draw.line((x, y, dim - 1 - x, dim - 1 - y), pen)


def generate_thinning_pair(dim=27, show=False):
    x = rand.randint(0, dim // 2)
    y = rand.randint(0, dim - 1)
    outline = rand.randint(1, 4)

    print('generate_image: x={}, y={}, outline={}'.format(x, y, outline))
    image_clear = gens.draw_image(dim, [x, y], line_drawer, antialias=False, show=show)
    image_aa = gens.draw_image(dim, [x, y], line_drawer, antialias=True, outline=outline, show=show)

    return image_clear, image_aa


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
