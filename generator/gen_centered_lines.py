import time as t
import numpy as np
import libs.generators as gens
from libs.concepts import Concept


def line_drawer(draw, dim, params):
    assert len(params) == 2
    x, y = params
    draw.line((x, y, dim - 1 - x, dim - 1 - y), fill=255)


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
