import math
import time as t
import numpy as np
import random as rand
import libs.generators as gens
from libs.concepts import Concept


def generate_star_angles(edges, debug=False):
    assert 2 <= edges <= 6
    angles = []

    for _ in range(edges):
        attempts = 0

        while True:
            found = True
            attempts += 1
            new_angle = rand.randint(0, 359)
            debug and print('attempts', attempts, 'new_angle', new_angle)

            for angle in angles:
                if abs(angle - new_angle) < 23:
                    debug and print('\ttoo close', angles[0], new_angle)
                    found = False
                if edges == 2 and len(angles) == 1 and abs(abs(angle - new_angle) - 180) < 23:
                    debug and print('\tlike a line', angles[0], new_angle)
                    found = False

            if found:
                break
            if attempts > 20:
                return

        angles.append(new_angle)

    return angles


def star_drawer(draw, dim, params, pen):
    assert len(params) == 1
    edges = params[0]
    print('edges', edges)

    angles = generate_star_angles(edges)
    print('angles', angles)
    if angles is None:
        return

    size = 5
    center = dim // 2

    for angle in angles:
        x = math.cos(math.radians(angle)) * size
        y = math.sin(math.radians(angle)) * size

        draw.line((center, center, center + x, center + y), pen)


def generate_thinning_pair(dim=27, show=False):
    edge_count = get_edge_count()
    params = [edge_count]
    outline = rand.randint(1, 4)

    print('generate_image: edge_count={}'.format(edge_count))
    image_clear = gens.draw_image(dim, params, star_drawer, antialias=False, show=show)
    image_aa = gens.draw_image(dim, params, star_drawer, antialias=True, outline=outline, show=show)

    return image_clear, image_aa


def get_edge_count():
    weights = {
        2: 20,
        3: 10,
        4: 20,
        5: 5,
        6: 10,
    }

    weights = list(weights.values())
    weights_sum = sum(weights)
    weights_limits = [sum(weights[:i + 1]) for i in range(len(weights))]
    choice = rand.randint(0, weights_sum)

    for edge_count, limit in enumerate(weights_limits):
        edge_count += 2
        if choice <= limit:
            return edge_count


def main(concept=None):
    m = 2000
    dim = 27
    images = []

    for _ in range(m):
        edge_count = get_edge_count()
        params = [edge_count]
        image = gens.draw_image(dim, params, drawer=star_drawer, show=False)
        images.append(image)

    images = np.array(images)
    print('shape:', images.shape)

    timestamp = int(t.time())
    filename = 'data/{}/{}-centered-{}-{}x{}x{}.npy'
    filename = filename.format(concept.code, concept.code, timestamp, len(images), dim, dim)
    np.save(filename, images)


if __name__ == '__main__':
    main(Concept.STAR)
    print('end')
