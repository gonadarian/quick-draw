import time as t
import numpy as np
import aggdraw as agg
import random as rand
import libs.generators as gens
from PIL import Image
from libs.concepts import Concept


def rotate(coords, angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array(((c, -s), (s, c)))
    coords = np.matmul(coords, rotation)

    return coords


def _get_path_string(base_width, control_point, mirror, angle, dim=27):
    assert 2 < base_width < dim, "base_width too small or too large"
    assert len(control_point) == 2, "control_point should have two coordinates"
    assert (2 < control_point[0] < dim and
            2 < control_point[1] < dim), "control points are too small or too large"

    c1 = [0, 0] if not mirror else [dim - 1, 0]
    c2 = control_point if not mirror else [dim - 1 - control_point[0], control_point[1]]
    c3 = [base_width, 0] if not mirror else [dim - 1 - base_width, 0]

    coords = np.array([c1, c2, c3])
    coords = rotate(coords, angle)
    coords = coords - coords.mean(axis=0) + dim
    coords = coords.reshape((6, )).tolist()

    path_string = "M{},{} Q{},{},{},{}".format(*coords)

    return path_string


def get_path_string(debug=False):
    dim = 27
    path_string = None

    while not path_string:
        coords = np.random.randint(0, dim, size=3, dtype=np.uint8)
        base_width = coords[0]
        control_point = [coords[1], coords[2]]
        mirror = bool(rand.getrandbits(1))
        angle = rand.randint(0, 359)

        try:
            path_string = _get_path_string(base_width, control_point, mirror, angle)
        except AssertionError as error:
            if debug:
                print('not a valid path for:', coords, mirror, "error:", error)

    return path_string


def generate_big_image(path_string, antialias, outline):
    dim = 27

    image = Image.new("L", (dim * 2, dim * 2))  # last part is image dimensions
    draw = agg.Draw(image)
    if not antialias:
        draw.setantialias(False)

    outline = agg.Pen("white", outline)

    symbol = agg.Symbol(path_string)
    draw.symbol((0, 0), symbol, outline)
    draw.flush()

    return np.asarray(image)


def generate_image(path_string, dim=27, antialias=False, outline=1, show=False):
    half_dim = dim // 2

    image = generate_big_image(path_string, antialias, outline)
    y, x = gens.calc_image_center(image)
    centered_image = image[y - half_dim: y + half_dim + 1, x - half_dim: x + half_dim + 1]

    if np.sum(image) != np.sum(centered_image):
        print('\trotated image did not fit within 27x27')
        return None

    final_center = gens.calc_image_center(centered_image)
    if not np.array_equal(final_center, [half_dim, half_dim]):
        print('\tbig center:', x, ',', y)
        print('\tfinal center:', final_center)
        return None

    if centered_image.shape != (dim, dim):
        print('\tfinal shape:', centered_image.shape)
        return None

    if show:
        print('\tpath:', path_string)
        gens.show_image(image)
        gens.show_image(centered_image)

    return centered_image


def generate_thinning_pair(show=False):
    image_aa = None
    image_clear = None

    while image_aa is None or image_clear is None:
        path_string = get_path_string()
        outline = rand.randint(1, 4)

        print('generate_image: path_string={}, outline={}'.format(path_string, outline))
        image_clear = generate_image(path_string, antialias=False)
        image_aa = generate_image(path_string, antialias=True, outline=outline, show=show)

    return image_clear, image_aa


def main(concept):
    m = 4000
    dim = 27
    images = []

    for i in range(m):
        if i % 100 == 0:
            print('starting sample', i)

        image = None
        while image is None:
            path_string = get_path_string()
            image = generate_image(path_string, dim=dim, show=False)

        assert image.shape == (27, 27)
        images.append(image)

    images = np.array(images)
    print('shape:', images.shape)

    timestamp = int(t.time())
    filename = 'data/{}/{}-centered-{}-{}x{}x{}.npy'
    filename = filename.format(concept.code, concept.code, timestamp, len(images), dim, dim)
    np.save(filename, images)


if __name__ == '__main__':
    main(Concept.BEZIER)
    print('end')
