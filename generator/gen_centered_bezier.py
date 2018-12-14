import time as t
import numpy as np
import aggdraw as agg
import random as rand
import libs.generators as gens
from PIL import Image
from libs.concepts import Concept


def show_image(image):
    image = Image.fromarray(image)
    image.show()
    return


def _get_path_string(base_width, control_point, mirror):
    dim = 27
    assert 2 < base_width < dim, "base_width too small or too large"
    assert len(control_point) == 2, "control_point should have two coordinates"
    assert (2 < control_point[0] < dim and
            2 < control_point[1] < dim), "control points are too small or too large"

    c1 = [0, 0] if not mirror else [dim - 1, 0]
    c2 = control_point if not mirror else [dim - 1 - control_point[0], control_point[1]]
    c3 = [base_width, 0] if not mirror else [dim - 1 - base_width, 0]
    path_string = "M{},{} Q{},{},{},{}".format(*[*c1, *c2, *c3])

    return path_string


def get_path_string(debug=False):
    dim = 27
    path_string = None

    while not path_string:
        coords = np.random.randint(0, dim, size=3, dtype=np.uint8)
        base_width = coords[0]
        control_point = [coords[1], coords[2]]
        mirror = bool(rand.getrandbits(1))

        try:
            path_string = _get_path_string(base_width, control_point, mirror)
        except AssertionError as error:
            if debug:
                print('not a valid path for:', coords, mirror, "error:", error)

    return path_string


def generate_big_image(rotation):
    dim = 27
    path_string = get_path_string()
    print('\tpath:', path_string, 'with rotation:', rotation)

    image = Image.new("L", (dim, dim))  # last part is image dimensions
    draw = agg.Draw(image)
    outline = agg.Pen("white", 1)  # 5 is the outline width in pixels

    symbol = agg.Symbol(path_string)
    xy = (0, 0)  # xy position to place symbol
    draw.symbol(xy, symbol, outline)
    draw.flush()

    if rotation:
        rotated = image.rotate(rotation, resample=Image.CUBIC, expand=True)
        image = Image.new('L', (3 * dim, 3 * dim), 'black')
        image.paste(rotated, (dim, dim), rotated)

    return np.asarray(image)


def generate_image(dim, show=False):
    rotation = rand.randint(0, 359)
    image = generate_big_image(rotation)

    center = gens.calc_image_center(image)
    x = center[1]
    y = center[0]
    half_dim = dim // 2
    centered_image = image[y-half_dim: y + half_dim + 1, x - half_dim: x + half_dim + 1]

    if np.sum(image) != np.sum(centered_image):
        print('\trotated image did not fit within 27x27')
        return None

    final_center = gens.calc_image_center(centered_image)
    if not np.array_equal(final_center, [half_dim, half_dim]):
        print('\tbig center:', center)
        print('\tfinal center:', final_center)
        return None

    if centered_image.shape != (dim, dim):
        print('\tfinal shape:', centered_image.shape)
        return None

    if show:
        show_image(image)
        show_image(centered_image)

    return centered_image


def main(concept):
    dim = 27
    images = []

    for i in range(10000):
        if i % 100 == 0:
            print('starting sample', i)

        image = generate_image(dim, show=False)
        images.append(image)

    images = np.array(images)
    print('shape:', images.shape)

    timestamp = int(t.time())
    filename = 'data/{}/{}_centered_{}_{}x{}x{}.npy'
    filename = filename.format(concept.code, concept.code, timestamp, len(images), dim, dim)
    np.save(filename, images)


if __name__ == '__main__':
    main(Concept.BEZIER)
    print('end')
