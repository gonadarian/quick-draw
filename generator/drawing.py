# import time as t
# import cv2 as cv
import numpy as np
import aggdraw as agg
from PIL import Image


def draw_image_agg(dim, params, drawer, rotate=None, show=False):
    image = Image.new("L", (dim, dim), "black")

    draw = agg.Draw(image)
    pen = agg.Pen("white", width=1)
    drawer(draw, dim, params, pen)
    draw.flush()

    if rotate:
        rotated = image.rotate(rotate, resample=Image.BICUBIC, expand=False)
        image = Image.new('L', (dim, dim), 'black')
        image.paste(rotated, (0, 0), rotated)

    image_array = np.asarray(image)

    if show:
        print(image_array)
        image.show()

    return image_array


def ellipse_drawer(draw, dim, params, pen):
    assert len(params) == 2
    x, y = params
    # draw.ellipse((100, 200, 400, 300), pen)
    # draw.ellipse((x, y, dim - 1 - x, dim - 1 - y), outline=255)
    draw.ellipse((x, y, dim - 1 - x, dim - 1 - y), pen)
    # draw.ellipse((0, 0, dim, dim), pen)


def test():
    test_case = 1

    if test_case == 1:
        image = Image.new("L", (500, 500), "black")
        draw = agg.Draw(image)
        pen = agg.Pen("white", 0.5)
        draw.ellipse((100, 200, 400, 300), pen)
        draw.flush()
        image.show()

    if test_case == 2:
        image = Image.new("RGB", (500, 500), "white")
        draw = agg.Draw(image)
        pen = agg.Pen("black", 0.5)

        draw.line((0, 500, 500, 0), pen)
        draw.ellipse((100, 200, 400, 300), pen)
        path = agg.Path()
        path.moveto(0, 0)
        path.curveto(0, 60, 40, 100, 300, 200)
        draw.path(path, pen)

        draw.flush()
        image.show()


def calc_center_debug(x):
    print(x.shape, x)
    indices = np.argwhere(x > 0)
    m = len(indices)
    print(m)
    print('indices', indices.shape)
    pixels = x[indices[:, 0], indices[:, 1]].reshape(m, 1)
    print('pixels', pixels.shape, pixels)
    weighted = np.multiply(pixels, indices)
    print('weighted', weighted.shape, weighted)
    pixels_sum = np.sum(pixels)
    print('pixels_sum', pixels_sum)
    weighted = np.sum(weighted, axis=0)
    print('weighted', weighted.shape, weighted)
    weighted = weighted / pixels_sum
    print('weighted', weighted.shape, weighted)
    weighted = np.rint(weighted).astype(np.int64)
    print('weighted', weighted)

    return weighted


def calc_center(x):
    indices = np.argwhere(x > 0)
    m = len(indices)

    pixels = x[indices[:, 0], indices[:, 1]].reshape(m, 1)
    pixels_sum = np.sum(pixels)

    weighted = np.multiply(pixels, indices)
    weighted = np.sum(weighted, axis=0)
    weighted = weighted / pixels_sum
    weighted = np.rint(weighted).astype(np.int64)

    return weighted


def draw_agg_symbol(randomize=True, debug=False):
    if randomize:
        # coords = np.array([1, 2, 2, 4, 8, 16])  # collinear
        # coords = np.array([1, 2, 2, 2, 8, 16])  # too near
        coords = np.random.randint(0, 27, size=6, dtype=np.uint8)
        c1, c2, c3 = np.split(coords, 3)
        # c2 = coords[0:2]
        # c2 = coords[2:4]
        # c3 = coords[4:6]
        too_near = (np.linalg.norm(c1 - c2) < 2 or
                    np.linalg.norm(c2 - c3) < 2 or
                    np.linalg.norm(c3 - c1) < 2)
        if too_near:
            print('coords {} are too near'.format(coords))
            return False
        collinear = np.cross(c1 - c2, c1 - c3)
        collinear = collinear == 0
        if collinear:
            print('coords {} are collinear'.format(coords))
            return False

        path_string = "m{},{} q{},{},{},{}".format(*coords)
    else:
        # path_string = "m0,0 c24,2,2,24,26,26"
        # path_string = "m0,0 q26,13,26,26"
        path_string = "m15,15 q19,8,21,25"

    print(path_string)

    image = Image.new("L", (27, 27))  # last part is image dimensions
    draw = agg.Draw(image)
    outline = agg.Pen("white", 1)  # 5 is the outline width in pixels

    symbol = agg.Symbol(path_string)
    xy = (0, 0)  # xy position to place symbol
    draw.symbol(xy, symbol, outline)
    draw.flush()

    image.show()

    image_array = np.asarray(image)
    x, y = calc_center_debug(image_array) if debug else calc_center(image_array)
    print(x, y)

    # print(image_array.shape, image_array)
    # indices = np.argwhere(image_array > 0)
    # print(indices.shape, indices)
    # pixels = image_array[indices[:, 0], indices[:, 1]].reshape(88, 1)
    # print(pixels.shape, pixels)
    # weighted = np.multiply(pixels, indices)
    # print(weighted.shape, weighted)
    # weighted = weighted / 88
    # weighted = np.average(weighted, axis=0)
    # print(weighted.shape, weighted)


# for i in range(100):
#     draw_agg_symbol()

draw_agg_symbol(randomize=True, debug=True)

# draw_image_agg(27, [2, 10], drawer=ellipse_drawer, rotate=30, show=True)

print('end')
