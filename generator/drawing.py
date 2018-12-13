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
    weighted = np.rint(np.average(np.multiply(pixels, indices) / 255, axis=0)).astype(np.int64)

    return weighted


def test_agg_symbols(randomize=True, debug=False):
    image = Image.new("L", (27, 27))  # last part is image dimensions
    draw = agg.Draw(image)
    outline = agg.Pen("white", 1)  # 5 is the outline width in pixels

    if randomize:
        coords = np.random.randint(0, 27, size=6, dtype=np.uint8).tolist()
        path_string = "m{},{} q{},{},{},{}".format(*coords)
    else:
        # path_string = "m0,0 c24,2,2,24,26,26"
        # path_string = "m0,0 q26,13,26,26"
        path_string = "m15,15 q19,8,21,25"

    print(path_string)

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


# def test_opencv():
#     # Create a black image
#     img = np.zeros((27, 27, 1), np.uint8)
#     # Draw a diagonal blue line with thickness of 5 px
#     img = cv.line(img, (1, 1), (25, 25), (255), 1)
#     cv.imshow('image', img)
#     cv.waitKey(0)
#     timestamp = int(t.time())
#     cv.imwrite('opencv_test_{}.png'.format(timestamp), img)
#     cv.destroyAllWindows()


# for i in range(100):
#     test_agg_symbols()

test_agg_symbols(randomize=True, debug=True)

# test_opencv()
# draw_image_agg(27, [2, 10], drawer=ellipse_drawer, rotate=30, show=True)

print('end')
