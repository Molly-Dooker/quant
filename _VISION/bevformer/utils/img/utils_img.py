import os.path as osp
from pathlib import Path
from enum import Enum

import cv2
from PIL import Image, ImageOps
import numpy as np
import tifffile
from turbojpeg import RGB, BGR, GRAY

jpeg = None
imread_backend = 'cv2'
supported_backends = ['cv2', 'turbojpeg', 'pillow', 'tifffile']

imread_flags = {
    'color': cv2.IMREAD_COLOR,
    'grayscale': cv2.IMREAD_GRAYSCALE,
    'unchanged': cv2.IMREAD_UNCHANGED,
    'color_ignore_orientation': cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR,
    'grayscale_ignore_orientation':
    cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_GRAYSCALE
}

def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))

def _jpegflag(flag='color', channel_order='bgr'):
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'color':
        if channel_order == 'bgr':
            return BGR
        elif channel_order == 'rgb':
            return RGB
    elif flag == 'grayscale':
        return GRAY
    else:
        raise ValueError('flag must be "color" or "grayscale"')    

def _pillow2array(img, flag='color', channel_order='bgr'):
    """Convert a pillow image to numpy array.

    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.

    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'unchanged':
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # Handle exif orientation tag
        if flag in ['color', 'grayscale']:
            img = ImageOps.exif_transpose(img)
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != 'RGB':
            if img.mode != 'LA':
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert('RGB')
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert('RGBA')
                img = Image.new('RGB', img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag in ['color', 'color_ignore_orientation']:
            array = np.array(img)
            if channel_order != 'rgb':
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag in ['grayscale', 'grayscale_ignore_orientation']:
            img = img.convert('L')
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale", "unchanged", '
                f'"color_ignore_orientation" or "grayscale_ignore_orientation"'
                f' but got {flag}')
    return array

def imread(img_or_path, flag='color', channel_order='bgr', backend=None):
    """Read an image.

    Args:
        img_or_path (ndarray or str or Path): Either a numpy array or str or
            pathlib.Path. If it is a numpy array (loaded image), then
            it will be returned as is.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale`, `unchanged`,
            `color_ignore_orientation` and `grayscale_ignore_orientation`.
            By default, `cv2` and `pillow` backend would rotate the image
            according to its EXIF info unless called with `unchanged` or
            `*_ignore_orientation` flags. `turbojpeg` and `tifffile` backend
            always ignore image's EXIF info regardless of the flag.
            The `turbojpeg` backend only supports `color` and `grayscale`.
        channel_order (str): Order of channel, candidates are `bgr` and `rgb`.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`.
            If backend is None, the global imread_backend specified by
            ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        ndarray: Loaded image array.
    """

    if backend is None:
        backend = imread_backend
    if backend not in supported_backends:
        raise ValueError(f'backend: {backend} is not supported. Supported '
                         "backends are 'cv2', 'turbojpeg', 'pillow'")
    if isinstance(img_or_path, Path):
        img_or_path = str(img_or_path)

    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif is_str(img_or_path):
        check_file_exist(img_or_path,
                         f'img file does not exist: {img_or_path}')
        if backend == 'turbojpeg':
            with open(img_or_path, 'rb') as in_file:
                img = jpeg.decode(in_file.read(),
                                  _jpegflag(flag, channel_order))
                if img.shape[-1] == 1:
                    img = img[:, :, 0]
            return img
        elif backend == 'pillow':
            img = Image.open(img_or_path)
            img = _pillow2array(img, flag, channel_order)
            return img
        elif backend == 'tifffile':
            img = tifffile.imread(img_or_path)
            return img
        else:
            flag = imread_flags[flag] if is_str(flag) else flag
            img = cv2.imread(img_or_path, flag)
            if flag == cv2.IMREAD_COLOR and channel_order == 'rgb':
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img
    else:
        raise TypeError('"img" must be a numpy array or a str or '
                        'a pathlib.Path object')

class Color(Enum):
    """An enum that defines common colors.

    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)
    
def color_val(color):
    """Convert various input to color tuples.

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if is_str(color):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    elif isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError(f'Invalid type for color: {type(color)}')


def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)

def convert_color_factory(src, dst):

    code = getattr(cv2, f'COLOR_{src.upper()}2{dst.upper()}')

    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img

    convert_color.__doc__ = f"""Convert a {src.upper()} image to {dst.upper()}
        image.

    Args:
        img (ndarray or str): The input image.

    Returns:
        ndarray: The converted {dst.upper()} image.
    """

    return convert_color


bgr2rgb = convert_color_factory('bgr', 'rgb')
rgb2bgr = convert_color_factory('rgb', 'bgr')