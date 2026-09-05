"""Tests for image preprocessing code"""

import numpy as np
import src.data_preprocessing.image_preprocessing as img_pre
from src.util.logger import Logger
import pytest

@pytest.fixture
def simple_image():
    """ make a simple rgb numpy image """
    # 2x3 rgb image data
    image_data= [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[255, 255, 255], [0, 0, 0], [128, 128, 128]],
        ]
    image= np.array(image_data, dtype=np.uint8)
    return image

def test_normalize_image(simple_image):
    """ Verify the normalize_image function works """
    normalized_image= img_pre.normalize_image(simple_image)

    # make sure all values are 0<=val<=1
    values_above_1= normalized_image > 1
    has_above_1= values_above_1.any()
    assert has_above_1 == False

    values_below_0= normalized_image < 0
    has_below_0= values_below_0.any()
    assert has_below_0 == False

@pytest.mark.parametrize(
    (  "new_size",     "expected_resized_shape"),
    [
    (     (1,1),               (1,1,3)          ),
    (  (1000,1000),        (1000, 1000, 3)      ),
    (   (100,500),            (500, 100, 3)     ),
    ],
)
def test_resize_image(simple_image, new_size, expected_resized_shape):
    """ Verify the resize_image function works """
    result= img_pre.resize_image(simple_image, new_size)

    assert result.shape == expected_resized_shape


@pytest.mark.parametrize(
    ("invalid_new_size"),
    [
    (-1, -1),
    (-1, 100),
    (100, -1),
    (0, 1)
    ],
)
def test_resize_image_invalid_size(simple_image, invalid_new_size):
    """ Verify resize_image properly rejects bad sizes """
    result= img_pre.resize_image(simple_image, invalid_new_size)
    assert result is None

def test_greyscale(simple_image):
    """ Verify the greyscale function works """
    height, width= simple_image.shape[:2]

    grey= img_pre.convert_to_greyscale(simple_image)

    assert grey.size == height*width*1 #greyscale sizing
    assert grey.size != height*width*3 #rgb sizing
    assert grey.shape == (2,3) # we knwo simple_image is a 2x3 rgb image, so greyscale, should be 2x3 with no rgb values

def test_load_image_invalid_path(tmp_path):
    """ verify load_image does not try to get invalid paths """
    bad_path= tmp_path / "bad_file.png"
    result= img_pre.load_image(bad_path)
    assert result is None

def test_save_and_load_image(tmp_path, simple_image):
    """ verify saving and loading the saved image works """
    image_path= tmp_path / "test_save_image.png"
    img_pre.save_image(simple_image, image_path)

    assert image_path.exists()

    loaded_image= img_pre.load_image(image_path)

    assert loaded_image is not None
    assert np.array_equal(loaded_image, simple_image) #make sure the loaded version matches the og version

    
