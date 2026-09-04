"""Tests for image preprocessing code"""

import numpy as np
import src.data_preprocessing.image_preprocessing as img_pre
from src.util.logger import Logger

def create_garbage_image():
    """Create a valid RGB image with random height and width."""

    random_generator = np.random.default_rng()

    # random height/width between 1 and 1000
    height = random_generator.integers(1, 1001)
    width = random_generator.integers(1, 1001)

    # make an rgb image that is heightxwidth in size
    image = random_generator.integers(
        0, 256,
        size=(height, width, 3),
        dtype=np.uint8,
    )

    return image

def test_normalize_image():
    """ Verify the normalize_image function works """
    img= create_garbage_image()
    normalized_image= img_pre.normalize_image(img)

    # make sure all values are 0<=val<=1
    values_above_1= normalized_image > 1
    has_above_1= values_above_1.any()
    assert has_above_1 == False

    values_below_0= normalized_image < 0
    has_below_0= values_below_0.any()
    assert has_below_0 == False

def test_resize_image():
    """ Verify the resize_image function works """
    img= create_garbage_image()
    result= img_pre.resize_image(img, (1,1))
    height, width= result.shape[:2]
    assert height == 1
    assert width == 1
    assert result.size == 3 # height*width*3
    
    img= create_garbage_image()
    result= img_pre.resize_image(img, (1000,1000))
    height, width= result.shape[:2]
    assert height == 1000
    assert width == 1000
    assert result.size == 3000000 # height*width*3

    img= create_garbage_image()
    result= img_pre.resize_image(img, (100,500))
    height, width= result.shape[:2]
    assert height == 500
    assert width == 100
    assert result.size == 150000 # height*width*3

    img= create_garbage_image()
    result= img_pre.resize_image(img, (-30,500))
    assert result == None #invalid shape given

    img= create_garbage_image()
    result= img_pre.resize_image(img, (100,0))
    assert result == None # invalid shape given

def test_greyscale():
    """ Verify the greyscale function works """
    img= create_garbage_image()

    height, width= img.shape[:2]

    grey= img_pre.convert_to_greyscale(img)

    assert grey.size == height*width*1 #greyscale sizing
    assert grey.size != height*width*3 #rgb sizing



    
