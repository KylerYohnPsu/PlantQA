"""
Testing file that verifies the environment and hardware are setup for the project
"""

import sys
import numpy
import pandas
import tensorflow as tf
import pytest
import matplotlib as mpl
import PIL as pillow

def test_python_version():
    """ Verify the python version is supported """
    assert sys.version_info[:2] == (3,10)

def test_packages():
    """ Verify vital packages are installed """
    assert numpy.__version__
    assert pandas.__version__
    assert tf.__version__
    assert mpl.__version__
    assert pillow.__version__

def test_tensorflow():
    """ Make sure tensorflow works """
    tensorflow_result= tf.reduce_sum(tf.constant([1,1,1]))
    assert tensorflow_result.numpy() == 3

@pytest.mark.hardware
def test_tensorflow_cpu():
    """ Make sure tensorflow can reach the CPU """
    cpu= tf.config.list_physical_devices("CPU")
    assert cpu, "Tensorflow can't find any CPUs"

@pytest.mark.hardware
def test_tensorflow_gpu():
    """ Make sure there is a tensorflow supported GPU """
    gpu= tf.config.list_physical_devices("GPU")
    if not gpu:
        pytest.skip("\n\nNo tensorflow supported GPU found.\nCPU execution will be used instead.\n\n")
    assert gpu

