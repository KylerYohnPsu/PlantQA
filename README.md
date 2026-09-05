# PlantQA
This is a VQA agent specialized in identifying plant life, diseases on plant, and answering questions regarding to plants. It uses both computer vision and RAG methods to build the VQA agent 


## Installation

To install the system, execute the setup script:

`./setup.sh`

This script does the following:
1. Creates a virtual environment named 'venv'
2. Installs required python libraries into venv
3. Installs sub-requirements (like the nltk stop words package)
4. Runs pytest to verify installation

The script will exit if any of the steps fail.

## Running

After the project has been installed, you can execute code through the `Capstone_Executables.ipynb` notebook.
Ensure the kernel the notebook is using is the created `venv` environment.

## Testing

Pytest is used to verify the installation/setup of the project.  To run all pytest tests run the following command from the project root directory:
`python3 -m pytest`

To run just the setup test:
`python3 -m pytest test/test_setup.py`

Verify all tests pass.  If any tests from the test_setup.py script fail, there is a chance your environment/hardware is not properly configured to support PlantVQA.