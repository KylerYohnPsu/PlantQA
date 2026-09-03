# PlantQA
This is a VQA agent specialized in identifying plant life, diseases on plant, and answering questions regarding to plants. It uses both computer vision and RAG methods to build the VQA agent 


## Installation

1. Create your virtual environment: `python3 -m venv venv`
2. Source your venv: `source venv/bin/activate`
3. Install pip libraries: `pip install -r requirements.txt`


## Testing

Pytest is used to verify the installation/setup of the project.  To run all pytest tests run the following command from the project root directory:
`python3 -m pytest`

To run just the setup test:
`python3 -m pytest test/test_setup.py`

Verify all tests pass.  If any tests from the test_setup.py script fail, there is a chance your environment/hardware is not properly configured to support PlantVQA.