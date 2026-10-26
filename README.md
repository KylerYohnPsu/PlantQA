# PlantQA
This is a VQA agent specialized in identifying plant life, diseases on plant, and answering questions regarding plants. 
Our system utilizes both computer vision and RAG methods to build the VQA agent from a sequence of chained model outputs.
The following models are incorporated in our design:

| Model                                       | Description                                                                                                                                                                                                                                                                                                                                                                                | Input Size |
|---------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| UNet_64x128x256x512x1024_PlantSeg           | A segmentation model used to pre-process raw plant images for segmentation                                                                                                                                                                                                                                                                                                                 | 384 x 384  |
| EffNetB0_224x224_MultiHeadClass_VisualModel | A visual model with transfer learning based on EfficientNetV2-B0 used to predict key disease characteristics from a specified list of classification heads. Training consisted of 30 completely frozen epochs and 5 epochs with the first 100 layers unfrozen. An optional mask branch is included with convolutional, pooling, and dense layers.                                          | 224 x 224  |
| MiniLM_L6_384_Embedder                      | An embedding model, utilizing default rates, for embedding raw user utterances and chunked plant information                                                                                                                                                                                                                                                                               | N/A        |
| Qwen2.5_3B_QLoRA_ResponseModel              | A local LLM based on Qwen/Qwen2.5-3B-Instruct, with 4-bit quantization across select adaptive heads. This model synthesizes information retrieved from pre-filtered chunks stored in a Supabase DB, visual predictions, and the user's question/utterance. This model was trained on a single epoch over 5,000 de-duplicated PlantExpertVQA rows in the form of question and answer pairs. | N/A        |

To populate our RAG pipeline, the USDA plant fact sheets were stored and retrieved via an RPC in Supabase. 

## Overview
Users are able to upload novel images of plants and unique questions about the image. The pipeline identifies the plant species, 
disease, severity of disease, and returns the classification as well as relevant plant information. The following heads are classified via 
EffNetB0_224x224_MultiHeadClass_VisualModel: crop, disease, severity, genus, family and USDA symbol. If the model was trained with masks,
U-Net inputs via UNet_64x128x256x512x1024_PlantSeg are included in this process. A retriever class embeds the user's question and these visual predictions
with MiniLM_L6_384_Embedder, and Supabase is queried with a semantic similarity threshold of .5 as a default. Lastly, Qwen2.5_3B_QLoRA_ResponseModel
writes an answer for the user for up to 70 tokens, based on the initial question and image. Websockets are utilized to share the 
loading status in the form of live updates to the user, since this process does take some time, depending on their local hardware.

### RAG & Knowledge Base
The USDA NRCS plant fact sheets are stored per plant symbol in the form of JSON. Text is chunked by word into 192 word chunks with an overlap of 48
These are then embedded and ingested into a remote Supabase database that is invoked via an RPC.

### Image Datasets
- ***PlantExpertVQA (HF SyedNazmusSakib/PlantExpertVQA)***: This dataset consists of image, question, answer, crop and disease classification, and severity label
- ***GBIF plant images***: This open source image dataset is split 70/10/20 for train/val/test and is labeled with USDA taxonomy. Preprocessing is conducted to ensure disease classification and severity are added

## Installation

To install the system, execute the setup script:

`./setup.sh`

This script does the following:
1. Creates a virtual environment named 'venv'
2. Installs required python libraries into venv
3. Installs sub-requirements (like the nltk stop words package)
4. Runs pytest to verify installation

The script will exit if any of the steps fail.

Credentials for any necessary service, Anthropic key for evaluation and Supabase key and url for chunk retrieval, you must create your own `.env` file. The `.env.example` file shows a template for users to follow, but crucially should never commit their credentials to our repository 

## Running

After the project has been installed, you can execute code through the `Full_System.ipynb` notebook.
Ensure the kernel the notebook is using is the created `venv` environment. Open http://127.0.0.1:8000 which will present the 
locally hosted website.

## Testing

Pytest is used to verify the installation/setup of the project.  To run all pytest tests run the following command from the project root directory:
`python3 -m pytest`

To run just the setup test:
`python3 -m pytest test/test_setup.py`

Verify all tests pass.  If any tests from the test_setup.py script fail, there is a chance your environment/hardware is not properly configured to support PlantVQA.

## Repo Structure
- ***src/agents***: The models and agent trained
- ***src/data_preprocessing***: Any code needed for ETL of data
- ***src/evaluation***: Visual metrics and LLM Judge for the response
- ***src/ui***: The FastAPI backend and frontend for the locally hosted website
- ***notebooks/{exploration, training, preprocessing. evaluation}***: The notebooks required to train your own models
- ***archive/***: Deprecated notebooks and/or code


## Results

### Visual model: evaluated against the test set of 61,138 images

| Head     | Accuracy | Top-3 Accuracy | Macro F1 |
|----------|---------|---------------|---------|
| Disease  | 0.800   | 0.987         | 0.708   |
| Family   | 0.749   | 0.865         | 0.415   |
| Genus    | 0.743   | 0.835         | 0.420   |
| Symbol   | 0.693   | 0.771         | 0.371   |
| Severity | 0.691   | 0.975         | 0.591   |
| Crop     | 0.636   | 0.761         | 0.383   |

#### Severity class results

| Class    | Precision | Recall | F1  | Support |
|----------|----------|-------|-----|---------|
| SEVERE   | 0.72     | 0.60  | 0.65 | 5,448   |
| MODERATE | 0.61     | 0.55  | 0.58 | 9,243   |
| UNKNOWN  | 0.47     | 0.47  | 0.47 | 7,847   |
| MILD     | 0.54     | 0.34  | 0.42 | 7,691   |

### Hyperparameter tuning over 16 trials

| Image Size | Mask | LR    | Dropout |     Val Crop Acc |
|-----------|-----|-------|--------|-----------------|
| **224**   | No  | 1e-3  | 0.5    | **0.505** - Best |
| 224       | No  | 1e-3  | 0.3    |            0.499 |
| 224       | Yes | 1e-3  | 0.3    |            0.492 |
| 224       | Yes | 1e-3  | 0.5    |            0.492 |
| 112       | No  | 1e-3  | 0.5    |            0.443 |
| 112       | No  | 1e-4  | 0.5    |            0.429 |

### U-Net results

| Metric               | Value |
|----------------------|-------|
| Val Binary IoU (@0.7)| 0.536 |
| Val Dice loss        | 0.661 |
| Input size           | 384×384|

