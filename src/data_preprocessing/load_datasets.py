import src.data_preprocessing.text_preprocessing as text_pre
import src.config as config

def load_vqa_dataset():
    config_vqa = config.Data.PlantExpertVQA
    files = {"TRAIN": config_vqa.TRAIN_FILE,
             "TEST": config_vqa.TEST_FILE,
             "VALID": config_vqa.VALIDATION_FILE}
    datasets = {}
    for name, file_path in files:
        data = text_pre.load_csv(file_path)
        text_pre.preprocess_dataframe(
            data,
            config_vqa.TEXT_COLUMNS,
            config_vqa.COLUMNS_TO_REMOVE,
            config_vqa.NA_FILL,
            ["image_path"],
            config_vqa.ROOT,
        )
        datasets[name] = data

    return datasets["TRAIN"], datasets["TEST"], datasets["VALID"]