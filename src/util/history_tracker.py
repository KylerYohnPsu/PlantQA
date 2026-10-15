import pandas as pd
import json
from pathlib import Path
import src.config as config
from datetime import date
import os
import numpy as np

from src.util.logger import Logger
import src.util.general as general_util

class HistoryTracker:
    def __init__(self):
        self._initialize_dir()
        self._load_existing_history()
        Logger.info(f"History initialized!")

    def add(self, job_id, question, image):
        entry = self._make_entry(job_id, question, image)
        new_row = pd.DataFrame([entry]).astype(self._history.dtypes.to_dict())
        self._history = pd.concat([self._history, new_row], ignore_index=True)
        self.save()

    def update_answer(self, job_id, answer, confidence):
        matching_jobs= self._history["job_id"] == job_id
        self._history.loc[matching_jobs, "answer"]= answer
        self._history.loc[matching_jobs, "confidence"]= confidence
        self.save()

    def save(self):
        log_file= self._get_log_file_path()
        history= self._history.copy()

        for row, data in history.iterrows():
            job_id= data["job_id"]
            image= data["image"]
            image_path= self._save_image(job_id, image)
            history.at[row, "image"]= str(image_path)

        history.to_json(
            log_file,
            orient="records",
            lines=True,
            mode="w",
        )

    def get(self):
        return self._history.copy()

    def _load_existing_history(self):
        histories= []

        for file in self._historyDir.iterdir():
            if not general_util.is_json(file):
                continue

            Logger.info(f"Loading existing history file: {file}")

            history= pd.read_json(
                file,
                orient="records",
                lines=True,
            )

            if not history.empty:
                histories.append(history)

        if histories:
            self._history= pd.concat(histories, ignore_index=True)
        else:
            self._initialize_history()

        if not self._history.empty:
            Logger.info(f"Loading image history...")
            self._history["image"]= self._history["image"].apply(
                lambda path:
                    np.load(path) if isinstance(path, str) else path
            )

    def _initialize_history(self):
        self._history = pd.DataFrame({
            "image": pd.Series(dtype="object"),
            "question": pd.Series(dtype="object"),
            "answer": pd.Series(dtype="object"),
            "confidence": pd.Series(dtype="float64"),
            "job_id": pd.Series(dtype="object"),
        })

    def _initialize_dir(self):
        history_dir= os.path.join(config.General.DATA_DIR, "history")
        self._historyDir= Path(history_dir)
        self._historyDir.mkdir(parents=True, exist_ok=True)
        self._imageDir= self._historyDir / Path("images")
        self._imageDir.mkdir(parents=True, exist_ok=True)
        Logger.info(f"Q/A history recorded to: {self._historyDir}")

    def _make_entry(self, id, q, i):
        entry= {
            "question": q,
            "job_id": id,
            "image": i,
            "answer": None,
            "confidence": None,
        }

        return entry

    def _get_log_file_path(self):
        today= date.today()
        today_str= today.strftime("%d_%m_%Y")
        file_name= f"history_{today_str}.json"
        file_name_path= Path(file_name)
        file_path= self._historyDir / file_name_path
        return file_path

    def _save_image(self, job_id, image):
        image_path= self._imageDir / Path(f"{job_id}.npy")
        np.save(image_path, image)
        return image_path

HISTORY= HistoryTracker()