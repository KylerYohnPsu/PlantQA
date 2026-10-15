from fastapi import FastAPI, UploadFile, WebSocketDisconnect, File, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from src.util.logger import Logger
import time
import json
import asyncio
import threading
from contextlib import asynccontextmanager
import queue
import uvicorn
import uuid
import io
import numpy as np
from PIL import Image, ImageOps
from fastapi.staticfiles import StaticFiles
import webbrowser
import shutil 
import subprocess
from pathlib import Path

from src.util.prompt import Prompt
from src.util.system_status import SystemStatus

from src.config import UI as ui_config

class DisplayAPI:

    def __init__(self):
        self._server= None
        self._serverThread= None
        self._run= True
        self._status= SystemStatus.INITIALIZING
        self._initialize_queues()
        self._initialize_fast()
        self._initialize_routes()
        self._initialize_ui()
        self._status= SystemStatus.READY

        web_dir = Path(ui_config.WEB_DIR).resolve()

    def server_running(self):
        if self._serverThread is None:
            return False
        return self._serverThread.is_alive()

    def configure_server(self, host, port):
        config= uvicorn.Config(self.app, host=host, port=port)
        self._server= uvicorn.Server(config)

    def start(self, host="127.0.0.1", port=8000):
        self._run= True
        self.configure_server(host, port)
        self._serverThread= threading.Thread(
            target=self._server.run,
            daemon=True,
        )
        self._serverThread.start()
        Logger.critical(f"UI running at: http://127.0.0.1:8000/ui.html")
    
    def stop(self):
        if self._server is None:
            Logger.info(f"Server does not exist, nothing to stop.")
            return

        self._run= False

        Logger.info("Attempting to stop server...")

        self._server.should_exit= True
        if self._serverThread is not None:
            self._serverThread.join()

        self._server= None
        self._serverThread= None

        Logger.info("Server stopped.")

    def get_prompt(self):
        return self._dequeue(self._promptQueue)

    def set_status(self, status: SystemStatus):
        self._status= status

    def send_answer(self, answer, confidence, job_id):
        answer= {
            "answer": answer, 
            "confidence": confidence, 
            "job_id": job_id
        }
        self._enqueue(self._answerQueue, answer)

    """def decode_image(self, image_data: bytes) -> np.array:
        image = Image.open(io.BytesIO(image_data))
        image = ImageOps.exif_transpose(image)
        image= image.convert("RGB")
        image= np.asarray(image)
        return image"""

    def decode_image(self, image_data: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(image_data)) as img:
            image = np.asarray(img.convert("RGB")).copy()

        Logger.info(
            f"Decoded image: shape={image.shape}, "
            f"dtype={image.dtype}, contiguous={image.flags['C_CONTIGUOUS']}"
        )

        return image

    def _initialize_ui(self):
        self.app.mount("/", StaticFiles(directory=ui_config.WEB_DIR, html=True), name="web")

    def _initialize_fast(self):
        # make it so the js can communicate with this end
        self.app= FastAPI()

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _enqueue(self, q: queue.Queue, data: any):
        try:
            q.put(data)
        except queue.Full:
            Logger.error(f"Unable to enqueue data.  Queue is full.")

    def _dequeue(self, q: queue.Queue):
        try:
            data= q.get(block=False)
            return data
        except queue.Empty:
            return None

    def _initialize_queues(self):
        self._promptQueue= queue.Queue(maxsize=5)
        self._answerQueue= queue.Queue(maxsize=5)

    def _get_answer(self):
        try:
            return self._dequeue(self._answerQueue)
        except queue.Empty:
            return None

    def _initialize_routes(self):
        @self.app.post("/log")
        async def log(log: str=Form(...)):
            Logger.info(f"[JS] {log}")
            return {"status": "received"}
            
        @self.app.post("/ask")
        async def ask_question(image: UploadFile=File(...), question: str=Form(...)):
            image_data= await image.read()

            Logger.debug(f"Enqueuing prompt: {question}")

            job_id= str(uuid.uuid4().hex)
            decoded_image= self.decode_image(image_data)
            prompt= Prompt(question, decoded_image, job_id)
            self._enqueue(self._promptQueue, prompt)

            response= {
                "status": "received",
                "job_id": job_id,
            }
            return response

        @self.app.websocket("/answer")
        async def answer_updater(websocket: WebSocket):
            await websocket.accept()

            Logger.info("Answer websocket connection made!")

            try:
                while self._run:
                    answer= await asyncio.to_thread(self._get_answer)
                    if answer is None:
                        await asyncio.sleep(0.1)
                        continue
                    await websocket.send_json(answer)
            except WebSocketDisconnect:
                pass
            except Exception as E:
                Logger.error(f"Answer Updater Error: {E}")
            finally:
                Logger.info(f"Exiting answer thread.")

        @self.app.websocket("/status")
        async def status_updater(websocket: WebSocket):
            # connect to the js side
            await websocket.accept()

            Logger.info(f"Status websocket connection made!")

            try:
                while self._run:
                    message= {"status": self._status.value}

                    # send the status to the js
                    await websocket.send_json(message)

                    # dont kill the cpu
                    await asyncio.sleep(0.5)
            except WebSocketDisconnect:
                pass
            except Exception as E:
                Logger.error(f"Status Updater Error: {E}")
            finally:
                Logger.info(f"Exiting status thread.")
