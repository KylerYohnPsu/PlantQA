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
import pandas as pd
from fastapi.responses import StreamingResponse

from src.util.prompt import Prompt
from src.util.system_status import SystemStatus

from src.config import UI as ui_config

from src.util.history_tracker import HISTORY

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
        """
        Return a boolean to indicate if the server is running
        """
        if self._serverThread is None:
            return False
        return self._serverThread.is_alive()

    def configure_server(self, host, port):
        """
        Prep the server settings
        """
        config= uvicorn.Config(self.app, host=host, port=port, access_log=False)
        self._server= uvicorn.Server(config)

    def start(self, host="127.0.0.1", port=8000):
        """
        Launch the server
        """
        Logger.info("Initializing UI Backend")
        self._run= True

        # Prep the server
        self.configure_server(host, port)

        # Launch the server thread
        self._serverThread= threading.Thread(
            target=self._server.run,
            daemon=True,
        )
        self._serverThread.start()
        Logger.critical(f"UI running at: http://127.0.0.1:8000/ui.html")
    
    def stop(self):
        """
        Kill the UI server
        """

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
        """
        Returns the oldest prompt the server has waiting
        """
        return self._dequeue(self._promptQueue)

    def set_status(self, status: SystemStatus):
        """
        Set the system status
        """
        self._status= status

    def send_answer(self, answer, confidence, job_id, classification=None, top_k=None,):
        """
        Send the given answer to the UI frontend
        """

        # parse the top K predictions from the visual model into a dictionary
        top_k_dict= {}
        for (plant, conf) in top_k:
            top_k_dict[plant]= round(conf,3)

        # This is the answer ditcionary that will be sent to the frontend
        answer= {
            "type": "answer",
            "answer": answer, 
            "confidence": round(confidence,3), 
            "job_id": job_id,
            "classification": classification,
            "top_k": top_k_dict,
        }

        # Put the answer in the answer send queue
        self._enqueue(self._answerQueue, answer)

    def decode_image(self, image_data: bytes) -> np.ndarray:
        """
        Parse image byte data into an np array
        """
        with Image.open(io.BytesIO(image_data)) as img:
            image = np.asarray(img.convert("RGB")).copy()

        return image

    def _initialize_ui(self):
        """
        Prep the ui by mounting the resource directory
        """
        self.app.mount("/", StaticFiles(directory=ui_config.WEB_DIR, html=True), name="web")

    def _initialize_fast(self):
        """
        Prep the fastapi server
        """

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
        """
        Put the given data in the given queue
        """
        try:
            q.put(data)
        except queue.Full:
            Logger.error(f"Unable to enqueue data.  Queue is full.")

    def _dequeue(self, q: queue.Queue):
        """
        Pop the oldest entry from the given queue
        """
        try:
            data= q.get(block=False)
            return data
        except queue.Empty:
            return None

    def _initialize_queues(self):
        """
        Prep queues the UI backend needs
        """
        self._promptQueue= queue.Queue(maxsize=5)
        self._answerQueue= queue.Queue(maxsize=5)

    def _get_answer(self):
        """
        Pop an answer from the answer queue, if there is one
        """
        try:
            return self._dequeue(self._answerQueue)
        except queue.Empty:
            return None

    def _initialize_routes(self):
        """
        Prep the FastAPI callback functions
        These are what the frontend and backend use to communicate with eachother
        """

        @self.app.post("/log")
        async def log(log: str=Form(...)):
            """
            Log the given string from the frontend to the terminal
            """
            Logger.info(f"[JS] {log}")
            return {"status": "received"}

        @self.app.get("/history")
        async def get_history():
            """
            The frontend calls this when it wants the history of user-system interactions
            Send the history to the frontend
            """
            if not HISTORY.initialized:
                return {"status": "fail"} # No history to send

            # get the history
            history= HISTORY.get()
            if history.empty:
                return {"status": "success", "history": []} # Empty history

            # Put each interaction into a list that we can send to the frontend
            records= []
            for record in history.to_dict(orient="records"):
            
                #Get the interaction info
                confidence= record["confidence"]
                question= record["question"]
                answer= record["answer"]
                job_id= record["job_id"]

                # put the interaction info into a dict
                this_record= {
                    "question": question,
                    "answer": answer,
                    "confidence": float(confidence) if pd.notna(confidence) else None,
                    "job_id": job_id,
                }

                # put the bbuilt interaction record into the list
                records.append(this_record)

            # send the interaction history to the frontend
            return {"status": "success", "history": records}

        @self.app.get("/history/{job_id}/image")
        async def get_history_image(job_id: str):
            """
            The frontend uses this to request an image from the history
            """
            # Get the history and find the matching image for the given job id
            history= HISTORY.get()
            matching= history[history["job_id"] == job_id]

            # if there is no matching image, send back nothing
            if matching.empty:
                return {"image": None}

            # Get the image data
            image_array= matching.iloc[-1]['image']

            try:
                # Turn the image into sendable byhtes
                image= Image.fromarray(np.asarray(image_array))
                buffer= io.BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)

                # Send the image back to the frontend
                return StreamingResponse(buffer, media_type="image/png")
            except Exception as E:
                Logger.error(f"Failed to get image: {E}")
                return {"image": None}
            
        @self.app.post("/ask")
        async def ask_question(image: UploadFile=File(...), question: str=Form(...), permission: bool=File(...)):
            """
            The frontend calls this when it has a prompt to give to the PLANTQA system
            """
            # Parse the image data
            image_data= await image.read()

            Logger.debug(f"Enqueuing prompt: {question}")

            # Put the prompt into the job queue so the system can answer it
            job_id= str(uuid.uuid4().hex)
            decoded_image= self.decode_image(image_data)
            prompt= Prompt(question, decoded_image, permission, job_id)
            self._enqueue(self._promptQueue, prompt)

            # Send the job id to the front end for tracking
            response= {
                "status": "received",
                "job_id": job_id,
            }
            return response

        @self.app.websocket("/answer")
        async def answer_updater(websocket: WebSocket):
            """
            Essentially a thread
            Send answers to the UI when the answers are ready
            """

            # Connect to the the frontend so we can send answers
            await websocket.accept()

            Logger.info("Answer websocket connection made!")

            try:
                # Run until the system shuts down
                while self._run:
                    # Get an answer from teh queue
                    answer= await asyncio.to_thread(self._get_answer)
                    if answer is None:
                        await websocket.send_json({"type":"heartbeat"}) #send connection heartbeat
                        await asyncio.sleep(0.3)
                        continue

                    # Send the answer to the UI for it to display
                    Logger.info("ANSWER: Attempting to send JSON")

                    await websocket.send_json(answer)

                    Logger.info("ANSWER: JSON sent successfully")
            except WebSocketDisconnect as E:
                Logger.info(f"Answer websocket disconnected. \nCode={E.code}, Reason={E.reason}")
            except Exception as E:
                Logger.error(f"Answer Updater Error: {E}")
            finally:
                Logger.info(f"Exiting answer thread.")

        @self.app.websocket("/status")
        async def status_updater(websocket: WebSocket):
            """
            Essentially a thread
            Send system status updates to the frontend
            """
            # connect to the js side
            await websocket.accept()

            Logger.info(f"Status websocket connection made!")

            try:
                # run untiltold to stop
                while self._run:
                    message= {"status": self._status.value}

                    # send the status to the js
                    await websocket.send_json(message)

                    # dont kill the cpu
                    await asyncio.sleep(0.5)
            except WebSocketDisconnect as E:
                Logger.info(f"Status websocket disconnected. \nCode={E.code}, Reason={E.reason}")
            except Exception as E:
                Logger.error(f"Status Updater Error: {E}")
            finally:
                Logger.info(f"Exiting status thread.")
