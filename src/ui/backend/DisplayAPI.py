from fastapi import FastAPI, UploadFile, File, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from src.util.logger import Logger
import time
import json

class DisplayAPI:

    def __init__(self):
        self._initialize_fast()
        self._initialize_routes()

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

    def _initialize_routes(self):
        @self.app.get("/")
        def root():
            return {"message": "Connection Running!"}

        @self.app.post("/log")
        async def log(log: str=Form(...)):
            Logger.info(f"[JS] {log}")
            return {"status": "received"}
            

        @self.app.post("/ask")
        async def ask_question(image: UploadFile=File(...), question: str=Form(...)):
            image_data= await image.read()

            Logger.info(f"Image: {image.filename}")
            Logger.info(f"Question: {question}")

            ###########################
            # Pass question/image to models here
            ############################

            response= {
                "status": "received",
                "job_id": "12345",
            }
            return response

        @self.app.websocket("/status")
        async def status_updater(websocket: WebSocket):
            # connect to the js side
            await websocket.accept()

            Logger.info(f"Status websocket connection made!")

            try:
                while True:
                    ignore_this= await websocket.receive_text()

                    ###########################
                    # UPDATE THE VQA's STATUS HERE
                    ###########################
                    status= "ready"
                    message= {"status": status}
                    message= json.dumps(message)

                    # send the status to the js
                    await websocket.send_text(message)

                    # dont kill the cpu
                    time.sleep(0.5)
            except Exception as E:
                Logger.error(f"Status Updater Error: {E}")


api= DisplayAPI()
app= api.app
