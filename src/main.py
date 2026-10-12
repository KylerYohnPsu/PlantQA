import src.config as config
from src.util.logger import Logger
from src.util.history_tracker import HISTORY
from src.ui.backend.DisplayAPI import DisplayAPI
from src.util.prompt import Prompt
from src.util.system_status import SystemStatus
from src.agents import NLP_Agent, U_NET, Visual_Agent
from src.agents.util import ResponseModel, Retriever

import argparse
import threading
import time
import random

class PlantQA:
    def __init__(self, ip_address, port):
        self._run= True
        self.launch_display(ip_address, port)

        ######
        # TODO: BUILD/LOAD MODELS HERE
        ######

    def launch_display(self, ip, port):
        """
        Make UI backend and launch UI communication thread
        """
        self._displayAPI= DisplayAPI()
        self._displayAPI.start(ip, port)

    def run(self):
        Logger.info(f"Launching primary PlantQA thread.")

        try :
            while self._run:
                self._vqa_loop()
        except KeyboardInterrupt:
            Logger.info(f"Shutdown command received.")
        except Exception as E:
            Logger.critical(f"Error in primary PlantQA thread: {E}")
        finally:
            self._cleanup()

    def _vqa_loop(self):
        
        self._displayAPI.set_status(SystemStatus.READY)

        prompt= self._displayAPI.get_prompt()
        if prompt == None:
            time.sleep(0.05)
            return

        self._displayAPI.set_status(SystemStatus.PREPROCESSING)

        HISTORY.add(prompt.job_id, prompt.question, prompt.image)

        Logger.info(f"Question received: {prompt.question}")
        Logger.info(f"Job ID: {prompt.job_id}")

        ########
        # TODO: preprocess prompt data here

        self._displayAPI.set_status(SystemStatus.PROCESSING)

        ########
        # TODO: hand question/image to models

        ########
        # TODO: get answer from models

        answer= self.TEMP_get_random_answer()
        confidence= round(random.random(), 2)

        HISTORY.update_answer(prompt.job_id, answer, confidence)

        self._displayAPI.send_answer(answer, confidence, prompt.job_id)

    def _cleanup(self):
        Logger.info(f"Shutting down PlantQA...")
        Logger.info(f"Stopping UI thread...")
        self._displayAPI.stop()
        Logger.info(f"UI thread stopped.")
        ########
        # TODO: Do any model cleanup here


    def TEMP_get_random_answer(self):
        answers = [
            "This appears to be a pine tree, identifiable by its needle-like leaves and woody stem.",
            "The browning may be caused by underwatering, excessive sunlight, or nutrient deficiencies.",
            "The leaves show possible signs of fungal infection, though further inspection would be needed to confirm.",
            "Most houseplants benefit from watering when the top inch or two of soil feels dry.",
            "Yellowing leaves can result from overwatering, poor drainage, or a lack of essential nutrients."
        ]

        return random.choice(answers)

def main(ip_address= "127.0.0.1", port=8000):
    plantqa= PlantQA(ip_address, port)
    plantqa.run()

if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="The IP address to serve the web UI over.")
    parser.add_argument("-p", "--port", type=int, default="8000", help="The port to serve the web UI over")
    args= parser.parse_args()

    main(args.ip, args.port)