import src.config as config
from src.util.logger import Logger
from src.util.history_tracker import HISTORY
from src.ui.backend.DisplayAPI import DisplayAPI
from src.util.prompt import Prompt
from src.util.system_status import SystemStatus
from src.agents.util import ResponseModel, Retriever
from src.agents import NLP_Agent, U_NET, Visual_Agent
from src.data_preprocessing import TextEmbedder, TextTokenizer
import src.util.supabase_utils as su
import src.data_preprocessing.text_preprocessing as text_pre
import src.data_preprocessing.image_preprocessing as img_pre
import keras

import argparse
import threading
import time
import random

class PlantQA:
    def __init__(self, ip_address="127.0.0.1", port=8000, match_threshold=0.5,
                 unet_weights= None, visual_model_weights=None,
                 visual_model_json=None, response_model_adapter=None):
        self._run= True

        # launch the UI server
        self.launch_display(ip_address, port)

        # build the models
        Logger.info(f"Prepping models...")
        self._initialize_models(
            match_threshold, 
            unet_weights,
            visual_model_weights,
            visual_model_json,
            response_model_adapter,    
        )
        Logger.info(f"Models ready!")

    def _initialize_models(self, match_threshold, unet_weights, visual_weights, visual_json, response_adapter):
        """
        Prep each of the models
        """
        Logger.info(f"Initializing embedding model...")
        self._embedder= TextEmbedder.TextEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        Logger.info(f"Embedding model initialized!")
        Logger.info(f"Connecting to supabase...")
        self._supabaseClient= su.connect_supabase()
        Logger.info(f"Supabase connected!")
        Logger.info(f"Initializing UNET model...")
        self._unet= keras.models.load_model(unet_weights, compile=False)
        Logger.info(f"UNET model initialized!")
        Logger.info(f"Building retriever...")
        self._retriever = Retriever.Retriever(self._supabaseClient, self._embedder, match_threshold=match_threshold)
        Logger.info(f"Retriever initialized!")
        Logger.info(f"Initializing response model...")
        self._responseModel= ResponseModel.ResponseModel(lora=True)
        Logger.info(f"Response model initialized!")
        Logger.info(f"Initializing visual model")
        self._visualModel= Visual_Agent.VisualModel(visual_weights, visual_json, unet=self._unet)
        Logger.info(f"Visual model initialized!")
        Logger.info(f"Prepping agent...")
        self._agent= NLP_Agent.NLPAgent(retriever=self._retriever, response_model=self._responseModel, visual_model=self._visualModel)
        Logger.info(f"Agent ready!")

    def launch_display(self, ip, port):
        """
        Make UI backend and launch UI communication thread
        """
        self._displayAPI= DisplayAPI()
        self._displayAPI.start(ip, port)

    def run(self):
        """
        Main thread
        Retrieve and answer questions
        """
        Logger.critical("PlantQA running and ready!")

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

        # get the prompt from UI server
        prompt= self._displayAPI.get_prompt()
        if prompt == None:
            time.sleep(0.05)
            return # no prompt, skip

        # put the prompt into the history, we update the answer later
        HISTORY.add(prompt.job_id, prompt.question, prompt.image)

        self._displayAPI.set_status(SystemStatus.PROCESSING)

        # Try to answer the question
        self._answer_question(prompt.question, prompt.image, prompt.job_id)

    def _answer_question(self, question, image, job_id):
        """
        Prep the inputs, ask the question, send the answer to the UI
        """
        # prep the question data
        question= self._prep_question(question)

        # Ask the agent the question and get the answer
        # TODO: have the agent return confidence, other metadata
        answer= self._agent.ask_question(image=image, question=question)
        confidence= round(random.random(), 2)

        # update the history to have the answer and confidence for the relevant question
        HISTORY.update_answer(job_id, answer, confidence)

        # send the answer to the display
        self._displayAPI.send_answer(answer, confidence, job_id)

    def _prep_question(self, question):
        """
        preprocess the question data
        """
        question= text_pre.normalize_whitespace(question)

    def _cleanup(self):
        """
        cleanup the plantqa system
        """
        Logger.info(f"Shutting down PlantQA...")
        Logger.info(f"Stopping UI thread...")
        self._displayAPI.stop()
        Logger.info(f"UI thread stopped.")


if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="The IP address to serve the web UI over.")
    parser.add_argument("-p", "--port", type=int, default="8000", help="The port to serve the web UI over")
    parser.add_argument("-u", "--unet", "--unet-weights", type=str, default=None, help="The path to the UNET model weights")
    parser.add_argument("-v", "--visual", "--visual-weights", type=str, default=None, help="The path to the visual model weights")
    parser.add_argument("-vj", "--json", "--visual-json", type=str, default=None, help="The path to the visual model class json")
    parser.add_argument("-a", "--adapter", "--response-adapter", type=str, default=None, help="The path to the response model adapter dir")
    parser.add_argument("-m", "--match", "--match-threshold", type=float, default=0.5, help="The match percentage for semantic search")
    
    args= parser.parse_args()

    plantqa= PlantQA(
        ip_address=args.ip, 
        port=args.port,
        match_threshold=args.match,
        unet_weights=args.unet,
        visual_model_weights=args.visual,
        visual_model_json=args.json,
        response_model_adapter=args.adapter,
    )

    plantqa.run()