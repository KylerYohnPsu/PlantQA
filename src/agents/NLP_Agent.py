'''
Steps to Agent

1. Recieve question and Visual breakdown (either class or separated by features of a plant)

2. preprocess features of plant and get returned 
'''

from src.util.system_status import SystemStatus
from src.util.logger import Logger

class NLPAgent:
    def __init__(self, retriever, response_model, visual_model, ui=None):
        self.retriever = retriever # A retriever class object to call out to Supabase
        self.response_model= response_model # The response model itself
        self.visual_model = visual_model # Visual model to reference predictions
        self.ui= ui

    def ask_question(self, image, question, plant_code_filter = None, num_results = 5, visual_predictions = None):
        prediction = None
        if visual_predictions is None and image is not None and self.visual_model is not None:
            self.set_ui_status(SystemStatus.PREDICTING)
            Logger.info(f"Predicting image class...")
            prediction = self.visual_model.predict(image) # Predict the image
            visual_predictions = prediction.get_observations() # List the observations of the image for selected classes

        self.set_ui_status(SystemStatus.EMBEDDING)
        Logger.info(f"Retrieving database embeddings...")
        question_embedding = self.retriever.retrieve(
            f"{question} {visual_predictions}" if visual_predictions else question,
            plant_code_filter = prediction.get_plant_code() if prediction else plant_code_filter,
            genus_filter = prediction.get_genus() if prediction else None,
            num_results = num_results) # We prefilter on genus, plant code, and include the question with visual observations for similarity

        self.set_ui_status(SystemStatus.GENERATING)
        Logger.info(f"Beginning answer generation...")
        answer = self.response_model.generate_answer(question = question, chunks = question_embedding, visual_predictions = visual_predictions)
        return answer

    def set_ui_status(self, status: SystemStatus):
        # update the UI's status
        if self.ui is not None:
            self.ui.set_status(status)