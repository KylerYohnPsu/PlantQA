'''
Steps to Agent

1. Recieve question and Visual breakdown (either class or separated by features of a plant)

2. preprocess features of plant and get returned 
'''

"""Are we able to leverage the local host model"""
"""Add in crowd sourced data to the model"""

import agents.util.ResponseModel as ResponseModel
import agents.util.Retriever as Retriever
import agents.Visual_Agent as Visual_Agent

class NLPAgent:
    def __init__(self, retriever, response_model, visual_model):
        self.retriever = retriever
        self.response_model= response_model
        self.visual_model = visual_model
    def ask_question(self, image, question, plant_code_filter = None, num_results = 5):
        if image is not None and self.visual_model is not None and plant_code_filter is None:
            pred = self.visual_model.predict(image)
            plant_code_filter = pred.crop

        question_embedding = self.retriever.retrieve(question, plant_code_filter, num_results = num_results)

        answer = self.response_model.generate_answer(question = question, chunks = question_embedding)
        return answer