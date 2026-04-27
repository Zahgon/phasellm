"""
Support for LLM evaluation.
"""

from typing import Optional, List

from .llms import OpenAIGPTWrapper, ChatBot

import pandas as pd

import random


def simulate_n_chat_simulations(chatbot: ChatBot, n: int, out_path_excel: Optional[str] = None) -> List[str]:
    """
    Reruns a chat message n times, returning a list of responses. Note that this will query an external API n times, so
    please be careful with costs.

    Args:
        chatbot: the chat sequence to rerun. The last message will be resent.
        n: number of times to run the simulation.
        out_path_excel: if provides, the output will also be written to an Excel file.

    Returns:
        A list of messages representing the responses in the chat.

    """
    pass


class BinaryPreference:

    def __init__(self, prompt: str, prompt_vars: str, response1: str, response2: str):
        """
        Tracks a prompt, prompt variables, responses, and the calculated preference.

        Args:
            prompt: The prompt
            prompt_vars: The variables to use in the prompt.
            response1: The first response.
            response2: The second response.

        """
        self.prompt = prompt
        self.prompt_vars = prompt_vars
        self.response1 = response1
        self.response2 = response2
        self.preference = -1

    def __repr__(self):
        return "<BinaryPreference>"

    def set_preference(self, pref):
        """
        Set the preference of the class.
        """
        pass

    def get_preference(self):
        """
        Get the preference of the class.
        """
        pass


class EvaluationStream:

    def __init__(self, objective, prompt, models):
        """
        Tracks human evaluation on the command line and records results.

        Args:
            objective: what you are trying to do.
            prompt: the prompt you are using. Could be a summary thereof, too. We do not actively use this prompt in
                generating data for evaluation.
            models: an array of two models. These can be referenced later if need be, but are not necessary for running
                the evaluation workflow.

        """
        self.models = models
        self.objective = objective
        self.prompt = prompt
        self.objective = objective
        self.evaluator = HumanEvaluatorCommandLine()
        self.prefs = [0] * len(models)  # This will be a simple counter for now.

    def __repr__(self):
        return f"<EvaluationStream>"

    def evaluate(self, response1, response2):
        """
        Shows both sets of options for review and tracks the result.
        """
        pass


class HumanEvaluatorCommandLine():

    def __init__(self):
        """
        Presents an objective, prompt, and two potential responses and has a human choose between the two.
        """
        pass

    def __repr__(self):
        return "<HumanEvaluatorCommandLine>"



class GPTEvaluator:

    def __init__(self, apikey, model="gpt-3.5-turbo"):
        """
        Passes two model outputs to GPT-3.5 or GPT-4 and has it decide which is the better output.

        Args:
            apikey: the OpenAI API key.
            model: the model to use. Defaults to GPT-3.5 Turbo.
        """
        self.model = OpenAIGPTWrapper(apikey, model=model)

    def __repr__(self):
        return f"GPT35Evaluator()"

    def choose(self, objective, prompt, response1, response2):
        """
        Presents the objective of the evaluation task, a prompt, and then two responses. GPT-3.5/GPT-4 chooses the
        preference.
        Args:
            objective: the objective of the modeling task.
            prompt: the prompt to use.
            response1: the first response.
            response2: the second response.

        Returns:
            1 if response1 is preferred, 2 if response2 is preferred.

        """
        pass
