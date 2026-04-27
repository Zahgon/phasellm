"""
Abstract classes and wrappers for LLMs, chatbots, and prompts.
"""

import re
import time
import json
import warnings

import httpx
import requests

from phasellm.llms_utils import (
    extract_vertex_ai_kwargs,
    extract_vertex_ai_response_metadata,
)

# Typing imports
from typing_extensions import TypedDict
from phasellm.types import CLAUDE_MODEL, OPENAI_API_CONFIG, VERTEXAI_API_CONFIG
from typing import Optional, List, Union, Generator, Any

# Configuration imports
from phasellm.configurations import OpenAIConfiguration, VertexAIConfiguration

# Abstract class imports
from abc import ABC, abstractmethod

from warnings import warn
from datetime import datetime
from dataclasses import asdict
from sseclient import SSEClient

# Support for VertexAI
from vertexai.generative_models import GenerationConfig, GenerativeModel
from vertexai.language_models import TextGenerationModel, ChatModel

# Imports for external APIs
import cohere

# Support for Replicate
import replicate

# Support for Anthropic's Messages API
import anthropic

# Precompiled regex for variables.
variable_pattern = r"\{\s*[a-zA-Z0-9_]+\s*\}"
variable_regex = re.compile(variable_pattern)

STOP_TOKEN = "<|END|>"


class Message(TypedDict):
    """
    Message type for chat messages.
    """

    role: str
    content: str


class EnhancedMessage(Message):
    """
    Message type for chat messages with additional metadata.
    """

    timestamp_utc: datetime
    log_time_seconds: float


def _fill_variables(source: str, **kwargs: Any) -> str:
    """
    Fills variables in a string with the values provided in kwargs.

    Args:
        source: The string to fill.
        **kwargs: The values to fill the string with.

    Returns:
        The filled string.

    """
    pass


def _clean_messages_to_prompt(messages: List[Message]) -> str:
    """
    Converts an array of messages in the form {"role": <str>, "content":<str>} into a String.

    This is influenced by the OpenAI chat completion API.

    Args:
        messages: The messages to convert.

    Returns:
        The messages as a String.

    """
    pass


def _truncate_completion(completion: str) -> str:
    """
    Truncates a completion to the first newline character.

    Args:
        completion: The completion to truncate.

    Returns:
        The truncated completion.

    """
    pass


def _remove_prompt_from_completion(prompt: str, completion: str) -> str:
    """
    Remove the prompt from the completion.

    Args:
        prompt: The prompt to remove.
        completion: The completion to remove the prompt from.

    Returns:
        The completion without the prompt.

    """
    pass


def _get_stop_sequences_from_messages(messages: List[Message]) -> List[str]:
    """
    Generates a list of strings of stop sequences from an array of messages in the form
    {"role": <str>, "content":<str>}.

    Args:
        messages: The messages to generate stop sequences from.

    Returns:
        A list of stop sequences.

    """
    pass


def _format_sse(content: str) -> str:
    """
    Formats the content for Server Sent Events (SSE). Additionally, handles newline characters gracefully.

    Args:
        content: The content to format.

    Returns:
        The formatted content.

    """
    pass


def _conditional_format_sse_response(content: str, format_sse: bool) -> str:
    """
    Conditionally formats the response as an SSE.

    Args:
        content: The content to format.
        format_sse: Whether to format the response as an SSE.

    Returns:
        The formatted content.

    """
    pass


def swap_roles(messages: List[Message], new_prompt: str) -> List[Message]:
    """
    Creates a new messages stack with the new_prompt as the system prompt and the 'user' and 'assistant' roles swapped.
    All other messages are ignored.

    Args:
        messages: the current messages.
        new_prompt: the new system prompt.

    Returns:
        A new list of messages with the new_prompt as the system prompt and user/assistant prompts swapped out.

    """
    pass


class LanguageModelWrapper(ABC):
    # default chat completion preamble
    chat_completion_preamble: str = (
        "You are a friendly chat assistant. "
        "You are speaking to the 'user' below and will respond at the end, where it says 'assistant'.\n"
    )

    _last_response_header: Optional[dict] = None

    def __init__(self, temperature: Optional[float] = None, **kwargs: Any):
        """
        Abstract Class for interacting with large language models.

        Args:
            temperature: The temperature to use for the language model.
            **kwargs: Keyword arguments to pass to the underlying language model API.

        """
        self.temperature: Optional[float] = temperature
        self.kwargs: Any = kwargs

    def __repr__(self):
        pass

    @property
    def last_response_header(self) -> Optional[dict]:
        """
        Returns the last response header from the LLM API.

        Returns:
            A dictionary containing the last response header.
        """
        pass

    @last_response_header.setter
    def last_response_header(self, header: dict) -> None:
        """
        Sets the last_response_header property.

        Returns:

        """
        pass

    @abstractmethod
    def complete_chat(
        self,
        messages: List[Message],
        append_role: Optional[str] = None,
        prepend_role: Optional[str] = None,
    ) -> Union[str, Generator]:
        """
        Takes an array of messages in the form {"role": <str>, "content":<str>} and generate a response.

        This is influenced by the OpenAI chat completion API.

        Args:
            messages: The messages to generate a response from.
            append_role: The role to append to the end of the prompt.
            prepend_role: The role to prepend to the beginning of the prompt.

        Returns:
            The chat completion string or generator, depending on if the class is implemented as a streaming language
            model wrapper.

        """
        pass

    @abstractmethod
    def text_completion(self, prompt: str) -> Union[str, Generator]:
        """
        Standardizes text completion for large language models.

        Args:
            prompt: The prompt to generate a response from.

        Returns:
            The text completion string or generator, depending on if the class is implemented as a streaming language
            model wrapper.

        """
        pass

    def prep_prompt_from_messages(
        self,
        messages: List[Message] = None,
        prepend_role: Optional[str] = None,
        append_role: Optional[str] = None,
        include_preamble: Optional[bool] = False,
    ) -> str:
        """
        Prepares the prompt for an LLM API call.

        Args:
            messages: The messages to prepare the prompt from.
            prepend_role: The role to prepend to the beginning of the prompt.
            append_role: The role to append to the end of the prompt.
            include_preamble: Whether to include the chat completion preamble.

        Returns:
            The prepared prompt.

        """
        pass

    @staticmethod
    def prep_prompt(
        prompt: str,
        prepend_role: Optional[str] = None,
        append_role: Optional[str] = None,
    ) -> str:
        """
        Prepares the prompt for an LLM API call.

        Args:
            prompt: The prompt to prepare.
            prepend_role: The role to prepend to the beginning of the prompt.
            append_role: The role to append to the end of the prompt.

        Returns:
            The prepared prompt.

        """
        pass

    def _prep_common_kwargs(self, api_config: Optional[OPENAI_API_CONFIG] = None):
        """
        This method prepares the common kwargs for the OpenAI APIs.

        Returns:
            The kwargs to pass to the API.

        """
        pass


class StreamingLanguageModelWrapper(LanguageModelWrapper):

    @abstractmethod
    def __init__(
        self,
        temperature: float,
        format_sse: bool,
        append_stop_token: bool = True,
        stop_token: str = STOP_TOKEN,
        **kwargs: Any,
    ):
        """
        Abstract class for streaming language models. Extends the regular LanguageModelWrapper.

        Args:
            temperature: The temperature to use for the language model.
            format_sse: Whether to format the response as an SSE.
            append_stop_token: Whether to append a stop token to the end of the prompt.
            stop_token: The stop token to append to the end of the prompt.
            **kwargs: Keyword arguments to pass to the underlying language model APIs.

        """
        super().__init__(temperature=temperature, **kwargs)
        self.format_sse = format_sse
        self.append_stop_token = append_stop_token
        self.stop_token = stop_token


class ChatPrompt:

    def __init__(self, messages: List[Message] = None):
        """
        This is used to generate messages for a ChatBot. Like the Prompt class, it enables you to to have variables that
        get replaced. This can be done for roles and messages.

        Args:
            messages: The messages to generate a chat prompt from.

        """
        # Set the messages
        if messages is None:
            self.messages = []
        else:
            self.messages = messages

    def __repr__(self):
        return "ChatPrompt()"

    def chat_repr(self) -> str:
        """
        Returns a string representation of the chat prompt.

        Returns:
            The string representation of the chat prompt.

        """
        pass

    def fill(self, **kwargs) -> List[Message]:
        """
        Fills the variables in the chat prompt.

        Args:
            **kwargs: The variables to fill.

        Returns:
            The filled chat prompt.

        """
        pass


class Prompt:

    def __init__(self, prompt: str):
        """
        Prompts are used to generate text completions. Prompts can be simple Strings. They can also include variables
        surrounded by curly braces.

        Example:
            >>> Prompt("Hello {name}!")
            In this case, 'name' can be filled using the fill() function. This makes it easier to loop through prompts
            that follow a specific pattern or structure.

        Args:
            prompt: The prompt to generate a text completion from.

        """
        self.prompt = prompt

    def __repr__(self):
        return self.prompt

    def get_prompt(self) -> str:
        """
        Return the raw prompt command (i.e., does not fill in variables.)

        Returns:
            The raw prompt command.

        """
        pass

    def fill(self, **kwargs: Any) -> str:
        """
        Return a prompt with variables filled in.

        Args:
            **kwargs: The variables to fill.

        Returns:
            The filled prompt.

        """
        pass


class HuggingFaceInferenceWrapper(LanguageModelWrapper):

    def __init__(
        self,
        apikey: str,
        model_url: str = "https://api-inference.huggingface.co/models/bigscience/bloom",
        temperature: float = None,
        **kwargs: Any,
    ):
        """
        Wrapper for Hugging Face's Inference API. Requires access to Hugging Face's inference API.

        Args:
            apikey: The API key to access the Hugging Face Inference API.
            model_url: The model URL to use for the Hugging Face Inference API.
            temperature: The temperature to use for the language model.
            **kwargs: Keyword arguments to pass to the Hugging Face Inference API.

        """
        super().__init__(temperature=temperature, **kwargs)
        self.apikey = apikey
        self.model_url = model_url

    def __repr__(self):
        return f"HuggingFaceInferenceWrapper()"

    def _call_model(self, prompt: str) -> str:
        """
        This method is used to call the Hugging Face Inference API. It is used by the complete_chat() and
        text_completion() methods.

        Args:
            prompt: The prompt to call the model with.

        Returns:
            The response from the Hugging Face Inference API.

        """
        pass

    def complete_chat(
        self, messages: List[Message], append_role: str = None, prepend_role: str = None
    ) -> str:
        """
        Mimics a chat scenario via a list of {"role": <str>, "content":<str>} objects.

        Args:
            messages: The messages to generate a chat completion from.
            append_role: The role to append to the end of the prompt.
            prepend_role: The role to prepend to the beginning of the prompt.

        Returns:
            The chat completion.

        """
        pass

    def text_completion(self, prompt: str) -> str:
        """
        Generates a text completion from a prompt.

        Args:
            prompt: The prompt to generate a text completion from.

        Returns:
            The text completion.

        """
        pass


# TODO consider deleting the BloomWrapper class since this functionality is in HuggingFaceInferenceWrapper
class BloomWrapper(LanguageModelWrapper):
    API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"

    def __init__(self, apikey: str, temperature: float = None, **kwargs: Any):
        """
        Wrapper for Hugging Face's BLOOM model. Requires access to Hugging Face's inference API.

        Args:
            apikey: The API key to access the Hugging Face Inference API.
            temperature: The temperature to use for the language model.
            **kwargs: Keyword arguments to pass to the underlying language model API.

        """
        super().__init__(temperature=temperature, **kwargs)
        self.apikey = apikey

    def __repr__(self):
        return f"BloomWrapper()"

    def _call_model(self, prompt: str) -> str:
        """
        This method is used to call the Hugging Face Inference API. It is used by the complete_chat() and
        text_completion() methods.

        Args:
            prompt: The prompt to call the model with.

        Returns:
            The response from the Hugging Face Inference API.

        """
        pass

    def complete_chat(
        self, messages: List[Message], append_role: str = None, prepend_role: str = None
    ) -> str:
        """
        Mimics a chat scenario with BLOOM, via a list of {"role": <str>, "content":<str>} objects.

        Args:
            messages: The messages to generate a chat completion from.
            append_role: The role to append to the end of the prompt.
            prepend_role: The role to prepend to the beginning of the prompt.

        Returns:
            The chat completion.

        """
        pass

    def text_completion(self, prompt: str) -> str:
        """
        Completes text via BLOOM (Hugging Face).

        Args:
            prompt: The prompt to generate a text completion from.

        Returns:
            The text completion.

        """
        pass


class StreamingOpenAIGPTWrapper(StreamingLanguageModelWrapper):

    def __init__(
        self,
        apikey: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        format_sse: bool = False,
        append_stop_token: bool = True,
        stop_token: str = STOP_TOKEN,
        temperature: float = None,
        api_config: Optional[OPENAI_API_CONFIG] = None,
        base_url:str = None,
        **kwargs: Any,
    ):
        """
        Streaming compliant wrapper for the OpenAI API. Supports all major text and chat completion models by OpenAI.

        This wrapper can be configured to use OpenAI's API or Microsoft Azure's API. To use Azure, pass in the
        appropriate api_config. To use OpenAI's API, pass in an apikey and model. If both api_config and apikey are
        passed in, api_config takes precedence.

        Examples:
            >>> from phasellm.llms import StreamingOpenAIGPTWrapper

            Use OpenAI's API:
                >>> llm = StreamingOpenAIGPTWrapper(apikey="my-api-key", model="gpt-3.5-turbo")
                >>> llm.text_completion(prompt="Hello, my name is")
                "Hello, my name is ChatGPT."

            Use OpenAI's API with api_config:
                >>> from phasellm.configurations import OpenAIConfiguration
                >>> llm = StreamingOpenAIGPTWrapper(api_config=OpenAIConfiguration(
                ...     api_key="my-api-key",
                ...     organization="my-org",
                ...     model="gpt-3.5-turbo"
                ... ))

            Use Azure's API:
                >>> from phasellm.configurations import AzureAPIConfiguration
                >>> llm = StreamingOpenAIGPTWrapper(api_config=AzureAPIConfiguration(
                ...     api_key="azure_api_key",
                ...     api_base='https://{your-resource-name}.openai.azure.com/openai/deployments/{your-deployment-id}',
                ...     api_version='2023-05-15',
                ...     deployment_id='your-deployment-id'
                ... ))
                >>> llm.text_completion(prompt="Hello, my name is")
                "Hello, my name is ChatGPT."

            Use Azure's API with Active Directory authentication:
                >>> from phasellm.configurations import AzureActiveDirectoryConfiguration
                >>> llm = StreamingOpenAIGPTWrapper(api_config=AzureActiveDirectoryConfiguration(
                ...     api_base='https://{your-resource-name}.openai.azure.com/openai/deployments/{your-deployment-id}',
                ...     api_version='2023-05-15',
                ...     deployment_id='your-deployment-id'
                ... ))
                >>> llm.text_completion(prompt="Hello, my name is")
                "Hello, my name is ChatGPT."

        Args:
            apikey: The API key to access the OpenAI API.
            model: The model to use. Defaults to "gpt-3.5-turbo".
            format_sse: Whether to format the SSE response from OpenAI. Defaults to False.
            append_stop_token: Whether to append the stop token to the end of the prompt. Defaults to True.
            stop_token: The stop token to use. Defaults to <|END|>.
            temperature: The temperature to use for the language model.
            api_config: The API configuration to use. Defaults to None. Takes precedence over apikey and model.
            base_url: The API endpoint, if you want to override it.
            **kwargs: Keyword arguments to pass to the OpenAI API.

        """
        super().__init__(
            format_sse=format_sse,
            append_stop_token=append_stop_token,
            stop_token=stop_token,
            temperature=temperature,
            **kwargs,
        )

        if base_url is None and model.find("deepseek") != -1:
            base_url="https://api.deepseek.com"

        if api_config and apikey:
            warn("api_config takes precedence over apikey and model arguments.")

        if apikey:
            self.api_config = OpenAIConfiguration(api_key=apikey, model=model, base_url=base_url)
        if api_config:
            self.api_config = api_config

        if not hasattr(self, "api_config"):
            raise Exception(
                "Must pass apikey or api_config. If using kwargs, check capitalization."
            )

        self.api_config.response_callback = self._set_last_response_header

        # Activate the configuration
        self.api_config()

    def __repr__(self):
        return f"StreamingOpenAIGPTWrapper(model={self.api_config.model})"

    def _yield_response(self, response) -> Generator:
        """
        Yields the response content. Can handle multiple API versions.

        Args:
            response: The response to yield text from.

        Returns:
            Text generator

        """
        pass

    def complete_chat(
        self, messages: List[Message], append_role: str = None, prepend_role: str = None
    ) -> Generator:
        """
        Completes chat with OpenAI. If using GPT 3.5 or 4, will simply send the list of {"role": <str>, "content":<str>}
        objects to the API.

        If using an older model, it will structure the messages list into a prompt first.

        Yields the text as it is generated, rather than waiting for the entire completion.

        Args:
            messages: The messages to generate a chat completion from.
            append_role: The role to append to the end of the prompt.
            prepend_role: The role to prepend to the beginning of the prompt.

        Returns:
            The chat completion generator.

        """
        pass

    # TODO Consider error handling for chat models.
    def text_completion(
        self, prompt: str, stop_sequences: List[str] = None
    ) -> Generator:
        """
        Completes text via OpenAI. Note that this doesn't support GPT 3.5 or later, as they are chat models.

        Yields the text as it is generated, rather than waiting for the entire completion.

        Args:
            prompt: The prompt to generate a text completion from.
            stop_sequences: The stop sequences to use. Defaults to None.

        Returns:
            The text completion generator.

        """
        pass

    def _set_last_response_header(self, response: httpx.Response) -> None:
        """
        Sets the last response header.

        Args:
            response: The response to set the last response header from.

        Returns:
            None

        """
        pass


class OpenAIGPTWrapper(LanguageModelWrapper):
    """
    Wrapper for the OpenAI API. Supports all major text and chat completion models by OpenAI.
    """

    def __init__(
        self,
        apikey: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = None,
        api_config: Optional[OPENAI_API_CONFIG] = None,
        base_url:str = None,
        **kwargs: Any,
    ):
        """
        Wrapper for the OpenAI API. Supports all major text and chat completion models by OpenAI.

        This wrapper can be configured to use OpenAI's API or Microsoft Azure's API. To use Azure, pass in the
        appropriate api_config. To use OpenAI's API, pass in an apikey and model. If both api_config and apikey are
        passed in, api_config takes precedence.

        Examples:
            >>> from phasellm.llms import OpenAIGPTWrapper

            Use OpenAI's API:
                >>> llm = OpenAIGPTWrapper(apikey="my-api-key", model="gpt-3.5-turbo")
                >>> llm.text_completion(prompt="Hello, my name is")
                "Hello, my name is ChatGPT."

            Use OpenAI's API with api_config:
                >>> from phasellm.configurations import OpenAIConfiguration
                >>> llm = OpenAIGPTWrapper(api_config=OpenAIConfiguration(
                ...     api_key="my-api-key",
                ...     organization="my-org",
                ...     model="gpt-3.5-turbo"
                ... ))

            Use Azure's API:
                >>> from phasellm.configurations import AzureAPIConfiguration
                >>> llm = OpenAIGPTWrapper(api_config=AzureAPIConfiguration(
                ...     api_key="azure_api_key",
                ...     api_base='https://{your-resource-name}.openai.azure.com/openai/deployments/{your-deployment-id}',
                ...     api_version='2023-08-01-preview',
                ...     deployment_id='your-deployment-id'
                ... ))
                >>> llm.text_completion(prompt="Hello, my name is")
                "Hello, my name is ChatGPT."

            Use Azure's API with Active Directory authentication:
                >>> from phasellm.configurations import AzureActiveDirectoryConfiguration
                >>> llm = OpenAIGPTWrapper(api_config=AzureActiveDirectoryConfiguration(
                ...     api_base='https://{your-resource-name}.openai.azure.com/openai/deployments/{your-deployment-id}',
                ...     api_version='2023-08-01-preview',
                ...     deployment_id='your-deployment-id'
                ... ))
                >>> llm.text_completion(prompt="Hello, my name is")
                "Hello, my name is ChatGPT."

        Args:
            apikey: The API key to access the OpenAI API.
            model: The model to use. Defaults to "gpt-3.5-turbo".
            temperature: The temperature to use for the language model.
            api_config: The API configuration to use. Defaults to None. Takes precedence over apikey and model.
            base_url: The API endpoint, if you want to override it.
            **kwargs: Keyword arguments to pass to the OpenAI API.

        """
        super().__init__(temperature=temperature, **kwargs)

        if base_url is None and model.find("deepseek") != -1:
            base_url="https://api.deepseek.com"

        if api_config and apikey:
            warn("api_config takes precedence over apikey and model arguments.")

        if apikey:
            self.api_config = OpenAIConfiguration(api_key=apikey, model=model, base_url=base_url)
        if api_config:
            self.api_config = api_config

        if not hasattr(self, "api_config"):
            raise Exception(
                "Must pass apikey or api_config. If using kwargs, check capitalization."
            )

        self.api_config.response_callback = self._set_last_response_header

        # Activate the configuration
        self.api_config()

    def __repr__(self):
        return f"OpenAIGPTWrapper(model={self.api_config.model})"

    def complete_chat(
        self, messages: List[Message], append_role: str = None, prepend_role: str = None
    ) -> str:
        """
        Completes chat with OpenAI. If using GPT 3.5 or 4, will simply send the list of {"role": <str>, "content":<str>}
        objects to the API.

        If using an older model, it will structure the messages list into a prompt first.

        Args:
            messages: The messages to generate a chat completion from.
            append_role: The role to append to the end of the prompt.
            prepend_role: The role to prepend to the beginning of the prompt.

        Returns:
            The chat completion.

        """
        pass

    # TODO Consider error handling for chat models.
    def text_completion(self, prompt: str, stop_sequences: List[str] = None) -> str:
        """
        Completes text via OpenAI. Note that this doesn't support GPT 3.5 or later, as they are chat models.

        Args:
            prompt: The prompt to generate a text completion from.
            stop_sequences: The stop sequences to use. Defaults to None.

        Returns:
            The text completion.

        """
        pass

    def _set_last_response_header(self, response: httpx.Response) -> None:
        """
        Sets the last response header.

        Args:
            response: The response to set the last response header from.

        Returns:
            None

        """
        pass


class StreamingVertexAIWrapper(StreamingLanguageModelWrapper):

    def __init__(
        self,
        model: str = None,
        format_sse: bool = False,
        append_stop_token: bool = True,
        stop_token: str = STOP_TOKEN,
        temperature: float = None,
        api_config: Optional[VERTEXAI_API_CONFIG] = None,
        **kwargs: Any,
    ):
        """
        Streaming wrapper for Vertex AI LLMs. Supports all major text and chat completion models, including Gemeni.

        This wrapper depends on Google's Application Default Credentials (ADC) to authenticate.

        Setting up ADC:
        1. Install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install
        2. Authenticate with gcloud: https://cloud.google.com/sdk/gcloud/reference/auth/application-default/login
        >>> gcloud auth application-default login

        Example:
            >>> from phasellm.llms import StreamingVertexAIWrapper

            Use Vertex AI's API:
                >>> llm = StreamingVertexAIWrapper(model="gemini-1.0-pro-001")
                >>> llm.text_completion(prompt="Hello, my name is")
                "Hello, my name is Gemeni."
            Note that when passing no model, the default model is "gemini-1.0-pro-001".

            Use Vertex AI's API with api_config:
                >>> from phasellm.configurations import VertexAIConfiguration
                >>> llm = StreamingVertexAIWrapper(api_config=VertexAIConfiguration(
                ...     model="gemini-1.0-pro-001"
                ... ))

            Use temperature parameter:
                >>> llm = VertexAIWrapper(model="gemini-1.0-pro-001", temperature=0.5)
                >>> llm.text_completion(prompt="Hello, my name is")
                "Hello, my name is Gemeni."

            Use max_output_tokens parameter:
                >>> llm = VertexAIWrapper(model="gemini-1.0-pro-001", max_output_tokens=50)
                >>> llm.text_completion(prompt="Hello, my name is")
                "Hello, my name is Gemeni."

            Potential parameters (model dependent):
                - max_output_tokens
                - candidate_count
                - top_p
                - top_k
                - logprobs
                - presence_penalty
                - frequency_penalty
                - logit_bias

        Args:
            model: The model to use. Defaults to "gemini-1.0-pro-001".
            format_sse: Whether to format the response as an SSE.
            append_stop_token: Whether to append a stop token to the end of the prompt.
            stop_token: The stop token to append to the end of the prompt.
            temperature: The temperature to use for the language model.
            api_config: The API configuration to use. Defaults to None. Takes precedence over model.
            **kwargs: Keyword arguments to pass to the Vertex AI API.

        """
        super().__init__(
            format_sse=format_sse,
            append_stop_token=append_stop_token,
            stop_token=stop_token,
            temperature=temperature,
            **kwargs,
        )

        if api_config and model:
            warn("api_config takes precedence over model arguments.")

        # Default model to gemini-1.0-pro.
        if not api_config and not model:
            model = "gemini-1.0-pro-001"

        if not api_config:
            self.api_config = VertexAIConfiguration(model=model)
        if api_config:
            self.api_config = api_config

        # Activate the configuration
        self.api_config()

    def __repr__(self):
        return f"StreamingVertexAIWrapper(model={self.api_config.model})"

    def _call_model(self, prompt: str, stop_sequences: List[str]) -> Generator:
        """
        Calls the model with the given prompt.

        Args:
            prompt: The prompt to generate a text completion from.
            stop_sequences: The stop sequences to use. Defaults to None.

        Returns:
            The text completion generator.

        """
        pass

    def complete_chat(
        self, messages: List[Message], append_role: str = None, prepend_role: str = None
    ) -> Generator:
        """
        Completes chat with Vertex AI.

        Args:
            messages: The messages to generate a chat completion from.
            append_role: The role to append to the end of the prompt.
            prepend_role: The role to prepend to the beginning of the prompt.

        Returns:
            The chat completion generator.

        """
        pass

    def text_completion(
        self, prompt: str, stop_sequences: List[str] = None
    ) -> Generator:
        """
        Completes text based on provided prompt.

        Yields the text as it is generated, rather than waiting for the entire completion.

        Args:
            prompt: The prompt to generate a text completion from.
            stop_sequences: The stop sequences to use. Defaults to None.

        Returns:
            The text completion generator.

        """
        pass


class VertexAIWrapper(LanguageModelWrapper):

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        api_config: Optional[VERTEXAI_API_CONFIG] = None,
        **kwargs: Any,
    ):
        """
        Wrapper for Vertex AI LLMs. Supports all major text and chat completion models, including Gemeni.

        This wrapper depends on Google's Application Default Credentials (ADC) to authenticate.

        Setting up ADC:
        1. Install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install
        2. Authenticate with gcloud: https://cloud.google.com/sdk/gcloud/reference/auth/application-default/login
        >>> gcloud auth application-default login

        Example:
            >>> from phasellm.llms import VertexAIWrapper

            Text completion:
                >>> llm = VertexAIWrapper(model="gemini-1.0-pro-001")
                >>> llm.text_completion(prompt="Hello, my name is")
                "Hello, my name is Gemeni."
            Note that when passing no model, the default model is "gemini-1.0-pro-001".

            Configure with api_config:
                >>> from phasellm.configurations import VertexAIConfiguration
                >>> llm = VertexAIWrapper(api_config=VertexAIConfiguration(
                ...     model="gemini-1.0-pro-001"
                ... ))

            Use temperature parameter:
                >>> llm = VertexAIWrapper(model="gemini-1.0-pro-001", temperature=0.5)
                >>> llm.text_completion(prompt="Hello, my name is")
                "Hello, my name is Gemeni."

            Use max_output_tokens parameter:
                >>> llm = VertexAIWrapper(model="gemini-1.0-pro-001", max_output_tokens=50)
                >>> llm.text_completion(prompt="Hello, my name is")
                "Hello, my name is Gemeni."

            Potential parameters (model dependent):
                - max_output_tokens
                - candidate_count
                - top_p
                - top_k
                - logprobs
                - presence_penalty
                - frequency_penalty
                - logit_bias

        Args:
            model: The model to use. Defaults to "gemini-1.0-pro-001".
            temperature: The temperature to use for the language model.
            api_config: The API configuration to use. Defaults to None. Takes precedence over model.
            **kwargs: Keyword arguments to pass to the Vertex AI API.
        """

        super().__init__(temperature=temperature, **kwargs)

        if api_config and model:
            warn("api_config takes precedence over model arguments.")

        # Default model to gemini-1.0-pro.
        if not api_config and not model:
            model = "gemini-1.0-pro-001"

        if not api_config:
            self.api_config = VertexAIConfiguration(model=model)
        if api_config:
            self.api_config = api_config

        # Activate the configuration
        self.api_config()

    def __repr__(self):
        return f"VertexAIWrapper(model={self.api_config.model})"

    def _call_model(self, prompt: str, stop_sequences: List[str]) -> str:
        """
        Calls the model with the given prompt.

        Args:
            prompt: The prompt to generate a text completion from.
            stop_sequences: The stop sequences to use. Defaults to None.

        Returns:
            The text completion.

        """
        pass

    def complete_chat(
        self, messages: List[Message], append_role: str = None, prepend_role: str = None
    ) -> str:
        """
        Completes chat.

        Args:
            messages: The messages to generate a chat completion from.
            append_role: The role to append to the end of the prompt.
            prepend_role: The role to prepend to the beginning of the prompt.

        Returns:
            The chat completion.

        """
        pass

    def text_completion(self, prompt: str, stop_sequences: List[str] = None) -> str:
        """
        Completes text based on provided prompt.

        Args:
            prompt: The prompt to generate a text completion from.
            stop_sequences: The stop sequences to use. Defaults to None.

        Returns:
            The text completion.

        """
        pass


class StreamingClaudeWrapper(StreamingLanguageModelWrapper):
    API_URL = "https://api.anthropic.com/v1/complete"

    def __init__(
        self,
        apikey: str,
        model: CLAUDE_MODEL = "claude-2",
        format_sse: bool = False,
        append_stop_token: bool = True,
        stop_token: str = STOP_TOKEN,
        temperature: float = None,
        anthropic_version: str = "2023-06-01",
        **kwargs: Any,
    ):
        """
        Streaming wrapper for Anthropic's Claude large language model.

        We've opted to call Anthropic's API directly rather than using their Python offering.

        Yields the text as it is generated, rather than waiting for the entire completion.

        Args:
            apikey: The API key to access the Anthropic API.
            model: The model to use. Defaults to "claude-2".
            format_sse: Whether to format the SSE response. Defaults to False.
            append_stop_token: Whether to append the stop token to the end of the prompt. Defaults to True.
            stop_token: The stop token to use. Defaults to <|END|>.
            temperature: The temperature to use for the language model.
            anthropic_version: The version of the Anthropic API to use. See https://docs.anthropic.com/claude/reference/versioning
            **kwargs: Keyword arguments to pass to the Anthropic API.

        """
        super().__init__(
            format_sse=format_sse,
            append_stop_token=append_stop_token,
            stop_token=stop_token,
            temperature=temperature,
            **kwargs,
        )
        self.apikey = apikey
        self.model = model
        self.anthropic_version = anthropic_version

    def __repr__(self):
        return f"StreamingClaudeWrapper(model={self.model})"

    def _call_model(self, prompt: str, stop_sequences: List[str]) -> Generator:
        """
        Calls the model with the given prompt.

        Args:
            prompt: The prompt to generate a text completion from.
            stop_sequences: The stop sequences to use. Defaults to None.

        Returns:
            The text completion generator.

        """
        pass

    def complete_chat(
        self,
        messages: List[Message],
        append_role: str = "Assistant",
        prepend_role: str = "Human",
    ) -> Generator:
        """
        Completes chat with Claude. Since Claude doesn't support a chat interface via API, we mimic the chat via a
        prompt.

        Args:
            messages: The messages to generate a chat completion from.
            append_role: The role to append to the end of the prompt. Defaults to "Assistant:".
            prepend_role: The role to prepend to the beginning of the prompt. Defaults to "Human:".

        Returns:
            The chat completion generator.

        """
        pass

    def text_completion(
        self, prompt: str, stop_sequences: List[str] = None
    ) -> Generator:
        """
        Completes text based on provided prompt.

        Yields the text as it is generated, rather than waiting for the entire completion.

        Args:
            prompt: The prompt to generate a text completion from.
            stop_sequences: The stop sequences to use. Defaults to None.

        Returns:
            The text completion generator.

        """
        pass


class ClaudeMessagesWrapper(LanguageModelWrapper):

    def __init__(
        self,
        apikey: str,
        model: str = "claude-3-5-sonnet-20240620",
        max_tokens: int = 4096,
        temperature: float = None,
        **kwargs: Any,
    ):
        super().__init__(temperature=temperature, **kwargs)
        self.apikey = apikey
        self.model = model
        self.max_tokens = max_tokens

    def __repr__(self):
        return f"ClaudeMessagesWrapper(model={self.model})"

    def complete_chat(
        self,
        messages: List[Message],
        append_role: str = "Assistant",
        prepend_role: str = "User",
    ) -> str:
        """
        Completes chat with Claude. Since Claude doesn't support a chat interface via API, we mimic the chat via a
        prompt.

        Args:
            messages: The messages to generate a chat completion from.
        Returns:
            The chat completion.

        """
        pass

    def text_completion(self, prompt: str, stop_sequences: List[str] = None) -> str:
        """
        Completes text based on provided prompt.

        Args:
            prompt: The prompt to generate a text completion from.
            stop_sequences: The stop sequences to use. Defaults to None.

        Returns:
            Error; ClaudeMessagesWrapper does not support text completion.

        """

        raise NotImplementedError(
            "ClaudeMessagesWrapper does not support text completion."
        )


class StreamingClaudeMessagesWrapper(StreamingLanguageModelWrapper):

    def __init__(
        self,
        apikey: str,
        model: str = "claude-3-5-sonnet-20240620",
        format_sse: bool = False,
        append_stop_token: bool = True,
        stop_token: str = STOP_TOKEN,
        max_tokens: int = 4096,
        temperature: float = None,
        **kwargs: Any,
    ):
        """
        Streaming wrapper for Anthropic's Claude large language model.

        We've opted to call Anthropic's API directly rather than using their Python offering.

        Yields the text as it is generated, rather than waiting for the entire completion.

        Args:
            apikey: The API key to access the Anthropic API.
            model: The model to use. Defaults to "claude-2".
            format_sse: Whether to format the SSE response. Defaults to False.
            append_stop_token: Whether to append the stop token to the end of the prompt. Defaults to True.
            stop_token: The stop token to use. Defaults to <|END|>.
            temperature: The temperature to use for the language model.
            **kwargs: Keyword arguments to pass to the Anthropic API.

        """
        super().__init__(
            format_sse=format_sse,
            append_stop_token=append_stop_token,
            stop_token=stop_token,
            temperature=temperature,
            **kwargs,
        )
        self.apikey = apikey
        self.model = model
        self.max_tokens = max_tokens

    def __repr__(self):
        return f"StreamingClaudeMessagesWrapper(model={self.model})"

    def _call_model(self, messages: List[Message], system_prompt=None) -> Generator:
        """
        Calls the model with the given prompt.

        Args:
            messages: The messages to generate a chat completion from.

        Returns:
            The text completion generator.

        """
        pass

    def complete_chat(
        self,
        messages: List[Message],
        append_role: str = "Assistant",
        prepend_role: str = "User",
    ) -> Generator:
        """
        Completes chat with Claude. Since Claude doesn't support a chat interface via API, we mimic the chat via a
        prompt.

        Args:
            messages: The messages to generate a chat completion from.
            append_role: The role to append to the end of the prompt. Defaults to "Assistant:".
            prepend_role: The role to prepend to the beginning of the prompt. Defaults to "Human:".

        Returns:
            The chat completion generator.

        """
        pass

    def text_completion(
        self, prompt: str, stop_sequences: List[str] = None
    ) -> Generator:
        """
        Completes text based on provided prompt.

        Yields the text as it is generated, rather than waiting for the entire completion.

        Args:
            prompt: The prompt to generate a text completion from.
            stop_sequences: The stop sequences to use. Defaults to None.

        Returns:
            Error; StreamingClaudeMessagesWrapper does not support text completion.

        """

        raise NotImplementedError(
            "ClaudeMessagesWrapper does not support text completion."
        )


class ClaudeWrapper(LanguageModelWrapper):
    API_URL = "https://api.anthropic.com/v1/complete"

    def __init__(
        self,
        apikey: str,
        model: CLAUDE_MODEL = "claude-2",
        temperature: float = None,
        anthropic_version: str = "2023-06-01",
        **kwargs: Any,
    ):
        """
        Wrapper for Anthropic's Claude large language model.

        We've opted to call Anthropic's API directly rather than using their Python offering.

        See here for model options: https://docs.anthropic.com/claude/reference/selecting-a-model

        Args:
            apikey: The API key to access the Anthropic API.
            model: The model to use. Defaults to "claude-v1".
            temperature: The temperature to use for the language model.
            anthropic_version: The version of the Anthropic API to use. See https://docs.anthropic.com/claude/reference/versioning
            **kwargs: Keyword arguments to pass to the Anthropic API.

        """
        super().__init__(temperature=temperature, **kwargs)
        self.apikey = apikey
        self.model = model
        self.anthropic_version = anthropic_version

    def __repr__(self):
        return f"ClaudeWrapper(model={self.model})"

    def _call_model(self, prompt: str, messages: List[Message]) -> str:
        """
        Calls the model with the given prompt.

        Args:
            prompt: The prompt to call the model with.
            messages: The messages to generate stop sequences from.

        Returns:
            The completion.

        """
        pass

    def complete_chat(
        self,
        messages: List[Message],
        append_role: str = "Assistant",
        prepend_role: str = "Human",
    ) -> str:
        """
        Completes chat with Claude. Since Claude doesn't support a chat interface via API, we mimic the chat via a
        prompt.

        Args:
            messages: The messages to generate a chat completion from.
            append_role: The role to append to the end of the prompt. Defaults to "Assistant:".
            prepend_role: The role to prepend to the beginning of the prompt. Defaults to "Human:".

        Returns:
            The chat completion.

        """
        pass

    def text_completion(self, prompt: str, stop_sequences: List[str] = None) -> str:
        """
        Completes text based on provided prompt.

        Args:
            prompt: The prompt to generate a text completion from.
            stop_sequences: The stop sequences to use. Defaults to None.

        Returns:
            The text completion.

        """
        pass


# TODO Might want to add stop sequences (new lines, roles) to make this better.
class GPT2Wrapper(LanguageModelWrapper):

    def __init__(self, temperature: float = None, **kwargs: Any):
        """
        Wrapper for GPT-2 implementation (via Hugging Face).

        Note that you must have the phasellm[complete] extra installed to use this wrapper.

        Args:
            temperature: The temperature to use for the language model.
            **kwargs: Keyword arguments to pass to the GPT-2 model.

        """
        super().__init__(temperature=temperature, **kwargs)

        # Dynamically import torch and transformers to avoid import errors for users who don't have phasellm[complete].
        from transformers import pipeline

        self.model_name = "GPT-2"
        self.pipeline = pipeline("text-generation", model="gpt2")

    def __repr__(self):
        return f"GPT2Wrapper({self.model_name})"

    def _call_model(self, prompt: str, max_length: int = 300) -> str:
        """
        Calls the model with the given prompt.

        Args:
            prompt: The prompt to call the model with.
            max_length: The maximum length of the completion. Defaults to 300.

        Returns:
            The completion.

        """
        pass

    def complete_chat(
        self,
        messages: List[Message],
        append_role: str = None,
        max_length: int = 300,
        prepend_role: str = None,
    ) -> str:
        """
        Mimics a chat scenario via a list of {"role": <str>, "content":<str>} objects.

        Args:
            messages: The messages to generate a chat completion from.
            append_role: The role to append to the end of the prompt. Defaults to None.
            max_length: The maximum length of the completion. Defaults to 300.
            prepend_role: The role to prepend to the beginning of the prompt. Defaults to None.

        Returns:
            The chat completion.

        """
        pass

    def text_completion(self, prompt: str, max_length: int = 200) -> str:
        """
        Completes text via GPT-2.

        Args:
            prompt: The prompt to generate a text completion from.
            max_length: The maximum length of the completion. Defaults to 200.

        Returns:
            The text completion.

        """
        pass


class DollyWrapper(LanguageModelWrapper):

    def __init__(self, temperature: float = None, **kwargs: Any):
        """
        Wrapper for Dolly 2.0 (via Hugging Face).

        Note that you must have the phasellm[complete] extra installed to use this wrapper.

        Args:
            temperature: The temperature to use for the language model.
            **kwargs: Keyword arguments to pass to the Dolly model.

        """
        super().__init__(temperature=temperature, **kwargs)

        # Dynamically import torch and transformers to avoid import errors for users who don't have phasellm[complete].
        from transformers import pipeline
        import torch

        self.model_name = "dolly-v2-12b"
        self.pipeline = pipeline(
            "text-generation",
            model="databricks/dolly-v2-12b",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    def __repr__(self):
        return f"DollyWrapper(model={self.model_name})"

    def _call_model(self, prompt: str) -> str:
        """
        Calls the model with the given prompt.

        Args:
            prompt: The prompt to call the model with.

        Returns:
            The completion.

        """
        pass

    def complete_chat(
        self, messages: List[Message], append_role: str = None, prepend_role: str = None
    ) -> str:
        """
        Mimics a chat scenario via a list of {"role": <str>, "content":<str>} objects.

        Args:
            messages: The messages to generate a chat completion from.
            append_role: The role to append to the end of the prompt. Defaults to None.
            prepend_role: The role to prepend to the beginning of the prompt. Defaults to None.

        Returns:
            The chat completion.

        """
        pass

    def text_completion(self, prompt: str) -> str:
        """
        Completes text via Dolly.

        Args:
            prompt: The prompt to generate a text completion from.

        Returns:
            The text completion.

        """
        pass


class ReplicateLlama2Wrapper(LanguageModelWrapper):

    base_system_chat_prompt = "You are a friendly chatbot."
    """str: Used in defining the system prompt for Llama 2 calls. Only used if a system prompt doesn't exist in the message stack provided."""

    first_user_message = "Hi."
    """str: Used as the first 'user' message in chat completions when the chat's first non-system message is from 'assistant'."""

    def __init__(
        self,
        apikey: str,
        model: str = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        temperature: float = None,
        **kwargs: Any,
    ):
        """
        Wrapper for Llama 2, provided via Replicate. See https://replicate.com/ for more information.

        Args:
            apikey: The Replicate API key to use.
            model: The Llama 2 model to use.
            temperature: The temperature to use for the language model.
            **kwargs: Keyword arguments to pass to the API.
        """
        super().__init__(temperature=temperature, **kwargs)
        self.model = model
        self.apikey = apikey
        self.temperature = temperature if temperature is not None else 0.5

    def __repr__(self):
        return f"ReplicateLlama2Wrapper(model={self.model})"

    def build_chat_completion_prompt(self, messages: List[Message]):
        """
        Converts a list of messages into a properly structured Llama 2 prompt for chats, as outlined here: https://huggingface.co/blog/llama2#how-to-prompt-llama-2

        Note that we make a few changes to the process above: (1) if a system prompt does not exist, we use a very basic default prompt. Otherwise, we convert the system prompt to the relevant instructions in the Llama 2 chat prompt.

        Next, we iterate through the the rest of the message stack. We assume that the message stack contains alternating messages from a 'user' and 'assistant'. If your first messages outside of a system prompt is from the 'assistant', then we include a 'Hi.' messages from the user.

        Please note that this is a work in progress. If you have feedback on the process about, email w --at-- phaseai --dot-- com

        Args:
            messages: The messages to generate a chat completion from.
            append_role: The role to append to the end of the prompt. Defaults to None.
            prepend_role: The role to prepend to the beginning of the prompt. Defaults to None.

        Returns:
            The chat completion.

        """
        pass

    def _clean_response(self, assistant_message: str) -> str:
        """
        Cleans up the chat response, mainly by stripping whitespace and removing "Assistant:" prepends.

        Args:
            assistant_message: The message received from the API.

        Returns:
            The chat completion, cleaned up.

        """
        pass

    def complete_chat(
        self, messages: List[Message], append_role: str = None, prepend_role: str = None
    ) -> str:
        """
        Mimics a chat scenario via a list of {"role": <str>, "content":<str>} objects.

        Args:
            messages: The messages to generate a chat completion from.

        Returns:
            The chat completion.

        """
        pass

    def text_completion(self, prompt: str, stop_sequences: List[str] = None) -> str:
        """
        Completes text via Replicate's Llama 2 service.

        Args:
            prompt: The prompt to generate a text completion from.
            stop_sequences: The stop sequences to use. Defaults to None.

        Returns:
            The text completion.

        """
        pass


class CohereWrapper(LanguageModelWrapper):

    def __init__(
        self,
        apikey: str,
        model: str = "xlarge",
        temperature: float = None,
        **kwargs: Any,
    ):
        """
        Wrapper for Cohere's API.

        Args:
            apikey: The API key to use.
            model: The model to use. Defaults to "xlarge".
            temperature: The temperature to use for the language model.
            **kwargs: Keyword arguments to pass to the Cohere API.

        """
        super().__init__(temperature=temperature, **kwargs)
        self.model = model
        self.co = cohere.Client(apikey)

    def __repr__(self):
        return f"CohereWrapper(model={self.model})"

    def _call_model(self, prompt, stop_sequences: List[str]):
        """
        Calls the model with the given prompt.

        Args:
            prompt: The prompt to call the model with.
            stop_sequences: The stop sequences to use.

        Returns:
            The completion.

        """
        pass

    def complete_chat(
        self, messages: List[Message], append_role: str = None, prepend_role: str = None
    ) -> str:
        """
        Mimics a chat scenario via a list of {"role": <str>, "content":<str>} objects.

        Args:
            messages: The messages to generate a chat completion from.
            append_role: The role to append to the end of the prompt. Defaults to None.
            prepend_role: The role to prepend to the beginning of the prompt. Defaults to None.

        Returns:
            The chat completion.

        """
        pass

    def text_completion(self, prompt: str, stop_sequences: List[str] = None) -> str:
        """
        Completes text via Cohere.

        Args:
            prompt: The prompt to generate a text completion from.
            stop_sequences: The stop sequences to use. Defaults to None.

        Returns:
            The text completion.

        """
        pass


class ChatBot:

    def __init__(
        self,
        llm: LanguageModelWrapper,
        initial_system_prompt: str = "You are a friendly chatbot assistant.",
    ):
        """
        Allows you to have a chat conversation with an LLM wrapper.

        In short, it manages the list of {"role": <str>, "content":<str>} objects for you, so you don't have to figure
        this out. It also interacts directly with the model.

        Warning: not all LLMs are trained to use instructions provided in a system prompt.

        Args:
            llm: The LLM wrapper to use for the ChatBot.
            initial_system_prompt: The initial system prompt to use. Defaults to "You are a friendly chatbot
            assistant.". Use this to change the behavior of the chatbot.
        """
        self.llm: LanguageModelWrapper = llm
        self.messages: List[EnhancedMessage] = []
        self.append_message("system", initial_system_prompt)

    def _response(self, response: str, start_time: float) -> str:
        """
        Handles a response from the LLM by appending it to the message stack.

        Args:
            response: The response from the LLM.
            start_time: The start time of the request.

        Returns:
            The response.

        """
        pass

    def _streaming_response(self, response: Generator, start_time: float) -> Generator:
        """
        Handles a streaming response from the LLM by appending it to the message stack.

        Since the response is a generator, we'll need intercept it so that we can append it to the message stack.
        (Generators only yield their results once).

        Args:
            response: The response from the LLM.
            start_time: The start time of the request.

        Returns:
            The response.

        """
        pass

    def append_message(
        self, role: str, message: str, log_time_seconds: float = None
    ) -> None:
        """
        Saves a message to the ChatBot message stack.

        Args:
            role: The role of the message.
            message: The message.
            log_time_seconds: The time it took to generate the message. Defaults to None.

        """
        pass

    def resend(self) -> Optional[Union[str, Generator]]:
        """
        If the last message in the messages stack (i.e. array of role and content pairs) is from the user, it will
        resend the message and return the response. It's similar to erasing the last message in the stack and resending
        the last user message to the chat model.

        This is useful if a model raises an error or if you are building a broader messages stack outside of the
        actual chatbot.

        Returns:
            The response from the chatbot if the last message in the stack was from the user. Otherwise, None.

        """
        pass

    def chat(self, message: str) -> Union[str, Generator]:
        """
        Chats with the chatbot.

        Args:
            message: The message to send to the chatbot.

        Returns:
            The response from the chatbot. Either a string or a generator, depending on if a streaming LLM wrapper is
            used.

        """
        pass
