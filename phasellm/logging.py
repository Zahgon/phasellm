"""
Logging support. This allows you to use the phasellm library to send chats to evals.phasellm.com and review them via our hosted front-end.
"""

import requests
import json

from typing import List, Optional

from .llms import Message, ChatBot

import hashlib
import os

_PHASELLM_EVALS_BASE_URL = "https://evals.phasellm.com/api/v0.1"


class FileLogger:
    """
    This logger will save chats to disk. It will export chats to a flat TXT file.
    """

    def __init__(self, folder_path: str, separator: str = "\n\n-----------------\n\n"):
        """
        Args:
            folder_path: The path to the folder where the logs will be saved.
            separator: The separator between messages in the log file.
        """
        self.folder_path = folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.separator = separator

    def log(
        self,
        messages: List[Message],
        chat_id: Optional[int] = None,
        title: Optional[str] = None,
        source_id: Optional[str] = None,
        file_name: str = None,
    ) -> str:
        """
        Saves or updates the relevant chat to a folder.

        Args:
            messages: The messages array from the chat.
            chat_id: Optional chat ID. If you provide a chat ID from an earlier log event, the messages will overwrite the original chat. This should be used for updating conversations rather than replacing them.
            title: Optional title for the chat.
            source_id: Optional String representing an ID for the chat. This is to enable easier referencing of chats for end users and is not used by PhaseLLM Evals.
            file_name: Optional String for what to call the file. Otherwise will use chat_id. If chat_id is not given, then will use an MD5 sum of the content.

        Returns:
            The chat_id associated with the chat.
        """
        pass

    def logChatBot(
        self,
        chatbot: ChatBot,
        chat_id: Optional[int] = None,
        title: Optional[str] = None,
        source_id: Optional[str] = None,
        file_name: str = None,
    ) -> str:
        """
        Logs the message stack for a chatbot to a folder.

        Args:
            chatbot: The chatbot object to log.
            chat_id: Optional chat ID. If you provide a chat ID from an earlier log event, the messages will overwrite the original chat. This should be used for updating conversations rather than replacing them.
            title: Optional title for the chat.
            source_id: Optional String representing an ID for the chat. This is to enable easier referencing of chats for end users and is not used by PhaseLLM Evals.
            file_name: Optional String for what to call the file. Otherwise will use chat_id. If chat_id is not given, then will use an MD5 sum of the content.

        Returns:
            The chat_id associated with the chat.
        """
        pass


class PhaseLogger:

    def __init__(
        self,
        apikey: str,
    ):
        """
        Helper class for logging chats to evals.phasellm.com.

        Args:
            apikey: The API key associated with your evals.phasellm.com account.
        """
        super().__init__()
        self.apikey = apikey

    def log(
        self,
        messages: List[Message],
        chat_id: Optional[int] = None,
        title: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> int:
        """
        Saves or updates the relevant chat at evals.phasellm.com

        Args:
            messages: The messages array from the chat.
            chat_id: Optional chat ID. If you provide a chat ID from an earlier log event, the messages will overwrite the original chat. This should be used for updating conversations rather than replacing them.
            title: Optional title for the chat.
            source_id: Optional String representing an ID for the chat. This is to enable easier referencing of chats for end users and is not used by PhaseLLM Evals.

        Returns:
            The chat_id associated with the chat.
        """
        pass

    def logChatBot(
        self,
        chatbot: ChatBot,
        chat_id: Optional[int] = None,
        title: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> int:
        """
        Logs the message stack for a chatbot to evals.phasellm.com.

        Args:
            chatbot: The chatbot object to log.
            chat_id: Optional chat ID. If you provide a chat ID from an earlier log event, the messages will overwrite the original chat. This should be used for updating conversations rather than replacing them.
            title: Optional title for the chat.
            source_id: Optional String representing an ID for the chat. This is to enable easier referencing of chats for end users and is not used by PhaseLLM Evals.

        Returns:
            The chat_id associated with the chat.
        """
        pass
