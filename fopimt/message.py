import uuid
from typing import Optional
import logging
import tiktoken
from datetime import datetime, timezone
import time

from pydantic import BaseModel


class MessageAPI(BaseModel):
    msg_id: str
    date_created: str
    tokens: int | None
    content: str
    model: str | None
    role: int


class Message:

    def __init__(self, role: str, message: str, model_encoding: Optional[str] = None):
        """
        Message class.
        Represent Message that will be sent from user or system to LLM.
        Arguments:
            model_encoding: str  -- Encoding of the LLM. Should be passed by instance of LLMConnector.
                                    Pass None if you don't want to predict the number of tokens. E.g. if the message is
                                    returned from LLM and you know exactly how many tokens were used.
            role: str            -- Role of the Message sender. Should be passed by instance of LLMConnector.
            message: str         -- Message content
        """

        self._id: str = str(uuid.uuid4())
        self._role: str = role
        self._content: str = message
        self._tokens: Optional[int] = None
        time.sleep(0.1) # Adams super HOTFIX
        self.update_timestamp()
        self._model = model_encoding
        self._metadata: dict = {}   # Optional metadata obtained from LLM

        self._calculate_tokens()

    ####################################################################
    #########  Public functions
    ####################################################################
    def get_API(self, roleProvider: list) -> MessageAPI:
        # role_dict = {"system": 0, "user": 1, "assistant": 2}
        return MessageAPI(
            msg_id=self._id,
            date_created=self._timestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            tokens=self._tokens,
            content=self._content,
            model=self._model,
            role=roleProvider.index(self._role)
        )


    def get(self) -> dict:
        """
        Get the formatted message suitable for LLM.
        Contains role definition and message.
        """
        return {"role": self._role,
                "content": self._content}

    def get_model(self) -> str:
        return self._model

    def set_tokens(self, tokens: int) -> None:
        """
        Set the number of used tokens. Probably as the result of the LLM received message.
        Arguments:
            tokens: int -- Number of used tokens of the message.
        """
        self._tokens = tokens

    def get_tokens(self) -> int:
        """
        Returns the number of used tokens.
        """
        return self._tokens

    def get_content(self) -> str:
        """
        Returns the content of the Message.
        """
        return self._content

    def set_model(self, model: str):
        """
        Set the model encoding of the used LLM connector. Calculate tokens using TikToken if necessary.
        :param model: Model (encoding) based on the LLM connector
        :return: None
        """
        self._model = model
        self._calculate_tokens()

    def get_timestamp(self):
        return self._timestamp

    def get_role(self) -> str:
        return self._role

    def get_id(self) -> str:
        return self._id

    def set_metadata(self, label: str, data) -> None:
        self._metadata[label] = data

    def get_metadata(self) -> dict:
        return self._metadata

    def update_timestamp(self) -> None:
        self._timestamp = datetime.now(timezone.utc)

    ####################################################################
    #########  Private functions
    ####################################################################

    def _calculate_tokens(self):
        """
        Calculate tokens based on the model. Only if the self._tokens == None or self._model != None
        :return: None
        """
        self._tokens = 0
        if self._model is not None and self._tokens is None:
            try:
                encoder = tiktoken.encoding_for_model(self._model)
                self._tokens = len(encoder.encode(self._content))
            except KeyError as e:
                logging.warning('Message: unknown model for tokenizer')
