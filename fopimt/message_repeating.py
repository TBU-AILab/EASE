from enum import Enum
from typing import Optional
import random
import logging

from pydantic import BaseModel, Field

from .message import Message


class MessageRepeatingTypeEnum(Enum):
    SINGLE = 0
    RANDOM = 1
    RANDOM_WEIGHTED = 2
    CIRCULAR = 3


class MessageRepeatingConfig(BaseModel):
    type: MessageRepeatingTypeEnum = Field(...,
                                           description=(
                                               "The type of message repeating behavior:\n"
                                               "- `1` (SINGLE): Single message (first if more are defined).\n"
                                               "- `2` (RANDOM): Message randomly selected.\n"
                                               "- `3` (RANDOM_WEIGHTED): Message selected with weighted randomness.\n"
                                               "- `4` (CIRCULAR): Message send in a circular pattern."
                                           )
                                           )
    msgs: list[str]
    weights: list[float]


class MessageRepeating(Message):

    def __init__(self, role: str, msgs: list[str] | str, model_encoding: str | None = None,
                 msg_type: MessageRepeatingTypeEnum = MessageRepeatingTypeEnum.SINGLE,
                 weights: Optional[tuple[float, ...]] = None):
        """
        Repeating message class. Supports several types of strategies how to select new message text. (SINGLE,
        RANDOM, RANDOM_WEIGHTED, CIRCULAR)
        :param msgs: List of messages (str).
        :param msg_type: MessageRepeatingTypeEnum type.
        :param weights: Tuple of weights (float)
        """
        self._msgs = msgs
        self._type = msg_type
        self._weights = weights
        self._role = role
        self._encoding = model_encoding

        if isinstance(self._msgs, list) and len(self._msgs) == 0:
            self._msgs = "Improve."
            self._type = MessageRepeatingTypeEnum.SINGLE
            logging.warning('RepeatingMessage: Parameter msgs set incorrectly. No specified message/s. Setting '
                            'message to "Improve."')

        if isinstance(self._msgs, str) and self._type != MessageRepeatingTypeEnum.SINGLE:
            self._type = MessageRepeatingTypeEnum.SINGLE
            logging.warning('RepeatingMessage: Parameter msgs set incorrectly. Switching to SINGLE mode.')

        if self._type == MessageRepeatingTypeEnum.RANDOM_WEIGHTED and\
                (self._weights is None or len(self._msgs) != len(self._weights)):
            self._type = MessageRepeatingTypeEnum.RANDOM
            logging.warning('RepeatingMessage: Parameter weight set incorrectly. Switching to RANDOM mode.')

        self._index = 0
        msg = self._get_msg()

        super().__init__(message=msg, role=self._role, model_encoding=self._encoding)

    @property
    def index(self):
        return self._index

    @property
    def role(self):
        return self._role

    @property
    def encoding(self):
        return self._encoding

    @property
    def type(self):
        return self._type

    @property
    def weights(self):
        return self._weights

    @property
    def msgs(self):
        return self._msgs

    ####################################################################
    #########  Public functions
    ####################################################################
    def get_content(self) -> str:
        ret = super().get_content()
        msg = self._get_msg()
        super().__init__(message=msg, role=self._role, model_encoding=self._encoding)
        return ret

    ####################################################################
    #########  Private functions
    ####################################################################
    def _get_msg(self) -> str:
        match self._type:
            case MessageRepeatingTypeEnum.SINGLE:

                if isinstance(self._msgs, list):
                    return self._msgs[0]

                return self._msgs
            case MessageRepeatingTypeEnum.RANDOM:
                return random.choice(self._msgs)
            case MessageRepeatingTypeEnum.RANDOM_WEIGHTED:
                return random.choices(self._msgs, self._weights)[0]
            case MessageRepeatingTypeEnum.CIRCULAR:
                msg = self._msgs[self._index]
                self._index += 1
                self._index %= len(self._msgs)
                return msg
