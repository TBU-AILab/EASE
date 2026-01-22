from abc import ABC, abstractmethod
from typing import Optional

from core.game_state import GameParams, GameState, Player, Action


DEFAULT_OPPONENT = "Anon"


# === Fully observable agent interface ===
class PlanetWarsAgent(ABC):

    @abstractmethod
    def get_action(self, game_state: GameState) -> Action:
        pass

    @abstractmethod
    def get_agent_type(self) -> str:
        pass

    def prepare_to_play_as(
        self,
        player: Player,
        params: GameParams,
        opponent: Optional[str] = DEFAULT_OPPONENT
    ) -> str:
        return self.get_agent_type()

    def process_game_over(self, final_state: GameState) -> None:
        pass


# === Fully observable abstract base class ===
class PlanetWarsPlayer(PlanetWarsAgent):
    def __init__(self):
        self.player: Player = Player.Neutral
        self.params: GameParams = GameParams()

    def prepare_to_play_as(
        self,
        player: Player,
        params: GameParams,
        opponent: Optional[str] = DEFAULT_OPPONENT
    ) -> str:
        self.player = player
        self.params = params
        return self.get_agent_type()
