import random
from typing import List

from core.game_state import Player, Vec2d, Planet, GameState, GameParams


class GameStateFactory:
    def __init__(self, params: GameParams):
        self.params = params

    def make_random_planet(self, owner: Player) -> Planet:
        x = random.uniform(0, self.params.width / 2)
        y = random.uniform(0, self.params.height)
        num_ships = random.uniform(
            self.params.min_initial_ships_per_planet,
            self.params.max_initial_ships_per_planet
        )
        growth_rate = random.uniform(
            self.params.min_growth_rate,
            self.params.max_growth_rate
        )
        radius = growth_rate * self.params.growth_to_radius_factor
        return Planet(
            owner=owner,
            n_ships=num_ships,
            position=Vec2d(x=x, y=y),
            growth_rate=growth_rate,
            radius=radius
        )

    def can_add(self, planets: List[Planet], candidate: Planet, radial_separation: float) -> bool:
        edge_sep = self.params.edge_separation
        if candidate.position.x - edge_sep < candidate.radius or \
           candidate.position.x + edge_sep > self.params.width / 2 - candidate.radius:
            return False
        if candidate.position.y - edge_sep < candidate.radius or \
           candidate.position.y + edge_sep > self.params.height - candidate.radius:
            return False
        for planet in planets:
            planet_radius = planet.growth_rate * self.params.growth_to_radius_factor
            dist = ((planet.position.x - candidate.position.x) ** 2 +
                    (planet.position.y - candidate.position.y) ** 2) ** 0.5
            if dist < radial_separation * (planet_radius + candidate.radius):
                return False
        return True

    def create_game(self) -> GameState:
        planets = []
        n_neutral = int(self.params.num_planets * self.params.initial_neutral_ratio) // 2

        while len(planets) < self.params.num_planets // 2:
            player = Player.Neutral if len(planets) < n_neutral else Player.Player1
            candidate = self.make_random_planet(player)
            if self.can_add(planets, candidate, self.params.radial_separation):
                planets.append(candidate)

        reflected_planets = []
        for planet in planets:
            reflected = Planet(
                owner=planet.owner,
                n_ships=planet.n_ships,
                position=Vec2d(
                    x=self.params.width - planet.position.x,
                    y=self.params.height - planet.position.y
                ),
                growth_rate=planet.growth_rate,
                radius=planet.radius
            )
            if planet.owner == Player.Player1:
                reflected.owner = Player.Player2
            reflected_planets.append(reflected)

        planets.extend(reflected_planets)

        for i, planet in enumerate(planets):
            planet.id = i

        return GameState(planets=planets)


if __name__ == "__main__":
    factory = GameStateFactory(GameParams())
    game = factory.create_game()
    print(game)
    for planet in game.planets:
        print(planet)
