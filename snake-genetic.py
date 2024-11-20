import torch
import random
import numpy as np
from collections import deque
import pygame
import copy
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


class GeneticAgent:
    def __init__(self, input_size=16, hidden_size=256, output_size=3):
        self.model = Linear_QNet(input_size, hidden_size, output_size)
        self.fitness = 0
        self.games_played = 0
        self.total_score = 0
        self.steps_survived = 0
        self.moves_history = deque(maxlen=50)  # Track recent moves to detect loops

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Enhanced state representation
        state = [
            # Dangers (immediate)
            game.is_collision(point_l),
            game.is_collision(point_r),
            game.is_collision(point_u),
            game.is_collision(point_d),

            # Dangers (one step ahead)
            game.is_collision(Point(point_l.x - 20, point_l.y)),
            game.is_collision(Point(point_r.x + 20, point_r.y)),
            game.is_collision(Point(point_u.x, point_u.y - 20)),
            game.is_collision(Point(point_d.x, point_d.y + 20)),

            # Current direction
            dir_l, dir_r, dir_u, dir_d,

            # Food direction
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y,  # food down
        ]
        return np.array(state, dtype=int)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state)

        # Adaptive exploration
        exploration_rate = max(0.05, 0.3 - (self.games_played * 0.01))
        if random.random() < exploration_rate:
            move = random.randint(0, 2)
        else:
            move = torch.argmax(prediction).item()

        # Check for loops
        self.moves_history.append(move)
        if len(self.moves_history) == 50:
            if self._is_stuck_in_loop():
                move = random.randint(0, 2)  # Force random move if stuck

        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

    def _is_stuck_in_loop(self):
        if len(self.moves_history) < 20:
            return False
        # Check for repeating patterns
        moves = list(self.moves_history)
        for pattern_length in [4, 6, 8]:
            if len(moves) >= pattern_length * 2:
                pattern = moves[-pattern_length:]
                previous_pattern = moves[-2 * pattern_length:-pattern_length]
                if pattern == previous_pattern:
                    return True
        return False


class GeneticTrainer:
    def __init__(self, population_size=50, mutation_rate=0.2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.population = [GeneticAgent() for _ in range(population_size)]
        self.best_agent = None
        self.best_fitness = 0
        self.stats = {'gen': [], 'max_score': [], 'avg_score': [], 'best_fitness': []}

        # Initialize visualization
        pygame.init()
        self.viz_surface = pygame.Surface((800, 600))
        self.plot_fig = plt.figure(figsize=(6, 4))

    def calculate_fitness(self, score, steps, max_steps, food_distances):
        # Enhanced fitness function
        base_fitness = score * 100

        # Reward for efficient pathfinding
        avg_food_distance = sum(food_distances) / len(food_distances) if food_distances else float('inf')
        efficiency_bonus = 50 / (avg_food_distance + 1)

        # Survival reward with diminishing returns
        survival_factor = min(1.0, steps / (max_steps * 0.7))
        survival_reward = (survival_factor ** 0.5) * 50

        # Exploration bonus
        unique_positions = len(set(food_distances))
        exploration_bonus = unique_positions * 2

        return base_fitness + efficiency_bonus + survival_reward + exploration_bonus

    def train_generation(self):
        generation_scores = []
        max_steps = 1000

        for agent in self.population:
            game = SnakeGameAI()
            steps = 0
            food_distances = []

            while steps < max_steps:
                state = agent.get_state(game)
                action = agent.get_action(state)
                reward, done, score = game.play_step(action)

                # Record distance to food
                food_distance = abs(game.food.x - game.head.x) + abs(game.food.y - game.head.y)
                food_distances.append(food_distance)

                steps += 1
                if done:
                    break

            agent.fitness = self.calculate_fitness(score, steps, max_steps, food_distances)
            generation_scores.append(score)

            # Update statistics
            if agent.fitness > self.best_fitness:
                self.best_fitness = agent.fitness
                self.best_agent = copy.deepcopy(agent)

        # Update stats
        self.stats['gen'].append(self.generation)
        self.stats['max_score'].append(max(generation_scores))
        self.stats['avg_score'].append(sum(generation_scores) / len(generation_scores))
        self.stats['best_fitness'].append(self.best_fitness)

        # Evolution step
        self._evolve()
        self.generation += 1

        # Update visualization
        self._update_visualization()

        return max(generation_scores), sum(generation_scores) / len(generation_scores)

    def _evolve(self):
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Dynamic mutation rate based on population diversity
        fitness_variance = np.var([agent.fitness for agent in self.population])
        self.mutation_rate = min(0.4, max(0.1, 0.2 * (1 - fitness_variance / 1000)))

        new_population = []

        # Elitism - keep top performers
        elite_count = max(2, self.population_size // 10)
        new_population.extend([copy.deepcopy(agent) for agent in self.population[:elite_count]])

        # Create rest of population
        while len(new_population) < self.population_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            child = self._crossover(parent1, parent2)
            self._mutate(child)
            new_population.append(child)

        self.population = new_population

    def _update_visualization(self):
        # Clear previous plot
        plt.clf()

        # Create subplots
        plt.subplot(2, 1, 1)
        plt.plot(self.stats['gen'], self.stats['max_score'], label='Max Score')
        plt.plot(self.stats['gen'], self.stats['avg_score'], label='Avg Score')
        plt.legend()
        plt.title(f'Generation {self.generation} Progress')

        plt.subplot(2, 1, 2)
        plt.plot(self.stats['gen'], self.stats['best_fitness'], label='Best Fitness')
        plt.legend()

        # Convert matplotlib figure to pygame surface
        canvas = FigureCanvasAgg(self.plot_fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()

        # Update pygame surface
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        pygame.display.get_surface().blit(surf, (0, 0))
        pygame.display.flip()


def train():
    trainer = GeneticTrainer(population_size=100)

    while True:
        max_score, avg_score = trainer.train_generation()
        print(f"Generation {trainer.generation}:")
        print(f"Max Score: {max_score}")
        print(f"Average Score: {avg_score}")
        print(f"Best Fitness: {trainer.best_fitness}")
        print(f"Mutation Rate: {trainer.mutation_rate:.3f}")
        print("-" * 50)

        # Save best model periodically
        if trainer.generation % 10 == 0:
            trainer.best_agent.model.save(f'model_genetic_gen_{trainer.generation}.pth')


if __name__ == '__main__':
    train()