
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchvision, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import torch


from torch import Tensor


from torch import nn


from typing import Callable


from torch.nn import CrossEntropyLoss


from torch.nn import Transformer


from torch.optim import Adam


from torch.optim.lr_scheduler import ReduceLROnPlateau


import torch.nn as nn


import torch.optim as optim


from typing import List


from typing import Dict


from typing import Any


from copy import deepcopy


import torch.nn.functional as F


from torch.utils.data import DataLoader


class AntColonyOptimization(nn.Module):
    """
    Ant Colony Optimization
    Overview: https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms

    How does it work?
    1. Initialize pheromone levels for each path
    2. For each ant, choose the next path based on the pheromone levels
    3. Update the pheromone levels
    4. Repeat step 2 to 3 until the maximum number of iterations is reached

    Parameters
    ----------
    goal: str
        The goal string to be optimized
    num_ants: int
        Number of ants
    evaporation_rate: float
        Evaporation rate

    Usage
    -----
    from swarms_torch import AntColonyOptimization

    goal_string = "Hello ACO"
    aco = AntColonyOptimization(goal_string, num_iterations=1000)
    best_solution = aco.optimize()

    print("Best Matched String:", best_solution)

    Features to implement
    --------
    1. Add a stopping criterion
    2. Add a callback function to track the progress
    3. Add a function to plot the pheromone levels
    4. Add a function to plot the ants
    5. Add a function to plot the best solution

    """

    def __init__(self, goal: 'str'=None, num_ants: 'int'=10000, evaporation_rate: 'float'=0.1, alpha: 'int'=1, beta: 'int'=1, num_iterations: 'int'=10010):
        self.goal = torch.tensor([ord(c) for c in goal], dtype=torch.float32)
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.num_iterations = num_iterations
        self.pheromones = torch.ones(num_ants)
        self.solutions = []

    def fitness(self, solution):
        """Fitness of a solution"""
        return -torch.norm(solution - self.goal)

    def update_pheromones(self):
        """Update pheromone levels"""
        for i, solution in enumerate(self.solutions):
            self.pheromones[i] = (1 - self.evaporation_rate) * self.pheromones[i] + self.fitness(solution)

    def choose_next_path(self):
        """Choose the next path based on the pheromone levels"""
        probabilities = self.pheromones ** self.alpha * (1.0 / (1 + self.pheromones)) ** self.beta
        probabilities /= probabilities.sum()
        return torch.multinomial(probabilities, num_samples=1).item()

    def optimize(self):
        """Optimize the goal string"""
        for iteration in range(self.num_iterations):
            self.solutions = []
            for _ in range(self.num_ants):
                solution = torch.randint(32, 127, (len(self.goal),), dtype=torch.float32)
                self.solutions.append(solution)
            self.update_pheromones()
        best_solution_index = self.pheromones.argmax().item()
        best_solution = self.solutions[best_solution_index]
        return ''.join([chr(int(c)) for c in best_solution])


class TransformerCell(nn.Module):

    def __init__(self, input_dim, nhead, num_layers=1, neighborhood_size=3):
        super(TransformerCell, self).__init__()
        self.transformer = nn.Transformer(input_dim, nhead=nhead, num_encoder_layers=num_layers)
        self.neighborhood_size = neighborhood_size

    def forward(self, x, neigbors):
        x = self.transformer(x, x)
        out = torch.cat([x] + neigbors, dim=0)
        return out


class CellularSwarm(nn.Module):
    """
    CellularSwarm

    Architecture:
        - Input -> TransformerCell -> TransformerCell -> ... -> Output

    Overview:
    CellularSwarm is a cellular neural network that uses a transformer cell
    to process the input.

    Args:
        cell_count (int): Number of transformer cells
        input_dim (int): Input dimension
        nhead (int): Number of heads in the transformer cell
        time_steps (int): Number of time steps to run the network

    Returns:
        torch.Tensor: Output tensor

    Usage:
        >>> x = torch.randn(10, 32, 512)
        >>> model = CellularSwarm(cell_count=5, input_dim=512, nhead=8)
        >>> output = model(x)
        >>> print(output)


    """

    def __init__(self, cell_count, input_dim, nhead, time_steps=4):
        super(CellularSwarm, self).__init__()
        self.cells = nn.ModuleList([TransformerCell(input_dim, nhead) for _ in range(cell_count)])
        self.time_steps = time_steps

    def forward(self, x):
        for _ in range(self.time_steps):
            for i, cell in enumerate(self.cells):
                start_idx = max(0, i - cell.neighborhood_size)
                end_idx = min(len(self.cells), i + cell.neighborhood_size + 1)
                neighbors = [self.cells[j].transformer(x, x) for j in range(start_idx, end_idx) if j != i]
                x = cell(x, neighbors)
        return x


class FireflyOptimizer(nn.Module):

    def __init__(self, cost_function: 'Callable[[Tensor], Tensor]', steps: 'int'=5000, species: 'int'=4, population_size: 'int'=1000, dimensions: 'int'=10, lower_bound: 'float'=-4.0, upper_bound: 'float'=4.0, mix_species_every: 'int'=25, beta0: 'float'=2.0, gamma: 'float'=1.0, alpha: 'float'=0.1, alpha_decay: 'float'=0.995, use_genetic_algorithm: 'bool'=False, breed_every: 'int'=10, tournament_size: 'int'=100, num_children: 'int'=500, use_cuda: 'bool'=True, verbose: 'bool'=True):
        """
        Initialize the FireflyOptimizer.

        Parameters
        ----------
        cost_function : Callable[[Tensor], Tensor]
            The objective function to minimize.
        steps : int, optional
            Number of optimization steps, by default 5000.
        species : int, optional
            Number of species, by default 4.
        population_size : int, optional
            Size of the population per species, by default 1000.
        dimensions : int, optional
            Dimensionality of the problem, by default 10.
        lower_bound : float, optional
            Lower bound for the variables, by default -4.0.
        upper_bound : float, optional
            Upper bound for the variables, by default 4.0.
        mix_species_every : int, optional
            Steps interval at which species mix, by default 25.
        beta0 : float, optional
            Base attractiveness, by default 2.0.
        gamma : float, optional
            Light absorption coefficient, by default 1.0.
        alpha : float, optional
            Randomness scaling factor, by default 0.1.
        alpha_decay : float, optional
            Decay rate of alpha per step, by default 0.995.
        use_genetic_algorithm : bool, optional
            Whether to use genetic algorithm operations, by default False.
        breed_every : int, optional
            Steps interval at which breeding occurs, by default 10.
        tournament_size : int, optional
            Size of the tournament for selection, by default 100.
        num_children : int, optional
            Number of children to produce during breeding, by default 500.
        use_cuda : bool, optional
            Whether to use CUDA if available, by default True.
        verbose : bool, optional
            Whether to print progress, by default True.
        """
        self.cost_function = cost_function
        self.steps = steps
        self.species = species
        self.population_size = population_size
        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mix_species_every = mix_species_every
        self.beta0 = beta0
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.use_genetic_algorithm = use_genetic_algorithm
        self.breed_every = breed_every
        self.tournament_size = tournament_size
        self.num_children = num_children
        self.use_cuda = use_cuda
        self.verbose = verbose
        assert self.tournament_size <= self.population_size, 'Tournament size must be less than or equal to population size.'
        assert self.num_children <= self.population_size, 'Number of children must be less than or equal to population size.'
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        self.fireflies = torch.zeros((self.species, self.population_size, self.dimensions), device=self.device).uniform_(self.lower_bound, self.upper_bound)
        self.current_alpha = self.alpha

    def optimize(self) ->None:
        """
        Run the Firefly optimization algorithm.
        """
        for step in range(self.steps):
            costs = self.cost_function(self.fireflies)
            if self.verbose:
                logger.info(f'Step {step}: Minimum cost {costs.amin():.5f}')
            move_mask = einx.greater('s i, s j -> s i j', costs, costs)
            delta_positions = einx.subtract('s j d, s i d -> s i j d', self.fireflies, self.fireflies)
            distance = delta_positions.norm(dim=-1)
            betas = self.beta0 * torch.exp(-self.gamma * distance ** 2)
            attraction = einx.multiply('s i j, s i j d -> s i j d', move_mask * betas, delta_positions)
            random_walk = self.current_alpha * (torch.rand_like(self.fireflies) - 0.5) * (self.upper_bound - self.lower_bound)
            self.fireflies += einx.sum('s i j d -> s i d', attraction) + random_walk
            self.fireflies.clamp_(min=self.lower_bound, max=self.upper_bound)
            self.current_alpha *= self.alpha_decay
            if self.species > 1 and step % self.mix_species_every == 0:
                midpoint = self.population_size // 2
                fireflies_a = self.fireflies[:, :midpoint]
                fireflies_b = self.fireflies[:, midpoint:]
                rotated_fireflies_b = torch.roll(fireflies_b, shifts=1, dims=(0,))
                self.fireflies = torch.cat((fireflies_a, rotated_fireflies_b), dim=1)
            if self.use_genetic_algorithm and step % self.breed_every == 0:
                self._genetic_operations(costs)

    def _genetic_operations(self, costs: 'Tensor') ->None:
        """
        Perform genetic algorithm operations: selection, crossover, and replacement.

        Parameters
        ----------
        costs : Tensor
            Costs associated with each firefly.
        """
        fitness = 1.0 / costs
        batch_randperm = torch.randn((self.species, self.num_children, self.population_size), device=self.device).argsort(dim=-1)
        tournament_indices = batch_randperm[..., :self.tournament_size]
        tournament_participants = einx.get_at('s [p], s c t -> s c t', fitness, tournament_indices)
        winners_per_tournament = tournament_participants.topk(2, dim=-1).indices
        parent1, parent2 = einx.get_at('s [p] d, s c parents -> parents s c d', self.fireflies, winners_per_tournament)
        crossover_mask = torch.rand_like(parent1) < 0.5
        children = torch.where(crossover_mask, parent1, parent2)
        _, sorted_indices = costs.sort(dim=-1)
        sorted_fireflies = einx.get_at('s [p] d, s sorted -> s sorted d', self.fireflies, sorted_indices)
        self.fireflies = torch.cat((sorted_fireflies[:, :-self.num_children], children), dim=1)

    def get_best_solution(self) ->Tensor:
        """
        Retrieve the best solution found by the optimizer.

        Returns
        -------
        Tensor
            The best solution vector.
        """
        fireflies_flat = einx.rearrange('s p d -> (s p) d', self.fireflies)
        costs = self.cost_function(fireflies_flat)
        sorted_costs, sorted_indices = costs.sort(dim=-1)
        best_firefly = fireflies_flat[sorted_indices[0]]
        best_cost = sorted_costs[0]
        logger.info(f'Best solution found with cost {best_cost:.5f}')
        return best_firefly

    def generate(self) ->Tensor:
        """
        Generate a new set of fireflies.

        Returns
        -------
        Tensor
            The new set of fireflies.
        """
        self.fireflies = torch.zeros((self.species, self.population_size, self.dimensions), device=self.device).uniform_(self.lower_bound, self.upper_bound)
        self.current_alpha = self.alpha
        return self.fireflies

    def reset(self) ->None:
        """
        Reset the optimizer to its initial state.
        """
        self.generate()
        self.current_alpha = self.alpha


class Fish(nn.Module):
    """
    A fish is a transformer model with a negative loss as food.

    Parameters
    ----------
    dim : int
        The number of expected features in the input (required).
    heads : int
        The number of heads in the multiheadattention models (required).
    depth : int
        The number of sub-encoder-layers in the encoder (required).

    Attributes

    model : torch.nn.Transformer
        The transformer model.
    food : float
        The fish's food, which is the negative loss of the model.

    Methods
    =======
    train(src, tgt, labels)
        Train the model with the given source, target, and labels.


    Usage:
    >>> fish = Fish(512, 8, 6)
    >>> fish.train(src, tgt, labels)
    >>> fish.food
    -0.123456789



    Example2
    # Create random source and target sequences
    src = torch.randn(10, 32, 512)
    tgt = torch.randn(10, 32, 512)

    # Create random labels
    labels = torch.randint(0, 512, (10, 32))

    # Create a fish and train it on the random data
    fish = Fish(512, 8, 6)
    fish.train(src, tgt, labels)
    print(fish.food)  # Print the fish's food

    # Create a fish school and optimize it on the random data
    school = FishSchool(10, 512, 8, 6, 100)
    school.forward(src, tgt, labels)
    print(school.fish[0].food)  # Print the first fish's food


    """

    def __init__(self, dim, heads, depth, dynamic_learning_rate=False, early_stopping=False, complexity_regularization=False, max_patience=None, alpha=0.1):
        super().__init__()
        self.model = Transformer(d_model=dim, nhead=heads, num_encoder_layers=depth, num_decoder_layers=depth)
        self.optimizer = Adam(self.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        self.complexity_regularization = complexity_regularization
        self.dynamic_learning_rate = dynamic_learning_rate
        self.early_stopping = early_stopping
        self.food = 0
        self.best_food = float('inf')
        self.patience = 0
        self.max_patience = max_patience
        self.alpha = alpha

    def train(self, src, tgt, labels):
        """Trains the fish school"""
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(src, tgt)
        loss = CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        if self.complexity_regularization:
            loss += self.alpha * sum(p.pow(2.0).sum() for p in self.model.parameters())
        loss.backward()
        if self.dynamic_learning_rate:
            self.scheduler.step(loss)
        else:
            forward = self.optimizer
            forward.step()
        self.food = -loss.item()
        if self.early_stopping:
            if loss < self.best_food:
                self.best_food = loss
                self.patience = 0
            else:
                self.patience += 1

    def forward(self, src, tgt):
        """Forward pass of the fish school"""
        return self.model(src, tgt)

    def early_stopping(self, food):
        """
        Early stopping if the fish is not improving.
        """
        if food < self.best_food:
            self.best_food = food
            self.patience = 0
        else:
            self.patience += 1
            if self.patience > 5:
                return True
        return False

    def generate(self, src, tgt):
        """
        Generate a sequence using the fish's model.
        """
        return self.model.generate(src, tgt)

    def save(self, path):
        """
        Save the fish's model.
        """
        torch.save(self.model.state_dict(), path)


class FishSchool(nn.Module):
    """
    Fish School is a collection of fish.

    Parameters
    ----------
    num_fish : int
        The number of fish in the school.
    dim : int
        The number of expected features in the input (required).
    heads : int
        The number of heads in the multiheadattention models (required).
    depth : int
        The number of sub-encoder-layers in the encoder (required).
    num_iter : int
        The number of iterations to train the fish school.


    Usage:
    >>> school = FishSchool(10, 512, 8, 6, 100)
    >>> school.train(src, tgt, labels)
    >>> school.fish[0].food

    """

    def __init__(self, num_fish, dim, heads, depth, num_iter, num_top_fish=None, complex_school=False, max_seq_len=8192):
        super().__init__()
        self.fish = [Fish(dim, heads, depth) for _ in range(num_fish)]
        self.num_iter = num_iter
        self.num_top_fish = num_top_fish
        self.complex_school = complex_school
        self.max_seq_len = max_seq_len

    def forward(self, src, tgt, labels):
        for _ in range(self.num_iter):
            total_food = 0
            for fish in self.fish:
                fish.train(src, tgt, labels)
                total_food += fish.food
            avg_food = total_food / len(self.fish)
            if self.complex_school:
                for fish in self.fish:
                    neighbor = self.fish[torch.randint(0, len(self.fish), (1,)).item()]
                    if neighbor.food > fish.food:
                        fish.model.load_state_dict(neighbor.model.state_dict())
            for fish in self.fish:
                if fish.food < avg_food:
                    best_fish = max(self.fish, key=lambda f: f.food)
                    fish.model.load_state_dict(best_fish.model.state_dict())

    def generate(self, src, tgt):
        """
        Generate a sequence using the fish school's model.
        """
        return self.fish[0].generate(src, tgt)

    def predict(self, src, tgt):
        """
        Ensemble learning => enseble prediction of top performing models

        averages outputs of the top peforming models
        """
        top_fish = sorted(self.fish, key=lambda f: f.food, reverse=True)[:self.num_top_fish]
        self.model.eval()
        with torch.no_grad():
            outputs = torch.stack([fish.model(src, tgt) for fish in top_fish])
        return sum(outputs) / len(outputs)

    def save(self, path):
        """
        Save the fish school's models.
        """
        for i, fish in enumerate(self.fish):
            fish.save(path + f'fish_{i}.pt')

    def load(self, path):
        """
        Load the fish school's models.
        """
        for i, fish in enumerate(self.fish):
            fish.model.load_state_dict(torch.load(path + f'fish_{i}.pt'))

    def early_stopping(self):
        """
        Early stopping if the fish school is not improving.
        """
        for fish in self.fish:
            if fish.early_stopping(fish.food):
                return True
        return False

    def dynamic_learning_rate(self):
        """
        Dynamic learning rate for the fish school.
        """
        for fish in self.fish:
            fish.dynamic_learning_rate = True

    def complexity_regularization(self):
        """
        Complexity regularization for the fish school.
        """
        for fish in self.fish:
            fish.complexity_regularization = True

    def reset(self):
        """
        Reset the fish school's food.
        """
        for fish in self.fish:
            fish.food = 0

    def __getitem__(self, index):
        """Get the fish at the given index"""
        return self.fish[index]

    def __len__(self):
        """Get the number of fish in the school"""
        return len(self.fish)

    def __iter__(self):
        """Iterate over the fish in the school"""
        return iter(self.fish)

    def __next__(self):
        """Get the next fish in the school"""
        return next(self.fish)

    def __str__(self):
        """Get the string representation of the fish school"""
        return str(self.fish)


class GraphCellularAutomata(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphCellularAutomata, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.mlp(x)


class ReplicationModel(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(ReplicationModel, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x):
        return self.mlp(x)


class WeightUpdateModel(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(WeightUpdateModel, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.mlp(x)


class NDP(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(NDP, self).__init__()
        self.gc_automata = GraphCellularAutomata(embedding_dim, hidden_dim, embedding_dim)
        self.replication_model = ReplicationModel(embedding_dim, hidden_dim)
        self.weight_update_model = WeightUpdateModel(2 * embedding_dim, hidden_dim)

    def forward(self, node_embeddings, adjacency_matrix):
        updated_embeddings = self.gc_automata(node_embeddings)
        replication_decisions = self.replication_model(updated_embeddings)
        num_nodes = node_embeddings.shape[0]
        edge_weights = torch.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                combined_embedding = torch.cat((updated_embeddings[i], updated_embeddings[j]))
                edge_weights[i, j] = self.weight_update_model(combined_embedding)
        return updated_embeddings, replication_decisions, edge_weights


class HivemindTransformer(nn.Module):

    def __init__(self, dim: 'int'=None, max_seq_len: 'int'=None, depth: 'int'=None, heads: 'int'=None, dim_head: 'int'=None, num_tokens: 'int'=None):
        super(HivemindTransformer, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.num_tokens = num_tokens
        self.model = Transformer(num_tokens=num_tokens, max_seq_len=max_seq_len, attn_layers=Decoder(dim=dim, depth=depth, dim_head=dim_head, heads=heads))

    def forward(self, x):
        return self.model(x)


class HivemindSwarm(nn.Module):
    """
    HiveMind Swarm Transformer

    This is a transformer that is composed of a swarm of transformers where each transformer shares the same weights.

    Args:
        dim: dimension of the model
        max_seq_len: maximum sequence length
        depth: depth of the model
        heads: number of heads
        dim_head: dimension of each head
        num_models: number of models in the swarm
        base_transformer: the base transformer to be used in the swarm


    Example::
    model = HivemindSwarm(
        dim=512,
        max_seq_len=1024,
        depth=6,
        heads=8,
        dim_head=64,
        num_models=4,
    )

    x = torch.randn(1, 1024, 512)
    y = model(x)
    print(y.shape)


    """

    def __init__(self, dim: 'int'=None, max_seq_len: 'int'=None, num_tokens: 'int'=None, depth: 'int'=None, heads: 'int'=None, dim_head: 'int'=None, num_models: 'int'=1, **kwargs):
        super(HivemindSwarm, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.heads = heads
        self.num_tokens = num_tokens
        self.dim_head = dim_head
        self.num_models = num_models
        self.base_transformer = HivemindTransformer(dim=dim, num_tokens=num_tokens, max_seq_len=max_seq_len, depth=depth, heads=heads, dim_head=dim_head)
        self.experts = nn.ModuleList([self.base_transformer for _ in range(num_models)])
        self.gate = nn.Linear(num_models, num_models)
        self.gate_activation = nn.Softmax(dim=-1)
        self.gate_bias = nn.Parameter(torch.zeros(num_models))

    def forward(self, x):
        logits = []
        for expert in self.experts:
            output = expert(x)
            logits.append(output)
        outputs = torch.stack(logits, dim=1)
        gate = self.gate_activation(self.gate_bias + self.gate(outputs))
        outputs = torch.sum(outputs * gate.unsqueeze(-1), dim=1)
        return outputs


class Agent(nn.Module):

    def __init__(self):
        super(Agent, self).__init__()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        raise NotImplementedError

    def update_parameters(self, shared_gradients: 'Dict[str, torch.Tensor]') ->None:
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.grad is not None:
                    param.grad = shared_gradients[name]
        self.optimizer.step()
        self.optimizer.zero_grad()


class MLPAgent(Agent):

    def __init__(self, input_size: 'int', hidden_size: 'int', output_size: 'int'):
        super(MLPAgent, self).__init__()
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))
        self
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        logger.debug(f'MLPAgent input shape: {x.shape}')
        output = self.model(x)
        logger.debug(f'MLPAgent output shape: {output.shape}')
        return output


class CNNAgent(Agent):

    def __init__(self, input_channels: 'int', num_classes: 'int'):
        super(CNNAgent, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(input_channels, 16, kernel_size=3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(16 * 28 * 28, num_classes))
        self
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        logger.debug(f'CNNAgent input shape: {x.shape}')
        output = self.model(x)
        logger.debug(f'CNNAgent output shape: {output.shape}')
        return output


class LSTMAgent(Agent):

    def __init__(self, input_size: 'int', hidden_size: 'int', output_size: 'int'):
        super(LSTMAgent, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        logger.debug(f'LSTMAgent input shape: {x.shape}')
        x = x.view(x.size(0), x.size(2), -1)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        logger.debug(f'LSTMAgent output shape: {output.shape}')
        return output


class TransformerAgent(Agent):

    def __init__(self, input_size: 'int', num_heads: 'int', num_layers: 'int', output_size: 'int'):
        super(TransformerAgent, self).__init__()
        self.embedding = nn.Linear(input_size, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(128, output_size)
        self
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        logger.debug(f'TransformerAgent input shape: {x.shape}')
        x = x.view(x.size(0), x.size(2), -1)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        transformer_out = self.transformer_encoder(x)
        transformer_out = transformer_out.permute(1, 0, 2)
        output = self.fc(transformer_out[:, -1, :])
        logger.debug(f'TransformerAgent output shape: {output.shape}')
        return output


class MultiArchitectureSwarm(nn.Module):

    def __init__(self, num_mlp_agents: 'int', num_cnn_agents: 'int', num_lstm_agents: 'int', num_transformer_agents: 'int', input_sizes: 'Dict[str, Any]', output_size: 'int'):
        super(MultiArchitectureSwarm, self).__init__()
        self.agents: 'List[Agent]' = []
        for _ in range(num_mlp_agents):
            agent = MLPAgent(input_size=input_sizes['mlp']['input_size'], hidden_size=input_sizes['mlp']['hidden_size'], output_size=output_size)
            self.agents.append(agent)
        for _ in range(num_cnn_agents):
            agent = CNNAgent(input_channels=input_sizes['cnn']['input_channels'], num_classes=output_size)
            self.agents.append(agent)
        for _ in range(num_lstm_agents):
            agent = LSTMAgent(input_size=input_sizes['lstm']['input_size'], hidden_size=input_sizes['lstm']['hidden_size'], output_size=output_size)
            self.agents.append(agent)
        for _ in range(num_transformer_agents):
            agent = TransformerAgent(input_size=input_sizes['transformer']['input_size'], num_heads=input_sizes['transformer']['num_heads'], num_layers=input_sizes['transformer']['num_layers'], output_size=output_size)
            self.agents.append(agent)
        logger.info(f'Initialized {len(self.agents)} agents.')

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        agent_outputs = []
        for agent in self.agents:
            agent_output = agent(x)
            agent_outputs.append(agent_output)
        global_output = self.aggregate_agent_outputs(agent_outputs)
        return global_output

    def aggregate_agent_outputs(self, agent_outputs: 'List[torch.Tensor]') ->torch.Tensor:
        logger.debug(f'Aggregating outputs from {len(agent_outputs)} agents.')
        stacked_outputs = torch.stack(agent_outputs)
        logger.debug(f'Stacked outputs shape: {stacked_outputs.shape}')
        global_output = torch.mean(stacked_outputs, dim=0)
        logger.debug(f'Global output shape: {global_output.shape}')
        return global_output


class MixtureOfMambas(nn.Module):
    """
    Mixtures of Mamba is a swarm of Mamba models. The swarm can be aggregated
    using a weighted average or a simple average. We plan to add more aggregation
    methods in the future like a gating mechanism or a neural network or a
    transformer.

    Args:
        num_mambas (int): _description_
        dim (int): _description_
        d_state (int): _description_
        d_conv (_type_): _description_
        expand (int): _description_
        fusion_method (str, optional): _description_. Defaults to "average".

    Example::
    >>> model = MixtureOfMambas(
    ...     num_mambas=2,
    ...     dim=512,
    ...     d_state=1024,
    ...     depth=4,
    ...     d_conv=1024,
    ...     expand=4,
    ...     fusion_method="average",
    ... )
    >>> x = torch.rand(1, 512, 512)
    >>> model(x).shape
    torch.Size([1, 512, 512])
    """

    def __init__(self, num_mambas: 'int', dim: 'int', d_state: 'int', depth: 'int', d_conv, expand: 'int', fusion_method: 'str'='average', custom_fusion_func: 'callable'=None, *args, **kwargs):
        super(MixtureOfMambas, self).__init__()
        self.num_mambas = num_mambas
        self.dim = dim
        self.d_state = d_state
        self.depth = depth
        self.d_conv = d_conv
        self.expand = expand
        self.fusion_method = fusion_method
        self.custom_fusion_func = custom_fusion_func
        self.models = nn.ModuleList()
        for _ in range(num_mambas):
            mamba_model = MambaBlock(dim, depth, d_state, expand, d_conv, *args, **kwargs)
            self.models.append(mamba_model)

    def forward(self, x: 'torch.Tensor', weights=None):
        """Forward pass of the swarm

        Args:
            x (torch.Tensor): _description_
            weights (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        outputs = [model(x) for model in self.models]
        if self.fusion_method == 'average':
            return self.average_aggregate(outputs)
        elif self.fusion_method == 'weighted':
            return self.weighted_aggregate(outputs, weights)
        elif self.fusion_method == 'absmax':
            return self.absmax_aggregate(outputs, weights)
        elif self.fusion_method == 'softmax':
            return self.softmax_aggregate(outputs, weights)
        elif self.fusion_method == 'custom':
            if self.custom_fusion_func is None:
                raise ValueError('custom_fusion_func must be provided if fusion_method is custom')
            return self.custom_fusion_func(outputs, weights)
        else:
            raise ValueError(f'Unknown aggregation method: {self.fusion_method}')

    def average_aggregate(self, outputs):
        """Average the outputs of the models in the swarm

        Args:
            outputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        return torch.mean(torch.stack(outputs), dim=0)

    def weighted_aggegrate(self, outputs, weights):
        """Weighted average the outputs of the models in the swarm

        Args:
            outputs (_type_): _description_
            weights (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if weights is None or len(weights) != len(outputs):
            raise ValueError('Weights must be the same length as outputs')
        weighted_outputs = [(weight * output) for weight, output in zip(weights, outputs)]
        return sum(weighted_outputs)

    def softmax_aggregate(self, outputs, weights):
        """Weighted average the outputs of the models in the swarm

        Args:
            outputs (_type_): _description_
            weights (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if weights:
            weighted_outputs = [(weight * output) for weight, output in zip(weights, outputs)]
            out = sum(weighted_outputs)
            out = torch.softmax(out, dim=1)
        else:
            out = torch.softmax(outputs, dim=1)
        return out

    def absmax(self, outputs):
        """Absolute maximum of the outputs of the models in the swarm

        Args:
            outputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        return torch.max(torch.abs(torch.stack(outputs)), dim=0)[0]

    def absmax_aggregate(self, outputs, weights=None):
        """
        Weighted average the outputs of the models in the swarm

        Args:
            outputs (_type_): _description_
            weights (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if weights:
            weighted_outputs = [(weight * output) for weight, output in zip(weights, outputs)]
            return self.absmax(weighted_outputs)
        else:
            return self.absmax(outputs)


class TransformerLayer(nn.Module):
    """
    Transformer Layer

    Architecture:
        - Input -> Linear -> ReLU -> Linear -> ReLU -> Output

    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension

    Returns:
        torch.Tensor: Output tensor

    Usage

    """

    def __init__(self, input_dim, output_dim, nhead: 'int'):
        super(TransformerLayer, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(self.transformer(x))


class Neuron(nn.Module):

    def __init__(self, num_states):
        super(Neuron, self).__init__()
        self.states = nn.Parameter(torch.randn(num_states))


class SynapseTransformer(nn.Module):

    def __init__(self, input_dim, output_dim, nhead: 'int'):
        super(SynapseTransformer, self).__init__()
        self.transformer = TransformerLayer(input_dim, output_dim, nhead)

    def forward(self, x):
        return self.transformer(x)


class NNTransformer(nn.Module):
    """
    Neural Network NNTransformer

    Args:
        neuron_count (int): Number of neurons
        num_states (int): Number of states
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        nhead (int): Number of heads in transformer layer

    Returns:
        torch.Tensor: Output tensor

    Architecture:
        - Input -> Linear -> ReLU -> Linear -> ReLU -> Output
        - Neuron states are updated after each synapse
        - Softmax is applied after each synapse
        - Layer normalization is applied after each synapse

    Usage:
        network = CellularNN(5, 10, 10, 10, 2)
        output = network(torch.randn(1, 10))
        print(output)


    Training:
    network = NNTransformer(5, 10, 10, 10, 2)
    output = network(torch.randn(1, 10))
    print(output)


    # Test the network
    import torch
    import torch.optim as optim
    import torch.nn.functional as F

    # Random dataset
    batch_size = 64
    input_size = 10
    output_size = 10

    x = torch.randn(batch_size, input_size)  # Random inputs
    y = torch.randn(batch_size, output_size)  # Random targets

    # Hyperparameters
    neuron_count = 5
    num_states = 10
    input_dim = input_size
    output_dim = output_size
    n_head = 2

    # Initialize the network
    network = CellularNN(neuron_count, num_states, input_dim, output_dim, n_head)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        # Forward pass
        outputs = network(x)

        # Compute loss
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 100 epochs
        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Test the network with a new random input
    test_input = torch.randn(1, input_size)
    test_output = network(test_input)
    print(test_output)


    """

    def __init__(self, neuron_count, num_states, input_dim, output_dim, nhead):
        super(NNTransformer, self).__init__()
        self.neurons = nn.ModuleList([Neuron(num_states) for _ in range(neuron_count)])
        self.synapses = nn.ModuleList([SynapseTransformer(input_dim, output_dim, nhead) for _ in range(neuron_count)])
        self.norm = nn.LayerNorm(output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for neuron, synapse in zip(self.neurons[:-1], self.synapses):
            x = self.norm(x)
            x = synapse(x)
            x = self.softmax(x)
            neuron.states.data = x
        return self.neurons[-1].states


class Particle(nn.Module):
    """
    Simple Transformer model for classification.

    Parameters
    ----------
    input_dim : int
        The number of expected features in the input (required).
    d_model : int
        The number of expected features in the encoder/decoder inputs (required).
    nhead : int
        The number of heads in the multiheadattention models (required).
    num_layers : int
        The number of sub-encoder-layers in the encoder (required).
    output_dim : int
        The number of classes to predict (required).

    Usage:
    >>> model = SimpleTransformer(1000, 512, 8, 6, 10)
    >>> model(x)


    """

    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(Particle, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        Forward pass through the model.

        """
        x = self.embedding(x)
        x = self.transformer(x, x)
        return self.fc(x[-1])


class TransformerParticleSwarmOptimization(nn.Module):
    """
    Transformer Particle Swarm Optimization.

    Parameters
    ----------
    model_constructor : function
        Function to create a new model instance.
    model_args : tuple
        Arguments for the model constructor.
    device : str
        'cuda' or 'cpu'.
    criterion : nn.Module
        Loss function.
    data_loader : torch.utils.data.DataLoader
        Data loader.
    n_particles : int
        Number of particles.
    inertia : float
        Inertia weight.
    personal_best_weight : float
        Personal best weight.
    global_best_weight : float
        Global best weight.

    Usage:
    >>> pso = TransformerParticleSwarmOptimization(
    ...     SimpleTransformer,
    ...     (1000, 512, 8, 6, 10),
    ...     device="cuda",
    ...     criterion=nn.CrossEntropyLoss(),
    ...     data_loader=your_dataloader
    ... )

    """

    def __init__(self, model_constructor, model_args, device, criterion, data_loader, n_particles=10, inertia=0.5, personal_best_weight=1.5, global_best_weight=1.5):
        super(TransformerParticleSwarmOptimization, self).__init__()
        self.model_constructor = model_constructor
        self.model_args = model_args
        self.criterion = criterion
        self.data_loader = data_loader
        self.device = device
        self.n_particles = n_particles
        self.inertia = inertia
        self.personal_best_weight = personal_best_weight
        self.global_best_weight = global_best_weight
        param_size = sum(p.numel() for p in model_constructor(*model_args).parameters())
        self.particles = [self.model_constructor(*model_args) for _ in range(n_particles)]
        self.velocities = [torch.zeros((param_size,)) for _ in range(n_particles)]
        self.personal_best = [deepcopy(p.state_dict()) for p in self.particles]
        self.global_best = deepcopy(self.particles[0].state_dict())

    def compute_fitness(self, model_state):
        """
        Compute the fitness of a model.
        """
        model = self.model_constructor(*self.model_args)
        model.load_state_dict(model_state)
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.data_loader:
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return 1.0 / (1.0 + total_loss)

    def update(self):
        """
        Update particles.
        """
        for idx, particle in enumerate(self.particles):
            fitness = self.compute_fitness(particle.state_dict())
            if fitness > self.compute_fitness(self.personal_best[idx]):
                self.personal_best[idx] = deepcopy(particle.state_dict())
            if fitness > self.compute_fitness(self.global_best):
                self.global_best = deepcopy(particle.state_dict())
            for name, param in particle.named_parameters():
                delta = self.personal_best_weight * torch.rand_like(param) * (self.personal_best[idx][name] - param.data) + self.global_best_weight * torch.rand_like(param) * (self.global_best[name] - param.data)
                self.velocities[idx] += self.inertia * self.velocities[idx] + delta
                param.data += self.velocities[idx]

    def optimize(self, iterations=1000):
        """Optimize the model."""
        for _ in range(iterations):
            self.update()
            best_particle_score = self.compute_fitness(self.global_best)
            None

    def get_best_model(self):
        """Get the best model."""
        best_model = self.model_constructor(*self.model_args)
        best_model.load_state_dict(self.global_best)
        return best_model


class QueenBeeGa(nn.Module):
    """
    Queen Bee evolution for genetic algos

    Inspired by the evolution of bees, the fittest solution is designated
    and the rest of the population contends to mate with it.

    The strong exploitation is balanced by a higher than a normal mutation rate.

    Reference:
    ---------
    https://www.researchgate.net/publication/228961729_A_Queen_Bee_GA_for_optimization

    Usage
    -----
    optimizer = QueenBeeGa(
        goal="Attention is all you need",
        pop_size=100,
        mutation_prob=0.04,
        strong_mutation_rate=0.1,
        strong_mutation_prob=0.25,
        num_tournament_participants=25
    )
    optimizer.run(max_generations=100)
    """

    def __init__(self, goal: 'str'='Attention is all you need', pop_size: 'int'=100, mutation_prob: 'float'=0.04, strong_mutation_rate: 'float'=0.1, strong_mutation_prob: 'float'=0.25, num_tournament_participants: 'int'=25):
        """
        QueenBeeGa with params and initial configs

        Parameters
        ----------
        goal: str
            The goal string to be optimized
        pop_size: int
            Number of ants
        mutation_prob: float
            Mutation rate
        strong_mutation_rate: float
            Strong mutation rate
        strong_mutation_prob: float
            Strong mutation probability
        num_tournament_participants: int
            Number of tournament participants
        """
        self.goal = goal
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.strong_mutation_rate = strong_mutation_rate
        self.strong_mutation_prob = strong_mutation_prob
        self.num_tournament_participants = num_tournament_participants
        self.gene_length = len(goal)
        self.gene_midpoint = self.gene_length // 2
        self.target_gene = self.encode(goal)
        self.strong_mutate_pool_size = strong_mutation_rate * pop_size
        self.num_code_mutate = mutation_prob * self.gene_length
        self.strong_num_code_mutate = strong_mutation_prob * self.gene_length
        self.pool = torch.randint(0, 255, (pop_size, self.gene_length))
        self.queen = None
        self.queen_fitness = None
        self.generation = 0

    @staticmethod
    def encode(s):
        """Convert string to it's values"""
        return torch.tensor([ord(c) for c in s])

    @staticmethod
    def decode(t):
        """Convert ASCII values tensor back to string"""
        return ''.join([chr(i) for i in t.tolist()])

    def run(self, max_generations: 'int'=1000):
        """
        Run the queen genertic algorithm evolution

        Parameters:
        -----------
        max_generations: int
            Maximum number of generations
        """
        for _ in range(max_generations):
            self.generation += 1
            None
            self._evolve()
            if self._check_convergence():
                pass

    def _evolve(self):
        """
        Execute one step of the evolution process.
        """
        fitnesses = 1.0 / torch.square(self.pool - self.target_gene).sum(dim=-1)
        indices = fitnesses.sort(descending=True).indices
        self.pool, fitnesses = self.pool[indices], fitnesses[indices]
        if self.queen is not None:
            None
            None
        for gene, fitness in zip(self.pool, fitnesses):
            None
        if self.queen is not None and self.queen_fitness < fitnesses[0]:
            self.pool = torch.cat((self.pool, self.queen[None, :]), dim=0)
            fitnesses = torch.cat((fitnesses, self.queen_fitness[None]), dim=0)
            self.queen = self.queen_fitness = None
        if self.queen is None:
            self.queen, self.pool = self.pool[0], self.pool[1:]
            self.queen_fitness, fitnesses = fitnesses[0], fitnesses[1:]
        contender_ids = torch.randn((self.pop_size - 1, self.pop_size - 1)).argsort(dim=-1)[..., :self.num_tournament_participants]
        participants, tournaments = self.pool[contender_ids], fitnesses[contender_ids]
        top_winner = tournaments.topk(1, dim=-1, largest=True, sorted=False).indices
        top_winner = top_winner.unsqueeze(-1).expand(-1, -1, self.gene_length)
        parents = participants.gather(1, top_winner).squeeze(1)
        queen_parents = self.queen.unsqueeze(0).expand(self.pop_size - 1, self.gene_length)
        self.pool = torch.cat((queen_parents[:, :self.gene_midpoint], parents[:, self.gene_midpoint:]), dim=-1)
        mutate_mask = torch.randn(self.pool.shape).argsort(dim=-1) < self.num_code_mutate
        noise = torch.randint(0, 2, self.pool.shape) * 2 - 1
        mutated_pool = torch.where(mutate_mask, self.pool + noise, self.pool)
        strong_mutate_mask = torch.randn(self.pool.shape).argsort(dim=-1) < self.strong_num_code_mutate
        noise = torch.randint(0, 2, self.pool.shape) * 2 - 1
        strong_mutated_pool = torch.where(strong_mutate_mask, self.pool + noise, self.pool)
        strong_mutate_pool_mask = torch.randn(self.pop_size - 1).argsort(dim=-1) < self.strong_mutate_pool_size
        self.pool = torch.where(strong_mutate_pool_mask[:, None], strong_mutated_pool, mutated_pool)
        self.pool.clamp_(0, 255)

    def _check_convergence(self):
        """
        Check if any of the solutions has achieved the goal
        """
        fitnesses = 1.0 / torch.square(self.pool - self.target_gene).sum(dim=-1)
        return (fitnesses == float('inf')).any().item()


class SPO(nn.Module):
    """
    Spiral Optimization (SPO) Algorithm in PyTorch.

    Implements the SPO algorithm for optimization towards a target string.

    How does it work?
    ----------
    1. Initialize the search points randomly
    2. Initialize the center randomly
    3. Update the search points based on the spiral model
    4. Find the best search point and set as the new center
    5. Repeat step 3 to 4 until the maximum number of iterations is reached

    Usage
    -----
    from swarms_torch import SPO

    goaling = "Hello SPO"
    spo = SPO(goaling, m=100, k_max=1000)
    spo.optimize()

    print("Best Matched String:", spo.best_string())

    Future Features to implement
    --------
    1. Add a stopping criterion
    2. Add a callback function to track the progress
    3. Add a function to plot the search points
    4. Add a function to plot the best solution

    """

    def __init__(self, goal: 'str'=None, m: 'int'=10, k_max: 'int'=1000):
        """
        Initialize the SPO class.

        Args:
        - goal: The target string.
        - m: Number of search points (strings).
        - k_max: Maximum number of iterations.
        """
        self.goal = torch.tensor([ord(c) for c in goal], dtype=torch.float32)
        self.m = m
        self.k_max = k_max
        self.n_dim = len(goal)
        self.points = torch.randint(32, 127, (self.m, self.n_dim), dtype=torch.float32)
        self.center = torch.randint(32, 127, (self.n_dim,), dtype=torch.float32)

    def _step_rate(self, k):
        """
        Define the step rate function.

        Args:
        - k: Current iteration.

        Returns: Step rate for the current iteration.
        """
        return 1 / (1 + k)

    def _update_points(self, k):
        """Update the search points based on the spiral model."""
        r = self._step_rate(k)
        R = torch.eye(self.n_dim)
        for i in range(self.m):
            self.points[i] = self.center + r * torch.mv(R, self.points[i] - self.center)

    def _update_center(self):
        """Find the best search point and set as the new center."""
        fitnesses = torch.norm(self.points - self.goal, dim=1)
        best_idx = torch.argmin(fitnesses)
        self.center = self.points[best_idx]

    def optimize(self):
        """Run the optimization loop."""
        for k in range(self.k_max):
            self._update_points(k)
            self._update_center()
            if torch.norm(self.center - self.goal) < 1e-05:
                break

    def best_string(self):
        """Convert the best found point to its string representation"""
        return ''.join([chr(int(c)) for c in self.center.round()])


class SwiGLU(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class TopKGate(nn.Module):

    def __init__(self, model_dim, num_experts, top_k):
        super(TopKGate, self).__init__()
        self.w_gate = nn.Linear(model_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        gate_logits = self.w_gate(x)
        top_logits, top_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_logits = torch.full_like(gate_logits, float('-inf'))
        top_k_logits.scatter_(1, top_indices, top_logits)
        return F.softmax(top_k_logits, dim=-1)


class MoE(nn.Module):

    def __init__(self, model_dim, hidden_dim, num_experts, top_k):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([SwiGLU(model_dim, hidden_dim, model_dim) for _ in range(num_experts)])
        self.gate = TopKGate(model_dim, num_experts, top_k)

    def forward(self, x):
        gate_scores = self.gate(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        weighted_expert_outputs = gate_scores.unsqueeze(-1) * expert_outputs
        return weighted_expert_outputs.sum(dim=2)


class ParallelSwarm(nn.Module):

    def __init__(self, models: 'List[nn.Module]'):
        """
        Initializes a parallel swarm of models.

        Args:
            models (List[nn.Module]): A list of PyTorch models.

        """
        super().__init__()
        self.models = models
        for model in models:
            self.model = model

    def forward(self, x: 'torch.Tensor', *args, **kwargs):
        """Forward pass of the swarm

        Args:
            x (torch.Tensor): _description_
            weights (_type_, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        outputs = []
        for model in self.models:
            outputs.append(model(x, *args, **kwargs))
        return outputs


class GatingMechanism(nn.Module):

    def __init__(self, dim: 'int', num_experts: 'int'):
        """
        GatingMechanism is a class that represents the gating mechanism in a mixture of experts model.

        Args:
            dim (int): The input dimension.
            num_experts (int): The number of experts in the mixture.

        """
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x: 'Tensor'):
        """
        Forward pass of the gating mechanism.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the gating mechanism.

        """
        return F.softmax(self.gate(x), dim=-1)


class SimpleMoE(nn.Module):
    """
    Simple Mixture of Experts (MoE) model.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward network.
        output_dim (int): Output dimension.
        num_experts (int): Number of experts in the MoE.
        mult (int, optional): Multiplier for the hidden dimension. Defaults to 4.
    """

    def __init__(self, dim, hidden_dim: 'int', output_dim: 'int', num_experts: 'int', mult: 'int'=4):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.mult = mult
        self.experts = nn.ModuleList([FeedForward(dim, dim, mult) for _ in range(num_experts)])
        self.gate = GatingMechanism(dim, num_experts)

    def forward(self, x: 'Tensor'):
        """
        Forward pass of the SimpleMoE model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, output_dim).
        """
        gating_scores = self.gate(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        output = torch.sum(gating_scores.unsqueeze(2) * expert_outputs, dim=-1)
        return output


class SwitchGate(nn.Module):
    """
    SwitchGate module for MoE (Mixture of Experts) model.

    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts.
        capacity_factor (float, optional): Capacity factor for sparsity. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, dim, num_experts: 'int', capacity_factor: 'float'=1.0, epsilon: 'float'=1e-06, *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: 'Tensor', use_aux_loss=False):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Gate scores.
        """
        gate_scores = F.softmax(self.w_gate(x), dim=-1)
        capacity = int(self.capacity_factor * x.size(0))
        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)
        mask = torch.zeros_like(gate_scores).scatter_(1, top_k_indices, 1)
        masked_gate_scores = gate_scores * mask
        denominators = masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        gate_scores = masked_gate_scores / denominators * capacity
        if use_aux_loss:
            load = gate_scores.sum(0)
            importance = gate_scores.sum(1)
            loss = ((load - importance) ** 2).mean()
            return gate_scores, loss
        return gate_scores, None


class SwitchMoE(nn.Module):
    """
    A module that implements the Switched Mixture of Experts (MoE) architecture.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
        mult (int, optional): The multiplier for the hidden dimension of the feedforward network. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float): The capacity factor that controls the capacity of the MoE.
        mult (int): The multiplier for the hidden dimension of the feedforward network.
        experts (nn.ModuleList): The list of feedforward networks representing the experts.
        gate (SwitchGate): The switch gate module.

    """

    def __init__(self, dim: 'int', hidden_dim: 'int', output_dim: 'int', num_experts: 'int', capacity_factor: 'float'=1.0, mult: 'int'=4, use_aux_loss: 'bool'=False, *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult
        self.use_aux_loss = use_aux_loss
        self.experts = nn.ModuleList([FeedForward(dim, dim, mult, *args, **kwargs) for _ in range(num_experts)])
        self.gate = SwitchGate(dim, num_experts, capacity_factor)

    def forward(self, x: 'Tensor'):
        """
        Forward pass of the SwitchMoE module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor of the MoE.

        """
        gate_scores, loss = self.gate(x, use_aux_loss=self.use_aux_loss)
        expert_outputs = [expert(x) for expert in self.experts]
        if torch.isnan(gate_scores).any():
            None
            gate_scores[torch.isnan(gate_scores)] = 0
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-1)
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[torch.isnan(stacked_expert_outputs)] = 0
        moe_output = torch.sum(gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1)
        return moe_output, loss


class SwarmalatorModel(nn.Module):
    """
    # Example
    N = 100  # number of swarmalators
    D = 3  # dimensions

    model = SwarmalatorModel(N, D)
    positions, orientations = model()
    print(positions, orientations)
    """

    def __init__(self, N, D, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super(SwarmalatorModel, self).__init__()
        self.N = N
        self.D = D
        self.positions = nn.Parameter(torch.randn(N, D))
        self.orientations = nn.Parameter(torch.randn(N, D))
        encoder_layer = nn.TransformerEncoderLayer(d_model=D, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=D, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, src_mask=None, tgt_mask=None, memory_mask=None):
        position_memory = self.transformer_encoder(self.positions.unsqueeze(1), mask=src_mask)
        orientation_memory = self.transformer_encoder(self.orientations.unsqueeze(1), mask=src_mask)
        updated_positions = self.transformer_decoder(self.positions.unsqueeze(1), position_memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        updated_orientations = self.transformer_decoder(self.orientations.unsqueeze(1), orientation_memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return updated_positions.squeeze(1), updated_orientations.squeeze(1)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Fish,
     lambda: ([], {'dim': 4, 'heads': 4, 'depth': 1}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (GatingMechanism,
     lambda: ([], {'dim': 4, 'num_experts': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GraphCellularAutomata,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MoE,
     lambda: ([], {'model_dim': 4, 'hidden_dim': 4, 'num_experts': 4, 'top_k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NDP,
     lambda: ([], {'embedding_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (NNTransformer,
     lambda: ([], {'neuron_count': 4, 'num_states': 4, 'input_dim': 4, 'output_dim': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ReplicationModel,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwiGLU,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwitchGate,
     lambda: ([], {'dim': 4, 'num_experts': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SynapseTransformer,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (TopKGate,
     lambda: ([], {'model_dim': 4, 'num_experts': 4, 'top_k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerLayer,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (WeightUpdateModel,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

