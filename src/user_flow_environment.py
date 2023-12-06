from collections import defaultdict
import random

from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import networkx as nx
import numpy as np
import pandas as pd


class UserFlowEnvironment(Env):
    def __init__(self, G, initial_state, action_map=None):
        self.G = G
        self.initial_state = initial_state
        self.action_map = {i: action for i, action in enumerate(action_map)} if action_map else {}
        
        self.action_space = Discrete(num_actions) if (num_actions := len(self.action_map)) > 0 else None
        self.observation_space = Box(0, G.number_of_nodes() - 1, shape=(1,), dtype="int")
        
        self.last_action = None
        self.history = defaultdict(list)
        self._state = initial_state
        
    @property
    def states(self):
        return self.G.nodes()
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, new_state):
        self.mark_state_as_visited(self.G, self.state)
        self._state = new_state
        self.history[self.last_action].append(new_state)
    
    def _get_obs(self):
        return np.array([self.state])

    def _get_info(self):
        return self.history

    def reset(self):
        self._state = self.initial_state
        self.history[self.last_action].append(self.initial_state)
        nx.set_node_attributes(self.G, False, "visited")
        self.move_until_action_is_required()
        return self._get_obs(), self._get_info()

    def step(self, action_code):
        action = self.action_map.get(action_code, None)
        if not action:
            raise ValueError(f"Action code {action_code} does not correspond to any action")
        
        self.last_action = action
        
        original_state = self.state
        reward = self.move_to_next_state(action)
        reward_automatic = self.move_until_action_is_required()
        truncated = original_state == self.state
        terminal = self.G.nodes[self.state].get("terminal")
        
        return self._get_obs(), reward + reward_automatic, terminal, truncated, self._get_info()

    def state_was_visited(self, state):
        return self._state_was_visited(self.G, state)
    
    def render(self):
        return draw_network(self.G)

    @staticmethod
    def _state_was_visited(G, state):
        return G.nodes[state].get("visited", False)

    def move_until_action_is_required(self):
        if self._state_was_visited(self.G, self.state):
            return 0
        
        total_reward = 0
        previous_state = None
        while previous_state != self.state:
            previous_state = self.state
            reward = self.move_to_next_state("Automatic")
            total_reward += reward

        return total_reward
    
    def move_to_next_state(self, action):
        next_state, reward = self.get_next_state(self.G, self.state, action)
        if next_state:
            self.state = next_state
        return reward

    @classmethod
    def get_next_state(cls, G, current_state, action):
        action_edges = [
            (destination, attributes.get("weight"))
            for origin, destination, attributes in G.edges(current_state, data=True)
            if attributes.get("action") == action and not cls._state_was_visited(G, destination)
        ]

        if len(action_edges) == 0:
            return None, 0
        
        nodes, weights = zip(*action_edges)

        next_state = cls.random_choice(nodes, weights)
        reward = G.edges[current_state, next_state].get("reward", 0)
        
        return next_state, reward
    
    @staticmethod
    def random_choice(options, weights):
        return random.choices(options, weights, k=1)[0]
    
    @staticmethod
    def mark_state_as_visited(G, state):
        nx.set_node_attributes(G, {state: {"visited": True}})
        
        
def draw_network(G):
    pos = nx.circular_layout(G)
    node_weight = pd.Series({node: data.get("weight", 1) for node, data in G.nodes(data=True)})
    node_terminal = pd.Series({node: data.get("terminal", False) for node, data in G.nodes(data=True)})
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=["aqua" if terminal else "indigo" for terminal in node_terminal],
        node_size=4 * node_weight ** 0.5
    )
    nx.draw_networkx_labels(G, pos)
    weights = np.array([np.log10(data["weight"]) for node_a, node_b, data in G.edges(data=True)])
    is_action_edges = [data.get("action", "Automatic") != "Automatic" for node_a, node_b, data in G.edges(data=True)]
    edges = nx.draw_networkx_edges(
        G,
        pos,
        G.edges(),
        arrowsize=10,
        arrows=True,
        connectionstyle="arc3,rad=0.1",
        edge_color=["fuchsia" if is_action else "gold" for is_action in is_action_edges],
        edge_vmin=min(weights),
        edge_vmax=max(weights),
        width=weights + 1
    )