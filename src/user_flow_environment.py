import random

from gymnasium.spaces import Box, Discrete
import networkx as nx
import numpy as np


class Environment:
    def __init__(self, G, initial_state, action_map=None):
        self.G = G
        self.initial_state = initial_state
        self.action_map = action_map or {}
        
        self.action_space = Discrete(num_actions) if (num_actions := len(self.action_map)) > 0 else None
        self.observation_space = Box(0, G.number_of_nodes() - 1, shape=(1,), dtype="int")
        
    def _get_obs(self):
        return np.array([self.state])

    def _get_info(self):
        return {
            "state": self.state,
            "visited_states": ["state_1"]
        }

    def reset(self):
        nx.set_node_attributes(self.G, False, "visited")
        self.state, _ = self.move_until_action_is_required(self.G, self.initial_state)
        return self._get_obs(), self._get_info()

    def step(self, action_code):
        action = self.action_map.get(action_code, None)
        if not action:
            raise ValueError(f"Action code {action_code} does not correspond to any action")
            
        next_state, reward = self.get_next_state(self.G, self.state, action)
        next_state, reward_automatic = self.move_until_action_is_required(self.G, next_state)
        truncated = next_state == self.state
        self.state = next_state
        terminal = self.G.nodes[self.state].get("terminal")
        
        return self._get_obs(), reward + reward_automatic, terminal, truncated, None

    def state_was_visited(self, state):
        return self._state_was_visited(self.G, state)

    @staticmethod
    def _state_was_visited(G, state):
        return G.nodes[state].get("visited", False)

    @classmethod
    def move_until_action_is_required(cls, G, start_state):
        if cls._state_was_visited(G, start_state):
            return start_state, 0
        
        total_reward = 0
        next_state = start_state
        current_state = None
        while next_state != current_state:
            current_state = next_state
            next_state, reward = cls.get_next_state(G, current_state, "Automatic")
            total_reward += reward
            cls.mark_state_as_visited(G, current_state)

        return next_state, total_reward

    @classmethod
    def get_next_state(cls, G, current_state, action):
        action_edges = [
            (destination, attributes.get("weight"))
            for origin, destination, attributes in G.edges(current_state, data=True)
            if attributes.get("action") == action and not cls._state_was_visited(G, destination)
        ]

        if len(action_edges) == 0:
            return current_state, 0
        
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