import random

import networkx as nx


class Environment:
    def __init__(self, G, initial_state, action_map=None):
        self.G = G
        self.initial_state = initial_state
        self.action_map = action_map or {}

    def reset(self):
        nx.set_node_attributes(self.G, False, "visited")
        self.state = self.move_until_action_is_required(self.G, self.initial_state)

    def step(self, action_code):
        action = self.action_map.get(action_code, None)
        if not action:
            raise ValueError(f"Action code {action_code} does not correspond to any action")
            
        next_state = self.get_next_state(self.G, self.state, action)
        self.state = self.move_until_action_is_required(self.G, next_state)
        return self.state

    def state_was_visited(self, state):
        return self._state_was_visited(self.G, state)

    @staticmethod
    def _state_was_visited(G, state):
        return G.nodes[state].get("visited", False)

    @classmethod
    def move_until_action_is_required(cls, G, start_state):
        next_state = start_state
        current_state = None
        while next_state != current_state:
            current_state = next_state
            next_state = cls.get_next_state(G, current_state, "Automatic")
            cls.mark_state_as_visited(G, current_state)

        return next_state

    @classmethod
    def get_next_state(cls, G, current_state, action):
        action_edges = [
            (destination, attributes.get("weight"))
            for origin, destination, attributes in G.edges(current_state, data=True)
            if attributes.get("action") == action and not cls._state_was_visited(G, destination)
        ]

        if len(action_edges) == 0:
            return current_state
        
        nodes, weights = zip(*action_edges)

        return cls.random_choice(nodes, weights)
    
    @staticmethod
    def random_choice(options, weights):
        return random.choices(options, weights, k=1)[0]
    
    @staticmethod
    def mark_state_as_visited(G, state):
        nx.set_node_attributes(G, {state: {"visited": True}})