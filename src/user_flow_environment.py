from collections import defaultdict, namedtuple
import random
from typing import Callable, Dict

from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class UserFlowEnvironment(Env):
    HistoryStep = namedtuple("HistoryStep", ["action", "steps"])
    
    def __init__(
        self,
        G,
        initial_state,
        action_map=None,
        additional_states=None,
        conditional_probability_matrices=None,
        truncate_if_transition_not_possible=True,
        truncation_reward=0,
        render_mode="human"
    ):
        self.G = G
        self.initial_state = initial_state
        self.action_map = {i: action for i, action in enumerate(action_map)} if action_map else {}
        self.conditional_probability_matrices = conditional_probability_matrices or {}
        self.initial_additional_states = additional_states or {}
        self.truncate_if_transition_not_possible = truncate_if_transition_not_possible
        self.truncation_reward = truncation_reward
        
        self.action_space = Discrete(num_actions) if (num_actions := len(self.action_map)) > 0 else Discrete(1)
        
        num_additional_states_values = {
            state_name: conditional_probability_matrix.shape[1]
            for state_name, conditional_probability_matrix in self.conditional_probability_matrices.items()
        }
        
        self.observation_space = Dict({
            "step": Box(0, G.number_of_nodes() - 1, shape=(1,), dtype="int"),
            **{
                state_name: Discrete(conditional_probability_matrix.shape[1])
                for state_name, conditional_probability_matrix in self.conditional_probability_matrices.items()
            }
        })
        
        self.last_action = None
        self.history = defaultdict(list)
        self._state = initial_state
        self.render_mode = render_mode
        
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
        self._add_step_to_last_action_history(new_state)
        self.update_additional_state_callback()
        
    def update_additional_state_callback(self):
        pass
    
    def _add_step_to_last_action_history(self, step):
        self.history[-1].steps.append(step)
        
    def _add_action_to_action_history(self, action):
        self.history.append(self.HistoryStep(action, []))

    @staticmethod
    def _create_additional_states(additional_states_info: Dict):
        return {
            state_name: value if not isinstance(value, Callable) else value()
            for state_name, value in additional_states_info.items()
        }
    
    def _get_obs(self):
        state_index = list(self.G.nodes).index(self.state)
        additional_state_indices = {
            state_name: self.conditional_probability_matrices[state_name].columns.get_loc(self.additional_states[state_name])
            for state_name in self.conditional_probability_matrices.keys()
        }
        return {"step": np.array([state_index]), **additional_state_indices}

    def _get_info(self):
        return {"history": self.history}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._state = self.initial_state
        self.additional_states = self._create_additional_states(self.initial_additional_states)
        self.history = []
        self._add_action_to_action_history(None)
        self._add_step_to_last_action_history(self.initial_state)
        nx.set_node_attributes(self.G, False, "visited")
        nx.set_node_attributes(self.G, [], "taken_actions")
        nx.set_edge_attributes(self.G, False, "taken")
        self.move_until_action_is_required()
        return self._get_obs(), self._get_info()

    def step(self, action_code):
        action = self.action_map.get(int(action_code), None)
        if not action:
            raise ValueError(f"Action code {action_code} does not correspond to any action")
        
        self._add_action_to_action_history(action)
        # import pdb; pdb.set_trace()
        original_state = self.state
        reward = self.move_to_next_state(action)
        reward_automatic = self.move_until_action_is_required()
        if original_state == self.state:
            reward += self.truncation_reward
            truncated = self.truncate_if_transition_not_possible
        else:
            truncated = False
        terminal = self.state_is_terminal(self.state)
        
        return self._get_obs(), reward + reward_automatic, terminal, truncated, self._get_info()

    def state_was_visited(self, state):
        return self._state_was_visited(self.G, state)
    
    def state_is_terminal(self, state):
        return self._state_is_terminal(self.G, state)
    
    @staticmethod
    def _state_is_terminal(G, state):
        return G.nodes[state].get("terminal", False)
    
    def render(self):
        return draw_network(self.G)

    @staticmethod
    def _state_was_visited(G, state):
        return G.nodes[state].get("visited", False)
    
    @staticmethod
    def _action_from_state_was_taken(G, state, action):
        return action in G.nodes[state]["taken_actions"]

    def move_until_action_is_required(self):
        total_reward = 0
        previous_state = None
        while previous_state != self.state:
            previous_state = self.state
            reward = self.move_to_next_state("Automatic")
            total_reward += reward

        return total_reward
    
    def move_to_next_state(self, action):
        next_state, reward = self.get_next_state(self.G, self.state, action, self.additional_states, self.conditional_probability_matrices)
        if next_state:
            self.state = next_state
        return reward

    @classmethod
    def get_next_state(cls, G, current_state, action, additional_states, conditional_probability_matrices):
        if cls._state_is_terminal(G, current_state):
            return None, 0
        
        if cls._action_from_state_was_taken(G, current_state, action):
            return None, 0
        
        action_edges = [
            (destination, attributes.get("weight"))
            for origin, destination, attributes in G.edges(current_state, data=True)
            if attributes.get("action") == action and not cls._state_was_visited(G, destination)
        ]

        if len(action_edges) == 0:
            return None, 0
        
        nodes, transition_weights = zip(*action_edges)
        weights = transition_weights
        
        for state_name in additional_states.keys():
            additional_weights = conditional_probability_matrices[state_name].loc[nodes, :][additional_states[state_name]]
            weights *= additional_weights

        if all(weight == 0 for weight in weights):
            return None, 0
        
        next_state = cls.random_choice(nodes, weights)
        reward = G.edges[current_state, next_state].get("reward", 0)
        
        cls._mark_action_from_state_as_taken(G, current_state, action)
        cls._mark_edge_as_taken(G, current_state, next_state, action)
        
        return next_state, reward
    
    @staticmethod
    def random_choice(options, weights):
        return random.choices(options, weights, k=1)[0]
    
    @staticmethod
    def mark_state_as_visited(G, state):
        nx.set_node_attributes(G, {state: {"visited": True}})
    
    @staticmethod
    def _mark_action_from_state_as_taken(G, state, action):
        taken_actions = G.nodes[state]["taken_actions"]
        nx.set_node_attributes(G, {state: {"taken_actions": taken_actions + [action]}})
    
    @staticmethod
    def _mark_edge_as_taken(G, current_state, next_state, action):
        nx.set_edge_attributes(G, {(current_state, next_state): {"taken": True}})
        
    @staticmethod
    def was_edge_taken(G, current_state, next_state):
        return G.edges[current_state, next_state].get("taken")

        
def draw_network(G):
    def get_node_color(node_data):
        if node_data.get("terminal", False):
            return "aqua"
        
        if node_data.get("visited", False):
            return "red"
        
        return "indigo"
    
    plt.figure()
    pos = nx.circular_layout(G)
    node_weight = pd.Series({node: data.get("weight", 1) for node, data in G.nodes(data=True)})
    node_terminal = pd.Series({node: data.get("terminal", False) for node, data in G.nodes(data=True)})
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[get_node_color(data) for _, data in G.nodes(data=True)],
        node_size=4 * node_weight ** 0.5
    )
    nx.draw_networkx_labels(G, pos)
    weights = np.array([np.log10(data["weight"]) for node_a, node_b, data in G.edges(data=True)])
    # is_action_edges = [data.get("action", "Automatic") != "Automatic" for node_a, node_b, data in G.edges(data=True)]
    edges_taken = [UserFlowEnvironment.was_edge_taken(G, node_a, node_b) for node_a, node_b in G.edges()]
    edges = nx.draw_networkx_edges(
        G,
        pos,
        G.edges(),
        arrowsize=10,
        arrows=True,
        connectionstyle="arc3,rad=0.1",
        edge_color=["fuchsia" if taken else "gold" for taken in edges_taken],
        width=weights + 1
    )