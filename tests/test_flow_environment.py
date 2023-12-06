import unittest
from unittest import mock

from gymnasium.spaces import Box, Discrete
import networkx as nx
import numpy as np

from src.user_flow_environment import Environment


class EnvironmentTests(unittest.TestCase):
    def test_one_state_environment_returns_state_when_resetting(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state"])
        environment = Environment(G, "state")
        # When
        environment.reset()
        # Then
        self.assertEqual(environment.state, "state")
        
    def test_two_states_with_no_edge_yields_first_state_when_resetting(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["initial_state", "other_state"])
        environment = Environment(G, "initial_state")
        # When
        environment.reset()
        # Then
        self.assertEqual(environment.state, "initial_state")
        
    def test_two_states_with_edge_but_no_automatic_action_yields_initial_state_when_resetting(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["initial_state", "other_state"])
        G.add_edges_from([("initial_state", "other_state")])
        nx.set_edge_attributes(
            G,
            {
                ("initial_state", "other_state"): {"action": "Other"}
            }
        )
        environment = Environment(G, "initial_state")
        # When
        environment.reset()
        # Then
        self.assertEqual(environment.state, "initial_state")
        
    def test_two_states_with_edge_with_automatic_action_with_0_probability_of_change_yields_initial_state_when_resetting(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["initial_state", "other_state"])
        G.add_edges_from([("initial_state", "initial_state"), ("initial_state", "other_state")])
        nx.set_edge_attributes(
            G,
            {
                ("initial_state", "other_state"): {"action": "Automatic", "weight": 0},
                ("initial_state", "initial_state"): {"action": "Automatic", "weight": 1}
            }
        )
        environment = Environment(G, "initial_state")
        # When
        environment.reset()
        # Then
        self.assertEqual(environment.state, "initial_state")
        
    def test_two_states_with_edge_with_automatic_action_with_1_probability_of_change_yields_other_state_when_resetting(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["initial_state", "other_state"])
        G.add_edges_from([("initial_state", "other_state")])
        nx.set_edge_attributes(
            G,
            {
                ("initial_state", "other_state"): {"action": "Automatic", "weight": 1},
                ("initial_state", "initial_state"): {"action": "Automatic", "weight": 0}
            }
        )
        environment = Environment(G, "initial_state")
        # When
        environment.reset()
        # Then
        self.assertEqual(environment.state, "other_state")
        
    def test_chain_of_three_states_when_resetting(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([
            ("state_1", "state_2"),
            ("state_2", "state_3"),
        ])
        nx.set_edge_attributes(
            G,
            {
                ("state_1", "state_2"): {"action": "Automatic", "weight": 1},
                ("state_2", "state_3"): {"action": "Automatic", "weight": 1}
            }
        )
        environment = Environment(G, "state_1")
        # When
        environment.reset()
        # Then
        self.assertEqual(environment.state, "state_3")
        
    @mock.patch.object(Environment, "random_choice")
    def test_use_decision_method_to_move_between_states(self, random_choice_mock):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([
            ("state_1", "state_2"),
            ("state_1", "state_3"),
        ])
        nx.set_edge_attributes(
            G,
            {
                ("state_1", "state_2"): {"action": "Automatic", "weight": 1},
                ("state_1", "state_3"): {"action": "Automatic", "weight": 2}
            }
        )
        environment = Environment(G, "state_1")
        
        # Then
        random_choice_mock.return_value = "state_2"
        environment.reset()
        self.assertEqual(environment.state, "state_2")
        
        random_choice_mock.return_value = "state_3"
        environment.reset()
        self.assertEqual(environment.state, "state_3")
        
        random_choice_mock.assert_has_calls([
            mock.call(("state_2", "state_3"), (1, 2)),
            mock.call(("state_2", "state_3"), (1, 2))
        ])
        
    def test_do_not_yield_a_state_that_was_already_visited(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([
            ("state_1", "state_2"),
            ("state_2", "state_3"),
            ("state_3", "state_1"),
        ])
        nx.set_edge_attributes(
            G,
            {
                ("state_1", "state_2"): {"action": "Automatic", "weight": 1},
                ("state_2", "state_3"): {"action": "Automatic", "weight": 1},
                ("state_3", "state_1"): {"action": "Automatic", "weight": 1}
            }
        )
        environment = Environment(G, "state_1")
        # When
        environment.reset()
        # Then
        self.assertEqual(environment.state, "state_3")

    @mock.patch.object(Environment, "random_choice")
    def test_reset_visited_states(self, random_choice_mock):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([
            ("state_1", "state_2"),
            ("state_1", "state_3"),
        ])
        nx.set_edge_attributes(
            G,
            {
                ("state_1", "state_2"): {"action": "Automatic", "weight": 1},
                ("state_1", "state_3"): {"action": "Automatic", "weight": 2}
            }
        )
        environment = Environment(G, "state_1")
        
        # Then
        random_choice_mock.return_value = "state_2"
        environment.reset()
        self.assertTrue(environment.state_was_visited("state_1"))
        self.assertTrue(environment.state_was_visited("state_2"))
        self.assertFalse(environment.state_was_visited("state_3"))
        
        random_choice_mock.return_value = "state_3"
        environment.reset()
        self.assertTrue(environment.state_was_visited("state_1"))
        self.assertTrue(environment.state_was_visited("state_3"))
        self.assertFalse(environment.state_was_visited("state_2"))
        
    def test_choose_non_existing_action_raises_error(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1"])
        environment = Environment(
            G,
            "state_1",
        )
        environment.reset()
        # When
        self.assertRaises(
            ValueError,
            environment.step,
            "non-existing-action"
        )
        
    def test_choose_action_without_connected_states_has_no_effect(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1"])
        environment = Environment(
            G,
            "state_1",
            action_map={0: "action"}
        )
        environment.reset()
        # When
        environment.step(0)
        # Then
        self.assertEqual("state_1", environment.state)

    def test_choose_action_without_connected_states_has_no_effect_two_nodes(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2"])
        G.add_edges_from([("state_1", "state_2")])
        environment = Environment(
            G,
            "state_1",
            action_map={0: "action"}
        )
        environment.reset()
        # When
        environment.step(0)
        # Then
        self.assertEqual("state_1", environment.state)
        
    def test_take_action(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2"])
        G.add_edges_from([("state_1", "state_2")])
        nx.set_edge_attributes(
            G,
            {
                ("state_1", "state_2"): {"action": "action", "weight": 1},
            }
        )
        environment = Environment(
            G,
            "state_1",
            action_map={0: "action"}
        )
        environment.reset()
        # When
        environment.step(0)
        # Then
        self.assertEqual("state_2", environment.state)
        
    def test_take_action_after_automatic_changes(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([("state_1", "state_2"), ("state_2", "state_3")])
        nx.set_edge_attributes(
            G,
            {
                ("state_1", "state_2"): {"action": "Automatic", "weight": 1},
                ("state_2", "state_3"): {"action": "action", "weight": 1},
            }
        )
        environment = Environment(
            G,
            "state_1",
            action_map={0: "action"}
        )
        environment.reset()
        # When
        environment.step(0)
        # Then
        self.assertEqual("state_3", environment.state)
        
    def test_take_action_before_automatic_changes(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([("state_1", "state_2"), ("state_2", "state_3")])
        nx.set_edge_attributes(
            G,
            {
                ("state_1", "state_2"): {"action": "action", "weight": 1},
                ("state_2", "state_3"): {"action": "Automatic", "weight": 1},
            }
        )
        environment = Environment(
            G,
            "state_1",
            action_map={0: "action"}
        )
        environment.reset()
        # When
        environment.step(0)
        # Then
        self.assertEqual("state_3", environment.state)
        
    def test_take_action_after_and_before_automatic_changes(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3", "state_4"])
        G.add_edges_from([("state_1", "state_2"), ("state_2", "state_3"), ("state_3", "state_4")])
        nx.set_edge_attributes(
            G,
            {
                ("state_1", "state_2"): {"action": "Automatic", "weight": 1},
                ("state_2", "state_3"): {"action": "action", "weight": 1},
                ("state_3", "state_4"): {"action": "Automatic", "weight": 1},
            }
        )
        environment = Environment(
            G,
            "state_1",
            action_map={0: "action"}
        )
        environment.reset()
        # When
        environment.step(0)
        # Then
        self.assertEqual("state_4", environment.state)
        
    def test_take_action_does_not_change_state_if_destination_has_been_visited(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2"])
        G.add_edges_from([("state_1", "state_2"), ("state_2", "state_1")])
        nx.set_edge_attributes(
            G,
            {
                ("state_1", "state_2"): {"action": "Automatic", "weight": 1},
                ("state_2", "state_1"): {"action": "action", "weight": 1},
            }
        )
        environment = Environment(
            G,
            "state_1",
            action_map={0: "action"}
        )
        environment.reset()
        # When
        environment.step(0)
        # Then
        self.assertEqual("state_2", environment.state)
        
    def test_take_action_returns_truncated_no_action_is_possible(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2"])
        G.add_edges_from([("state_1", "state_2"), ("state_2", "state_1")])
        nx.set_edge_attributes(
            G,
            {
                ("state_1", "state_2"): {"action": "action", "weight": 1},
                ("state_2", "state_1"): {"action": "action", "weight": 1},
            }
        )
        environment = Environment(
            G,
            "state_1",
            action_map={0: "action"}
        )
        environment.reset()
        # When
        _, _, _, truncated_step_1, _ = environment.step(0)
        _, _, _, truncated_step_2, _ = environment.step(0)
        # Then
        self.assertFalse(truncated_step_1)
        self.assertTrue(truncated_step_2)

    def test_take_action_chooses_improbable_state_if_probable_destination_has_been_visited(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([("state_1", "state_2"), ("state_2", "state_1"), ("state_2", "state_3")])
        nx.set_edge_attributes(
            G,
            {
                ("state_1", "state_2"): {"action": "Automatic", "weight": 1},
                ("state_2", "state_1"): {"action": "action", "weight": 1000000},
                ("state_2", "state_3"): {"action": "action", "weight": 0.001},
            }
        )
        environment = Environment(
            G,
            "state_1",
            action_map={0: "action"}
        )
        environment.reset()
        # When
        environment.step(0)
        # Then
        self.assertEqual("state_3", environment.state)    

    def test_get_reward_from_step(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2"])
        G.add_edges_from([("state_1", "state_2")])
        nx.set_edge_attributes(
            G,
            {
                ("state_1", "state_2"): {"action": "SomeAction", "weight": 1, "reward": 10},
            }
        )
        environment = Environment(
            G,
            "state_1",
            action_map={0: "SomeAction"}
        )
        environment.reset()
        # When
        _, reward, *_ = environment.step(0)
        # Then
        self.assertEqual(10, reward)
        
    def test_get_reward_from_step_and_posterior_automatic_changes(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3", "state_4"])
        G.add_edges_from([
            ("state_1", "state_2", {"action": "SomeAction", "weight": 1, "reward": 10}),
            ("state_2", "state_3", {"action": "Automatic", "weight": 1, "reward": 20}),
            ("state_3", "state_4", {"action": "Automatic", "weight": 1, "reward": 15}),
        ])
        environment = Environment(
            G,
            "state_1",
            action_map={0: "SomeAction"}
        )
        environment.reset()
        # When
        state, reward, *_ = environment.step(0)
        # Then
        self.assertEqual("state_4", state)
        self.assertEqual(45, reward)
        
    def test_observation_space_is_one_integer_with_a_range_equal_to_the_number_of_nodes(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3", "state_4"])
        # When
        environment = Environment(
            G,
            "state_1",
            action_map={0: "SomeAction"}
        )
        # Then
        self.assertEqual(Box(0, 3, shape=(1,), dtype="int"), environment.observation_space)
        
    def test_action_space_is_integer_with_a_range_equal_to_the_number_of_actions(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3", "state_4"])
        # When
        environment = Environment(
            G,
            "state_1",
            action_map={0: "SomeAction", 1: "AnotherAction", 2: "Third Action"}
        )
        # Then
        self.assertEqual(Discrete(3), environment.action_space)
        
    def test_reset_method_returns_observations_and_info(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3", "state_4"])
        environment = Environment(
            G,
            "state_1",
            action_map={0: "SomeAction"}
        )
        # When
        observation, info = environment.reset()
        # Then
        self.assertTrue(np.array_equal(np.array(["state_1"]), observation))
        self.assertEqual(
            {
                "state": "state_1",
                "visited_states": ["state_1"]
            },
            info
        )
        
    def test_step_return_observation(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3", "state_4"])
        G.add_edges_from([
            ("state_1", "state_2", {"action": "SomeAction", "weight": 1, "reward": 10}),
            ("state_2", "state_3", {"action": "Automatic", "weight": 1, "reward": 20}),
            ("state_3", "state_4", {"action": "Automatic", "weight": 1, "reward": 15}),
        ])
        environment = Environment(
            G,
            "state_1",
            action_map={0: "SomeAction"}
        )
        environment.reset()
        # When
        observation, *_ = environment.step(0)
        # Then
        self.assertTrue(np.array_equal(np.array(["state_4"]), observation))
        
    def test_step_return_terminated_false_if_not_final_state(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3", "state_4"])
        G.add_edges_from([
            ("state_1", "state_2", {"action": "SomeAction", "weight": 1, "reward": 10}),
            ("state_2", "state_3", {"action": "Automatic", "weight": 1, "reward": 20}),
            ("state_3", "state_4", {"action": "Automatic", "weight": 1, "reward": 15}),
        ])
        environment = Environment(
            G,
            "state_1",
            action_map={0: "SomeAction"}
        )
        environment.reset()
        # When
        _, _, terminated, *_ = environment.step(0)
        # Then
        self.assertFalse(terminated)
        
    def test_step_returns_terminated_true_if_final_state_reached(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from([
            "state_1", "state_2", "state_3", ("state_4", dict(terminal=True))
        ])
        G.add_edges_from([
            ("state_1", "state_2", {"action": "SomeAction", "weight": 1, "reward": 10}),
            ("state_2", "state_3", {"action": "Automatic", "weight": 1, "reward": 20}),
            ("state_3", "state_4", {"action": "Automatic", "weight": 1, "reward": 15}),
        ])
        environment = Environment(
            G,
            "state_1",
            action_map={0: "SomeAction"}
        )
        environment.reset()
        # When
        _, _, terminated, *_ = environment.step(0)
        # Then
        self.assertTrue(terminated)
