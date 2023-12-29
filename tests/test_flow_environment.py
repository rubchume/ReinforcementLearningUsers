import unittest
from unittest import mock

from gymnasium.spaces import Box, Discrete
import networkx as nx
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env

from src.user_flow_environment import UserFlowEnvironment as Environment


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
    
    def test_environment_should_pass_native_check(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state"])
        environment = Environment(G, "state", action_map=["Action"])
        # When
        check_env(environment, warn=True)
    
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
        G.add_nodes_from(["state_1", "state_2", "state_3", "state_4"])
        G.add_edges_from([
            ("state_1", "state_2"),
            ("state_1", "state_3"),
            ("state_2", "state_4"),
            ("state_3", "state_4"),
        ])
        nx.set_edge_attributes(
            G,
            {
                ("state_1", "state_2"): {"action": "Automatic", "weight": 1},
                ("state_1", "state_3"): {"action": "Automatic", "weight": 2},
                ("state_2", "state_4"): {"action": "Automatic", "weight": 1},
                ("state_3", "state_4"): {"action": "Automatic", "weight": 1}
            }
        )
        environment = Environment(G, "state_1")
        
        # Then
        random_choice_mock.side_effect = ["state_2", "state_4"]
        environment.reset()
        self.assertTrue(environment.state_was_visited("state_1"))
        self.assertTrue(environment.state_was_visited("state_2"))
        self.assertFalse(environment.state_was_visited("state_3"))
        
        random_choice_mock.side_effect = ["state_3", "state_4"]
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
            action_map=["action"]
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
            action_map=["action"]
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
            action_map=["action"]
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
            action_map=["action"]
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
            action_map=["action"]
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
            action_map=["action"]
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
            action_map=["action"]
        )
        environment.reset()
        # When
        environment.step(0)
        # Then
        self.assertEqual("state_2", environment.state)
        
    def test_take_action_returns_truncated_if_no_action_is_possible(self):
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
            action_map=["action"]
        )
        environment.reset()
        # When
        _, _, _, truncated_step_1, _ = environment.step(0)
        _, _, _, truncated_step_2, _ = environment.step(0)
        # Then
        self.assertFalse(truncated_step_1)
        self.assertTrue(truncated_step_2)
        
    def test_take_action_does_nothing_if_no_action_is_possible(self):
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
            action_map=["action"],
            truncate_if_transition_not_possible=False,
        )
        environment.reset()
        # When
        observation_1, _, _, truncated_step_1, _ = environment.step(0)
        observation_2, _, _, truncated_step_2, _ = environment.step(0)
        # Then
        self.assertFalse(truncated_step_1)
        self.assertFalse(truncated_step_2)
        self.assertEqual(1, observation_1["step"])
        self.assertEqual(1, observation_2["step"])

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
            action_map=["action"]
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
            action_map=["SomeAction"]
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
            action_map=["SomeAction"]
        )
        environment.reset()
        # When
        state, reward, *_ = environment.step(0)
        # Then
        self.assertEqual(3, state["step"])
        self.assertEqual(45, reward)
        
    def test_observation_space_first_element_is_one_integer_with_a_range_equal_to_the_number_of_nodes(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3", "state_4"])
        # When
        environment = Environment(
            G,
            "state_1",
            action_map=["SomeAction"]
        )
        # Then
        self.assertEqual(Box(0, 3, shape=(1,), dtype="int"), environment.observation_space["step"])
        
    def test_action_space_is_integer_with_a_range_equal_to_the_number_of_actions(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3", "state_4"])
        # When
        environment = Environment(
            G,
            "state_1",
            action_map=["SomeAction", "AnotherAction", "Third Action"]
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
            action_map=["SomeAction"]
        )
        # When
        observation, info = environment.reset()
        # Then
        self.assertEqual(0, observation["step"])
        self.assertEqual(
            {"history": [(None, ["state_1"])]},
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
            action_map=["SomeAction"]
        )
        environment.reset()
        # When
        observation, *_ = environment.step(0)
        # Then
        self.assertEqual(3, observation["step"])
        
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
            action_map=["SomeAction"]
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
            action_map=["SomeAction"]
        )
        environment.reset()
        # When
        _, _, terminated, *_ = environment.step(0)
        # Then
        self.assertTrue(terminated)

    def test_choose_between_different_actions(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from([
            "state_1", "state_2", "state_3"
        ])
        G.add_edges_from([
            ("state_1", "state_2", {"action": "Action 1", "weight": 1}),
            ("state_1", "state_3", {"action": "Action 2", "weight": 1}),
        ])
        environment = Environment(
            G,
            "state_1",
            action_map=["Action 1", "Action 2"]
        )
        
        # Then
        environment.reset()
        environment.step(1)
        self.assertEqual("state_3", environment.state)
        
        # Then
        environment.reset()
        environment.step(0)
        self.assertEqual("state_2", environment.state)
        
    def test_return_visited_nodes_in_info(self):
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
            action_map=["SomeAction"]
        )
        environment.reset()
        # When
        _, _, _, _, info = environment.step(0)
        # Then
        self.assertEqual(
            info,
            {"history": [
                (None, ["state_1"]),
                ("SomeAction", ["state_2", "state_3", "state_4"])
            ]}
        )
    
    def test_move_towards_the_same_step_only_happens_once(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2"])
        G.add_edges_from([
            ("state_1", "state_1", {"action": "Automatic", "weight": 1}),
            ("state_1", "state_2", {"action": "Automatic", "weight": 0}),
        ])
        environment = Environment(
            G,
            "state_1",
            action_map=["SomeAction"]
        )
        obs, info = environment.reset()
        # Then
        self.assertEqual(
            info,
            {"history": [
                (None, ["state_1", "state_1"]),
            ]}
        )
        
    def test_illegal_action_returns_truncated(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([
            ("state_1", "state_1", {"action": "Automatic", "weight": 1}),
            ("state_1", "state_2", {"action": "Automatic", "weight": 0}),
            ("state_2", "state_3", {"action": "Action", "weight": 1}),
        ])
        environment = Environment(
            G,
            "state_1",
            action_map=["Action"]
        )
        environment.reset()
        # When
        _, _, _, truncated, info = environment.step(0)
        # Then
        self.assertEqual(
            info,
            {"history": [
                (None, ["state_1", "state_1"]),
                ("Action", [])
            ]}
        )
        self.assertTrue(truncated)
        
    def test_illegal_action_returns_negative_reward_and_truncates(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([
            ("state_1", "state_1", {"action": "Automatic", "weight": 1}),
            ("state_1", "state_2", {"action": "Automatic", "weight": 0}),
            ("state_2", "state_3", {"action": "Action", "weight": 1}),
        ])
        environment = Environment(
            G,
            "state_1",
            action_map=["Action"],
            truncation_reward=-3
        )
        environment.reset()
        # When
        _, reward, _, truncated, info = environment.step(0)
        # Then
        self.assertEqual(
            reward,
            -3
        )
        self.assertTrue(truncated)
        
    def test_illegal_action_does_nothing(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([
            ("state_1", "state_1", {"action": "Automatic", "weight": 1}),
            ("state_1", "state_2", {"action": "Automatic", "weight": 0}),
            ("state_2", "state_3", {"action": "Action", "weight": 1}),
        ])
        environment = Environment(
            G,
            "state_1",
            action_map=["Action"],
            truncate_if_transition_not_possible=False
        )
        environment.reset()
        # When
        _, _, _, truncated, info = environment.step(0)
        # Then
        self.assertEqual(
            info,
            {"history": [
                (None, ["state_1", "state_1"]),
                ("Action", []),
            ]}
        )
        self.assertFalse(truncated)
        
    def test_illegal_action_returns_negative_reward_and_does_not_truncate(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([
            ("state_1", "state_1", {"action": "Automatic", "weight": 1}),
            ("state_1", "state_2", {"action": "Automatic", "weight": 0}),
            ("state_2", "state_3", {"action": "Action", "weight": 1}),
        ])
        environment = Environment(
            G,
            "state_1",
            action_map=["Action"],
            truncate_if_transition_not_possible=False,
            truncation_reward=-3
        )
        environment.reset()
        # When
        _, reward, _, truncated, info = environment.step(0)
        # Then
        self.assertEqual(
            reward,
            -3
        )
        self.assertFalse(truncated)
        
    def test_reset_action_makes_all_states_not_visited(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3", "state_4"])
        G.add_edges_from([
            ("state_1", "state_2", {"action": "Action", "weight": 1}),
            ("state_2", "state_3", {"action": "Automatic", "weight": 1}),
            ("state_3", "state_4", {"action": "Automatic", "weight": 1}),
        ])
        environment = Environment(
            G,
            "state_1",
            action_map=["Action"]
        )
        environment.reset()
        environment.step(0)
        self.assertEqual("state_4", environment.state)
        environment.reset()
        self.assertEqual("state_1", environment.state)
        environment.step(0)
        self.assertEqual("state_4", environment.state)
        
    def test_reset_makes_new_visited_list(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3", "state_4"])
        G.add_edges_from([
            ("state_1", "state_2", {"action": "Action", "weight": 1}),
            ("state_2", "state_3", {"action": "Automatic", "weight": 1}),
            ("state_3", "state_4", {"action": "Automatic", "weight": 1}),
        ])
        environment = Environment(
            G,
            "state_1",
            action_map=["Action"]
        )
        environment.reset()
        environment.step(0)
        # When
        _, info = environment.reset()
        # Then
        self.assertEqual(info, {"history": [(None, ["state_1"])]})
        
    def test_arriving_to_a_terminal_state_stops_at_that_state(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", ("state_3", {"terminal": True}), "state_4"])
        G.add_edges_from([
            ("state_1", "state_2", {"action": "Action", "weight": 1}),
            ("state_2", "state_3", {"action": "Automatic", "weight": 1}),
            ("state_3", "state_4", {"action": "Automatic", "weight": 1}),
        ])
        environment = Environment(
            G,
            "state_1",
            action_map=["Action"]
        )
        environment.reset()
        # When
        state, _, terminated, *_ = environment.step(0)
        # Then
        self.assertEqual(state["step"], 2)
        self.assertTrue(terminated)
        
    def test_arriving_to_a_terminal_state_stops_at_that_state_even_if_no_action_was_taken(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", ("state_3", {"terminal": True}), "state_4"])
        G.add_edges_from([
            ("state_1", "state_2", {"action": "Automatic", "weight": 1}),
            ("state_2", "state_3", {"action": "Automatic", "weight": 1}),
            ("state_3", "state_4", {"action": "Automatic", "weight": 1}),
        ])
        environment = Environment(
            G,
            "state_1",
            action_map=["Action"]
        )
        environment.reset()
        # Then
        self.assertEqual(environment.state, "state_3")
        
    @mock.patch.object(Environment, "random_choice", side_effect=["state_2", "state_2", "state_3"])
    def test_do_not_trigger_automatic_action_if_state_stopped_at_itself_and_illegal_action_was_taken(self, _):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([
            ("state_1", "state_2", {"action": "Automatic", "weight": 1}),
            ("state_2", "state_2", {"action": "Automatic", "weight": 1}),
            ("state_2", "state_3", {"action": "Automatic", "weight": 1}),
        ])
        environment = Environment(
            G,
            "state_1",
            action_map=["Action"]
        )
        # When
        environment.reset()
        environment.step(0)
        # Then
        self.assertEqual(environment.state, "state_2")
        
    def test_get_observations_as_gymnasium_space(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([
            ("state_1", "state_2", {"action": "Automatic", "weight": 1}),
            ("state_2", "state_2", {"action": "Automatic", "weight": 1}),
        ])
        environment = Environment(
            G,
            "state_1",
        )
        # When
        obs, _ = environment.reset()
        # Then
        self.assertEqual(1, obs["step"])
        
    def test_use_additional_naive_bayes_matrix(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([
            ("state_1", "state_2", {"action": "Automatic", "weight": 1}),
            ("state_1", "state_3", {"action": "Automatic", "weight": 1000000}),
        ])
        conditional_probability_matrix = pd.DataFrame({
            "state_1": [0.5, 0.5, 0],
            "state_2": [0.3, 0.3, 0.4],
            "state_3": [0.3, 0.6, 0],
        }, index=["value1", "value2", "value3"]).T
        environment = Environment(
            G,
            "state_1",
            additional_states={"additional": "value3"},
            conditional_probability_matrices={"additional": conditional_probability_matrix}
        )
        # When
        obs, info = environment.reset()
        # Then
        self.assertEqual(info, {"history": [(None, ["state_1", "state_2"])]})
        self.assertEqual(obs["step"], 1)
        
    def test_additional_naive_bayes_matrix_makes_weights_0_and_behaves_as_if_no_action_is_possible(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([
            ("state_1", "state_2", {"action": "Automatic", "weight": 1}),
            ("state_1", "state_3", {"action": "Automatic", "weight": 1}),
        ])
        conditional_probability_matrix = pd.DataFrame({
            "state_1": [0.5, 0.5, 0],
            "state_2": [0.3, 0.3, 0],
            "state_3": [0.3, 0.6, 0],
        }, index=["value1", "value2", "value3"]).T
        environment = Environment(
            G,
            "state_1",
            additional_states={"additional_state": "value3"},
            conditional_probability_matrices={"additional_state": conditional_probability_matrix}
        )
        environment.reset()
        # When
        obs, info = environment.reset()
        # Then
        self.assertEqual(info, {"history": [(None, ["state_1"])]})
        self.assertEqual(obs["step"], 0)
        
    def test_update_additional_state_with_callback(self):
        # Given
        class EnvironmentWithCallback(Environment):
            def update_additional_state_callback(self):
                if (current_step := int(self.state.split("_")[-1])) > int(self.additional_states["additional_state"].split("_")[-1]):
                    self.additional_states["additional_state"] = f"current_step_{current_step}"
            
        
        G = nx.DiGraph()
        G.add_nodes_from(["accumulative_state_1", "accumulative_state_2", "accumulative_state_3"])
        G.add_edges_from([
            ("accumulative_state_1", "accumulative_state_2", {"action": "Automatic", "weight": 1}),
            ("accumulative_state_2", "accumulative_state_3", {"action": "Automatic", "weight": 1}),
        ])
        environment = EnvironmentWithCallback(
            G,
            "accumulative_state_1",
            additional_states={"additional_state": "current_step_0"},
            conditional_probability_matrices={"additional_state": pd.DataFrame(
                np.ones((3, 4)),
                index=["accumulative_state_1", "accumulative_state_2", "accumulative_state_3"],
                columns=["current_step_0", "current_step_1", "current_step_2", "current_step_3"]
            )}
        )
        # When
        obs, info = environment.reset()
        # Then
        self.assertEqual(info, {"history": [(None, ["accumulative_state_1", "accumulative_state_2", "accumulative_state_3"])]})
        self.assertEqual(obs["step"], 2)
        self.assertEqual(obs["additional_state"], 3)

    def test_additional_state_gets_updated_when_resetting(self):
        # Given
        class EnvironmentWithCallback(Environment):
            def update_additional_state_callback(self):
                if (current_step := int(self.state.split("_")[-1])) == int(self.additional_states["additional_state"].split("_")[-1]) + 1:
                    self.additional_states["additional_state"] = f"current_step_{current_step}"
                else:
                    raise RuntimeError("There was a discontinuity")
        
        G = nx.DiGraph()
        G.add_nodes_from(["accumulative_state_1", "accumulative_state_2", "accumulative_state_3"])
        G.add_edges_from([
            ("accumulative_state_1", "accumulative_state_2", {"action": "Automatic", "weight": 1}),
            ("accumulative_state_2", "accumulative_state_3", {"action": "Automatic", "weight": 1}),
        ])
        environment = EnvironmentWithCallback(
            G,
            "accumulative_state_1",
            additional_states={"additional_state": "current_step_1"},
            conditional_probability_matrices={"additional_state": pd.DataFrame(
                np.ones((3, 4)),
                index=["accumulative_state_1", "accumulative_state_2", "accumulative_state_3"],
                columns=["current_step_0", "current_step_1", "current_step_2", "current_step_3"]
            )}
        )
        # When
        obs, info = environment.reset()
        obs, info = environment.reset()
        
    def test_use_multiple_additional_states_with_naive_bayes_matrices(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3", "state_4"])
        G.add_edges_from([
            ("state_1", "state_2", {"action": "Automatic", "weight": 1}),
            ("state_1", "state_3", {"action": "Automatic", "weight": 1000000}),
            ("state_1", "state_4", {"action": "Automatic", "weight": 1}),
        ])
        conditional_probability_matrix_1 = pd.DataFrame({
            "state_1": [0.5, 0.5, 0],
            "state_2": [0.3, 0.6, 0],
            "state_3": [0.3, 0.3, 0.4],
            "state_4": [0.3, 0.3, 0.4],
        }, index=["A", "B", "C"]).T
        conditional_probability_matrix_2 = pd.DataFrame({
            "state_1": [0.5, 0.5, 0],
            "state_2": [0.3, 0.4, 0.3],
            "state_3": [0.3, 0, 0.6],
            "state_4": [0.3, 0.4, 0.3],
        }, index=["D", "E", "F"]).T
        environment = Environment(
            G,
            "state_1",
            additional_states={"additional_state_1": "C", "additional_state_2": "E"},
            conditional_probability_matrices={
                "additional_state_1": conditional_probability_matrix_1,
                "additional_state_2": conditional_probability_matrix_2
            }
        )
        # When
        obs, info = environment.reset()
        # Then
        self.assertEqual(info, {"history": [(None, ["state_1", "state_4"])]})
        self.assertEqual(obs["step"], 3)

    def test_random_inititalization_of_additional_state(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2", "state_3"])
        G.add_edges_from([
            ("state_1", "state_2", {"action": "Automatic", "weight": 1}),
            ("state_1", "state_3", {"action": "Automatic", "weight": 1000000}),
        ])
        conditional_probability_matrix = pd.DataFrame({
            "state_1": [0.5, 0.5, 0],
            "state_2": [0.3, 0.3, 0.4],
            "state_3": [0.3, 0.6, 0],
        }, index=["value1", "value2", "value3"]).T
        
        generate_state = mock.MagicMock(return_value="value3")
        
        environment = Environment(
            G,
            "state_1",
            additional_states={"additional": generate_state},
            conditional_probability_matrices={"additional": conditional_probability_matrix}
        )
        # When
        obs, info = environment.reset()
        # Then
        self.assertEqual(info, {"history": [(None, ["state_1", "state_2"])]})
        self.assertEqual(obs["step"], 1)
        
    def test_self_referencing_probability_should_not_be_influenced_by_additional_states(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(["state_1", "state_2"])
        G.add_edges_from([
            ("state_1", "state_1", {"action": "Automatic", "weight": 1}),
            ("state_1", "state_2", {"action": "Automatic", "weight": 1}),
        ])
        conditional_probability_matrix = pd.DataFrame({
            "state_1": [0.5, 0.5, 0],
            "state_2": [0.3, 0.3, 0.4],
        }, index=["value1", "value2", "value3"]).T
        environment = Environment(
            G,
            "state_1",
            additional_states={"additional": "value3"},
            conditional_probability_matrices={"additional": conditional_probability_matrix}
        )
        # When
        obs, info = environment.reset()
        # Then
        self.assertEqual(info, {"history": [(None, ["state_1", "state_1"])]})
        self.assertEqual(obs["step"], 0)