from .viz_agent import VizAgent
from tqdm import tqdm
import networkx as nx
import numpy as np
import random


# ## Progressing with infection


class SIR:
    # It should be protected from creating additional classes for SIR in the future
    def __init__(self, viz_agent: VizAgent) -> None:
        self.sim_results: dict = {}
        self._current_step: int = 0
        self._current_sim_name: str = "default"
        self._viz_agent = viz_agent
        # ? p- Disease Transmission Probability
        self.transmission_prob: float = 0.0
        # ? tI - Duration of Infection
        self.infection_duration: int = 0
        # ? q - Recovery Probability
        self.recovery_prob: float = 0.0

    @property
    def viz_agent(self):
        return self._viz_agent

    @viz_agent.setter
    def viz_agent(self, viz_agent: VizAgent):
        self._viz_agent = viz_agent

    def attempt_infection(
        self,
        node,
    ):
        """Attempts to infect neighbors of a node based on infection probability p."""
        new_states = {}
        for neighbor in self.curr_net_state.neighbors(node):
            if (
                self.curr_net_state.nodes[neighbor]["state"] == "S"
                and np.random.random() < self.transmission_prob
            ):
                new_states[neighbor] = {"state": "I"}
                new_states[neighbor]["infection_step"] = self._current_step
        return new_states

    def manage_recovery(self, node):
        """Determines whether an infected node recovers based on recovery probability q and minimum infection time tI."""
        should_recover = (
            np.random.random() < self.recovery_prob
            and self.curr_net_state.nodes[node]["infection_cooldown"]
            >= self.infection_duration
        )
        if should_recover:
            self.curr_net_state.nodes[node]["infection_cooldown"] = 0
            self.curr_net_state.nodes[node]["recovery_step"] = self._current_step
            return "R"
        else:
            self.curr_net_state.nodes[node]["infection_cooldown"] += 1
            return "I"

    def update_graph(
        self,
    ):
        new_state = {}
        for node in self.curr_net_state:
            current_state = self.curr_net_state.nodes[node]["state"]
            if current_state == "I":
                new_state.update(self.attempt_infection(node))
                new_state[node] = self.manage_recovery(node)
        # Update states
        for node, state in new_state.items():
            # G.nodes[node]['state'] = state
            if type(state) == dict:
                self.curr_net_state.nodes[node]["state"] = state.get("state", "error")
                self.curr_net_state.nodes[node]["infection_step"] = state.get(
                    "infection_step", -10
                )
            else:
                self.curr_net_state.nodes[node]["state"] = state
                self.curr_net_state.nodes[node]["infection_step"] = 0
            # G.nodes[node]['infection_step'] = state.get('infection_step', 0)

    def count_states(self) -> dict:
        states = {"S": 0, "I": 0, "R": 0}
        for node in self.curr_net_state.nodes:
            states[self.curr_net_state.nodes[node]["state"]] += 1
        return states

    def simulate_sir(
        self, pos, steps, snapshot_interval: int = 50, experiment_num: int = 1
    ):
        # To keep track of S, I, R counts
        history_states: list = []
        # To keep track of graphs for snapshots
        snapshots: list = []

        with tqdm(total=steps) as pbar:
            for step in range(steps):
                self._current_step = step
                pbar.set_description(
                    f"Running simulation {experiment_num} for {steps} steps"
                )
                self.update_graph()
                history_states.append(self.count_states())
                should_snapshot = (
                    step % snapshot_interval == 0
                    or step == steps - 1
                    or self.count_states()["I"] == 0
                )

                self._viz_agent.draw_graph(
                    G=self.curr_net_state,
                    pos=pos,
                    step=self._current_step,
                    experiment_num=experiment_num,
                    snapshot_mode=should_snapshot,
                    params={
                        "p": self.transmission_prob,
                        "tI": self.infection_duration,
                        "q": self.recovery_prob,
                    },
                )

                if should_snapshot:
                    snapshots.append(f"{experiment_num}-snapshot_{step}.png")
                pbar.update(1)
                # Check if any infected are left
                if self.count_states()["I"] == 0:
                    break
            pbar.close()
        return history_states, snapshots

    def infect_by_betweenness(self, initial_infections: int):
        infected_nodes = []
        centrality = nx.betweenness_centrality(self.curr_net_state)
        sorted_by_centrality = sorted(
            centrality.items(), key=lambda x: x[1], reverse=True
        )
        infected_nodes = [node for node, _ in sorted_by_centrality[:initial_infections]]

        return infected_nodes

    def infect_by_closeness(self, initial_infections: int):
        closeness = nx.closeness_centrality(self.curr_net_state)
        sorted_by_closeness = sorted(
            closeness.items(), key=lambda x: x[1], reverse=True
        )
        infected_nodes = [node for node, _ in sorted_by_closeness[:initial_infections]]
        return infected_nodes

    def infect_by_degree(self, initial_infections: int):
        degrees = dict(self.curr_net_state.degree())
        sorted_by_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        infected_nodes = [node for node, _ in sorted_by_degree[:initial_infections]]
        return infected_nodes

    def infect_by_eigenvector(self, initial_infections: int):
        eigenvector = nx.eigenvector_centrality(self.curr_net_state)
        sorted_by_eigenvector = sorted(
            eigenvector.items(), key=lambda x: x[1], reverse=True
        )
        infected_nodes = [
            node for node, _ in sorted_by_eigenvector[:initial_infections]
        ]
        return infected_nodes

    def infect_by_pagerank(self, initial_infections: int):
        pagerank = nx.pagerank(self.curr_net_state)
        sorted_by_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        infected_nodes = [node for node, _ in sorted_by_pagerank[:initial_infections]]
        return infected_nodes

    def run_experiments(
        self,
        G,
        pos,
        experiments,
        steps=500,
        snapshot_interval=50,
        strategy: str = "random",
    ):
        results = {}
        experiment_num = 1
        self.curr_net_state = G
        strategy_methods: dict = {
            "closeness": self.infect_by_closeness,
            "degree": self.infect_by_degree,
            "eigenvector": self.infect_by_eigenvector,
            "pagerank": self.infect_by_pagerank,
        }

        for exp_id, params in experiments.items():
            self._current_sim_name = exp_id
            # Reset the graph to all susceptible
            nx.set_node_attributes(self.curr_net_state, "S", "state")
            intiial_infections = params["i0"]
            if strategy in strategy_methods:
                infected_nodes = strategy_methods[strategy](intiial_infections)
            else:
                # Random infection as fallback
                infected_nodes = np.random.choice(
                    a=list(G.nodes()), size=params["i0"], replace=False
                )
            print(f"Initial infections: {len(infected_nodes)}")
            for node in infected_nodes:
                self.curr_net_state.nodes[node]["state"] = "I"
                self.curr_net_state.nodes[node]["infection_cooldown"] = 0
                self.curr_net_state.nodes[node]["infection_step"] = -1

            self.transmission_prob = params["p"]
            self.infection_duration = params["tI"]
            self.recovery_prob = params["q"]

            history, snapshots = self.simulate_sir(
                pos, steps, snapshot_interval, experiment_num=experiment_num
            )
            results[exp_id] = {
                "history": history,
                "snapshots": snapshots,
                "params": params,
            }
            print(f"Experiment {exp_id} {experiment_num}  done!")
            experiment_num += 1
        return results
