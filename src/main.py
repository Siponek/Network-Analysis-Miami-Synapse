# # Network analsys for SIR model

# ## Libraries

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from PIL import Image
import shutil

# from tqdm.notebook import tqdm
from tqdm import tqdm
from typing import Any

# ## Init
#

SEED: int = 2137
np.random.seed(SEED)
random.seed(SEED)

DATA_PATH: Path = Path("data/bn-mouse-kasthuri_graph_v4.edges")


def setup_graph():
    print(f"Using dataset from {DATA_PATH.absolute()}")
    G = nx.read_edgelist(path=DATA_PATH, create_using=nx.Graph(), nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.k_core(G, k=2)
    print("nodes:", len(G.nodes()), "and edges:", len(G.edges()))
    pos = nx.spring_layout(G, seed=SEED)

    infected_time_init: int = 0
    nx.set_node_attributes(G, "S", "state")
    nx.set_node_attributes(G, infected_time_init, "infection_cooldown")
    nx.set_node_attributes(G, 0, "recovery_step")
    nx.set_node_attributes(G, 0, "infection_step")

    return G, pos


# ## Visualization function


class VizAgent:

    def __init__(
        self,
        snapshot_path: Path = Path("out/snapshots"),
        frames_out_path: Path = Path("out/frames"),
        gif_out_path: Path = Path("out/gifs"),
        plot_out_path: Path = Path("out/plots"),
    ) -> None:
        self._snapshot_path = snapshot_path
        self._frames_out_path = frames_out_path
        self._gif_out_path = gif_out_path
        self._plot_out_path = plot_out_path

        self._snapshot_path.mkdir(exist_ok=True, parents=True)
        self._frames_out_path.mkdir(exist_ok=True, parents=True)
        self._gif_out_path.mkdir(exist_ok=True, parents=True)
        self._plot_out_path.mkdir(exist_ok=True, parents=True)

    @property
    def snapshot_path(self):
        return self._snapshot_path

    @snapshot_path.setter
    def snapshot_path(self, snapshot_path: Path):
        self._snapshot_path = snapshot_path

    @property
    def frames_out_path(self):
        return self._frames_out_path

    @frames_out_path.setter
    def frames_out_path(self, frames_out_path: Path):
        self._frames_out_path = frames_out_path

    @property
    def gif_out_path(self):
        return self._gif_out_path

    @gif_out_path.setter
    def gif_out_path(self, gif_out_path: Path):
        self._gif_out_path = gif_out_path

    @property
    def plot_out_path(self):
        return self._plot_out_path

    @plot_out_path.setter
    def plot_out_path(self, plot_out_path: Path):
        self._plot_out_path = plot_out_path

    def draw_graph(self, G, pos, step, experiment_num, snapshot_mode=True, params=None):
        """
        Draw the SIR model graph with optional parameter display and save to file.

        Args:
        G (networkx.Graph): The graph to draw.
        pos (dict): The position dictionary for nodes.
        step (int): The current simulation step.
        experiment_num (str): Identifier for the experiment.
        path_to_frames (Path): Path where frames are stored.
        snapshot_mode (bool): Whether to save snapshot mode images.
        params (dict, optional): Simulation parameters to display in the title.
        """
        plt.figure(figsize=(8, 6))
        state_color = {"S": "blue", "I": "red", "R": "green"}
        colors = [state_color[G.nodes[n]["state"]] for n in G.nodes()]
        nx.draw_networkx(
            G,
            pos,
            node_color=colors,
            node_size=25,
            width=0.4,
            with_labels=False,
            edge_color=(0, 0, 0, 0.5),
        )
        nx.draw
        if params:
            title_text = f'SIR at step {step} (p={params["p"]}, tI={params["tI"]}, q={params["q"]})'
            number_of_nodes = len(G.nodes())
            handles = []
            for s, c in state_color.items():
                percentage = (
                    len([n for n in G.nodes() if G.nodes[n]["state"] == s])
                    / number_of_nodes
                )
                patch = mpatches.Patch(color=c, label=f"{s} ({percentage:.3f})")
                handles.append(patch)
            plt.legend(handles=handles)
        else:
            title_text = f"SIR at step {step}"
            plt.legend(
                handles=[
                    mpatches.Patch(color=c, label=s) for s, c in state_color.items()
                ],
                loc="upper right",
            )

        plt.title(title_text)

        output_path = self.frames_out_path / f"{experiment_num}-frame_{step}.png"
        plt.savefig(output_path)
        if snapshot_mode:
            snapshot_path = self.snapshot_path / f"{experiment_num}-snapshot_{step}.png"
            plt.savefig(snapshot_path)
        plt.close()

    def plot_sir_data(self, results: dict):

        for exp_id, result in results.items():
            history = result["history"]
            parameters = result["params"]
            S, I, R = (
                [h["S"] for h in history],
                [h["I"] for h in history],
                [h["R"] for h in history],
            )
            susceptible_patch = mpatches.Patch(
                color="blue", label=f'Susceptible (p={parameters["p"]})'
            )
            infected_patch = mpatches.Patch(
                color="red", label=f'Infected (tI={parameters["tI"]})'
            )
            recovered_patch = mpatches.Patch(
                color="green", label=f'Recovered (q={parameters["q"]})'
            )
            # Create a new figure for each experiment
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(S, label="Susceptible", color="blue")
            ax.plot(I, label="Infected", color="red")
            ax.plot(R, label="Recovered", color="green")

            ax.legend(handles=[susceptible_patch, infected_patch, recovered_patch])
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Number of Nodes")
            ax.set_title(
                f'SIR - {exp_id} p = {parameters["p"]} tI= {parameters["tI"]} q = {parameters["q"]}'
            )
            ax.grid(True)
            plt.savefig(
                self.plot_out_path
                / f'{exp_id}-{parameters["p"]}-{parameters["tI"]}-{parameters["q"]}.png'
            )
            plt.close()

    def create_gif(
        self,
        experiment_id,
        gif_name,
        duration=250,
    ):
        """
        Creates a GIF for a specific experiment.

        Args:
        experiment_id (str): Identifier for the experiment to create GIF.
        gif_name (str): The name of the output GIF file.
        duration (int): The duration each frame appears in the GIF (in milliseconds).
        path_to_frames (Path): Path to the directory containing frame images.
        """
        frames = []
        # Filter and sort frame files based on experiment_id and step number
        frame_files = sorted(
            self.frames_out_path.glob(f"{experiment_id}-frame_*.png"),
            key=lambda x: int(x.stem.split("_")[1].split(".")[0]),
        )

        for frame_path in frame_files:
            frames.append(Image.open(frame_path))

        if frames:
            # Save the frames as a GIF
            frames[0].save(
                self.gif_out_path / f"{gif_name}.gif",
                format="GIF",
                append_images=frames[1:],
                save_all=True,
                duration=duration,
                loop=0,
            )
            print(f"GIF created: {self.gif_out_path / gif_name}.gif")
        else:
            print(f"No frames found for the specified {experiment_id} experiment.")


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

    def infect_community(self, initial_infections_per_community):
        infected_nodes = []
        communities = nx.algorithms.community.louvain_communities(self.curr_net_state)
        if communities is None:
            raise ValueError("No communities found in the graph.")
        try:

            for community in communities:  # type: ignore
                if len(community) > initial_infections_per_community:
                    infected_nodes.extend(
                        random.sample(community, initial_infections_per_community)
                    )
                else:
                    infected_nodes.extend(community)
            return infected_nodes
        except Exception as e:
            print(f"Error: {e}")
            return infected_nodes

    def run_experiments(
        self,
        G,
        pos,
        experiments,
        steps=500,
        snapshot_interval=50,
        target_communities=False,
    ):
        results = {}
        experiment_num = 1
        self.curr_net_state = G

        for exp_id, params in experiments.items():
            self._current_sim_name = exp_id
            # Reset the graph to all susceptible
            nx.set_node_attributes(self.curr_net_state, "S", "state")

            if target_communities:
                infected_nodes = self.infect_community(
                    initial_infections_per_community=params["i0"]
                )
            else:
                infected_nodes = np.random.choice(
                    a=list(G.nodes()), size=params["i0"], replace=False
                )

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


# ## Random infection simulation

# Define different experiments with varying parameters
experiments = {
    # ? p- Disease Transmission Probability
    # ? tI - Duration of Infection
    # ? q - Recovery Probability
    # ? i0- Initial Number of Infected Individuals
    "exp1": {"p": 0.05, "tI": 3, "q": 0.1, "i0": 5},
    "exp2": {"p": 0.1, "tI": 3, "q": 0.1, "i0": 5},
    "exp3": {"p": 0.15, "tI": 3, "q": 0.1, "i0": 5},
    "exp4": {"p": 0.2, "tI": 3, "q": 0.1, "i0": 5},
    "exp5": {"p": 0.25, "tI": 3, "q": 0.1, "i0": 5},
}
(
    G,
    pos,
) = setup_graph()
Viz_Agent: VizAgent = VizAgent(
    frames_out_path=Path("out/random_pick/frames"),
    gif_out_path=Path("out/random_pick/gifs"),
    plot_out_path=Path("out/random_pick/plots"),
    snapshot_path=Path("out/random_pick/snapshots"),
)
SIR_simulator: SIR = SIR(viz_agent=Viz_Agent)


results = SIR_simulator.run_experiments(
    G=G, pos=pos, experiments=experiments, target_communities=False
)
Viz_Agent.plot_sir_data(
    results,
)

# Create the GIF
for idx, _ in enumerate(results):
    Viz_Agent.create_gif(
        experiment_id=idx + 1,
        gif_name=f"sir_simulation-{idx}",
        duration=150,
    )

shutil.rmtree(path=Path("out/random_pick/frames"))

# ## Community target infection simulation

G, pos = setup_graph()
Viz_Agent: VizAgent = VizAgent(
    frames_out_path=Path("out/target_communities/frames"),
    gif_out_path=Path("out/target_communities/gifs"),
    plot_out_path=Path("out/target_communities/plots"),
    snapshot_path=Path("out/target_communities/snapshots"),
)
SIR_simulator.viz_agent = Viz_Agent
results = SIR_simulator.run_experiments(
    G=G, pos=pos, experiments=experiments, target_communities=True
)

Viz_Agent.plot_sir_data(
    results,
)

# Create the GIF
for idx, _ in enumerate(results):
    Viz_Agent.create_gif(
        experiment_id=idx + 1,
        gif_name=f"sir_simulation-{idx}",
        duration=150,
    )

shutil.rmtree(path=Path("out/target_communities/frames"))
