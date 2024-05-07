# # Network analsys for SIR model

# ## Libraries

import numpy as np
import random
from pathlib import Path
import shutil
import networkx as nx
from typing import Any
from math import sqrt


# ## Init
#

SEED: int = 2137
np.random.seed(SEED)
random.seed(SEED)
from classes import sir, viz_agent

DATA_PATH: Path = Path("data/bn-mouse-kasthuri_graph_v4.edges")


def setup_graph():
    print(f"Using dataset from {DATA_PATH.absolute()}")
    G = nx.read_edgelist(path=DATA_PATH, create_using=nx.Graph(), nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.k_core(G, k=2)
    print("nodes:", len(G.nodes()), "and edges:", len(G.edges()))
    pos = nx.spring_layout(G, seed=SEED, k=15 / sqrt(len(G.nodes())))
    infected_time_init: int = 0
    nx.set_node_attributes(G, "S", "state")
    nx.set_node_attributes(G, infected_time_init, "infection_cooldown")
    nx.set_node_attributes(G, 0, "recovery_step")
    nx.set_node_attributes(G, 0, "infection_step")

    return G, pos


def run_simulation(strategy, frame_dir, gif_dir, plot_dir, snapshot_dir):
    # Setup the VizAgent with paths based on the strategy
    Viz_Agent = viz_agent.VizAgent(
        frames_out_path=Path(frame_dir),
        gif_out_path=Path(gif_dir),
        plot_out_path=Path(plot_dir),
        snapshot_path=Path(snapshot_dir),
    )

    # Update the VizAgent in the SIR simulator
    SIR_simulator.viz_agent = Viz_Agent

    # Run the experiments
    results = SIR_simulator.run_experiments(
        G=G, pos=pos, experiments=experiments, strategy=strategy
    )

    # Plot the SIR data
    Viz_Agent.plot_sir_data(results)

    # Create GIFs for each experiment
    for idx in range(len(results)):
        Viz_Agent.create_gif(
            experiment_id=idx + 1,
            gif_name=f"sir_simulation-{idx + 1}-{strategy}",
            duration=150,
        )

    # Clean up frames directory
    shutil.rmtree(path=Path(frame_dir))

    return results


# ## Random infection simulation

# Define different experiments with varying parameters
experiments = {
    # ? p- Disease Transmission Probability
    # ? tI - Duration of Infection
    # ? q - Recovery Probability
    # ? i0- Initial Number of Infected Individuals
    "exp11": {"p": 0.1, "tI": 3, "q": 0.3, "i0": 5},
    "exp12": {"p": 0.2, "tI": 3, "q": 0.3, "i0": 5},
    "exp13": {"p": 0.3, "tI": 3, "q": 0.3, "i0": 5},
    "exp14": {"p": 0.4, "tI": 3, "q": 0.3, "i0": 5},
    "exp15": {"p": 0.5, "tI": 3, "q": 0.3, "i0": 5},
    "exp16": {"p": 0.5, "tI": 3, "q": 0.3, "i0": 5},
    #
    "exp21": {"p": 0.3, "tI": 3, "q": 0.1, "i0": 5},
    "exp22": {"p": 0.3, "tI": 3, "q": 0.2, "i0": 5},
    "exp23": {"p": 0.3, "tI": 3, "q": 0.3, "i0": 5},
    "exp24": {"p": 0.3, "tI": 3, "q": 0.4, "i0": 5},
    "exp25": {"p": 0.3, "tI": 3, "q": 0.5, "i0": 5},
    "exp26": {"p": 0.3, "tI": 3, "q": 0.7, "i0": 5},
    "exp27": {"p": 0.3, "tI": 3, "q": 0.9, "i0": 5},
}
(
    G,
    pos,
) = setup_graph()


Viz_Agent: viz_agent.VizAgent = viz_agent.VizAgent(
    plot_out_path=Path("out/random/plots"),
    frames_out_path=Path("out/random/frames"),
    gif_out_path=Path("out/random/gifs"),
    snapshot_path=Path("out/random/snapshots"),
)
SIR_simulator: sir.SIR = sir.SIR(viz_agent=Viz_Agent)

# Running simulations for different strategies
base_dir = "out"
strategies = ["random", "betweenness", "degree", "closeness"]
strategy_results = {}
for strategy in strategies:
    dir_path = f"{base_dir}/{strategy}"
    strategy_result = run_simulation(
        strategy=strategy,
        frame_dir=f"{dir_path}/frames",
        gif_dir=f"{dir_path}/gifs",
        plot_dir=f"{dir_path}/plots",
        snapshot_dir=f"{dir_path}/snapshots",
    )
    strategy_results[strategy] = strategy_result

print("Simulation completed")
Viz_Agent.plot_infection_comparison(strategy_results)
print("Plotting infected completed")
