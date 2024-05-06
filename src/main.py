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

from tqdm import tqdm
from typing import Any

from ..lib import viz_agent, sir

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
Viz_Agent: viz_agent.VizAgent = viz_agent.VizAgent(
    frames_out_path=Path("out/random_pick/frames"),
    gif_out_path=Path("out/random_pick/gifs"),
    plot_out_path=Path("out/random_pick/plots"),
    snapshot_path=Path("out/random_pick/snapshots"),
)
SIR_simulator: sir.SIR = sir.SIR(viz_agent=Viz_Agent)


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
Viz_Agent = viz_agent.VizAgent(
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
