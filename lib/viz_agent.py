from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from networkx import draw_networkx
from PIL import Image

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
        draw_networkx(
            G,
            pos,
            node_color=colors,
            node_size=25,
            width=0.4,
            with_labels=False,
            edge_color=(0, 0, 0, 0.5),
        )
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

        plt.title(label=title_text)

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
