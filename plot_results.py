from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from video_processing import read_video_array
from typing import Any, Dict, Tuple, Union, List
from dataclasses import field
import pandas as pd
import numpy as np

START_FRAME = 142
END_FRAME = 268
FIG_SIZE = (20, 10)

graphs = {
    (
        0,
        1,
    ): "theta",
    (
        1,
        1,
    ): "velocity",
    (
        2,
        1,
    ): "acceleration",
    (
        0,
        2,
    ): "mechanical_energy",
    (
        1,
        2,
    ): "kinetic_energy",
    (
        2,
        2,
    ): "potential_energy",
}


@dataclass
class Result(object):
    theta: Union[float, plt.Axes]
    velocity: Union[float, plt.Axes]
    acceleration: Union[float, plt.Axes]
    kinetic_energy: Union[float, plt.Axes]
    potential_energy: Union[float, plt.Axes]
    mechanical_energy: Union[float, plt.Axes]


def make_figures() -> Tuple[plt.Figure, plt.Axes, Result]:
    widths = [4, 4, 4]
    heights = [1, 1, 1]

    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(
        ncols=3,
        nrows=3,
        width_ratios=widths,
        height_ratios=heights,
    )
    video_plot = fig.add_subplot(gs[:, 0])
    axes: Dict[str, plt.Axes] = dict()

    for rowi in range(3):
        for coli in range(1, 3):
            key = (
                rowi,
                coli,
            )
            title = graphs[key]
            axes[title] = fig.add_subplot(gs[rowi, coli])
            axes[title].set_title(title)
    result_plot = Result(**axes)
    return fig, video_plot, result_plot


@dataclass()
class DataGenerator(object):
    video: np.ndarray
    start_frame: int
    end_frame: int
    expdf: pd.DataFrame
    theodf: pd.DataFrame

    def __post_init__(self):
        self.current_frame = self.start_frame

    def __call__(self) -> Tuple[float, np.ndarray, Result, Result]:
        while self.current_frame < self.end_frame:
            frame = self.video[self.current_frame]
            exp = self.get_result(self.expdf)
            theo = self.get_result(self.theodf)
            time = self.expdf.loc[self.current_frame, "t"]
            self.current_frame += 1
            yield time, frame, exp, theo

    def get_result(self, df: pd.DataFrame) -> Result:
        result_args: Dict[str, float] = dict()

        for col in graphs.values():
            result_args[col] = df.loc[self.current_frame, col]
        return Result(**result_args)


@dataclass
class AnimationRunner(object):
    subfigs: Dict[str, plt.Axes]

    def __post_init__(self):
        self.time: List[float] = []
        self.exp_data: Dict[str, List[float]] = dict()
        self.theo_data: Dict[str, List[float]] = dict()

        for sysmetric in graphs.values():
            self.exp_data[sysmetric] = []
            self.theo_data[sysmetric] = []

    def __call__(self, data: Tuple[float, np.ndarray, Result, Result]):
        time, frame, exp, theo = data
        self.time.append(time)
        for axis in ["x", "y"]:
            self.subfigs["video"].tick_params(
                axis=axis,  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
            )  # labels along the bottom edge are off

        self.subfigs["video"].imshow(frame)
        for sysmetric in graphs.values():
            self.exp_data[sysmetric].append(getattr(exp, sysmetric))
            self.theo_data[sysmetric].append(getattr(theo, sysmetric))
            sysfig: plt.Axes = self.subfigs[sysmetric]
            sysfig.clear()
            sysfig.grid()
            sysfig.plot(
                self.time,
                self.exp_data[sysmetric],
                label="experimental",
                color="red",
                linestyle="dashed",
                marker="o",
                markersize=5,
            )
            sysfig.plot(
                self.time,
                self.theo_data[sysmetric],
                label="theoretical",
                color="blue",
                linestyle="dashed",
                marker="o",
                markersize=5,
            )
            sysfig.set_title(sysmetric)
            sysfig.legend()
        return self.subfigs.values()


def initialize_animation(
    video_axes: plt.Axes, values_axes: Result, h: int, w: int
) -> Dict[str, plt.Axes]:
    video_axes.set_title("Video")
    init_image = np.zeros((h, w, 3), dtype=np.uint8)
    video_axes.imshow(init_image)
    axes = {"video": video_axes}

    for sysmetric in graphs.values():
        ax = getattr(values_axes, sysmetric)
        ax.plot([], [])
        ax.set_title(sysmetric)
        axes[sysmetric] = ax
    return axes


def main():
    video_arr, fps = read_video_array("data/raw/fisica_video.mp4")
    exp = pd.read_csv("data/processed/experimental_system.csv", index_col="frame")
    tho = pd.read_csv("data/processed/theoretical_system.csv", index_col="frame")
    time_btw_frames = (exp["t"].values[1:] - exp["t"].values[:-1]).mean()

    fig, video_axes, axes = make_figures()
    subfigs = initialize_animation(
        video_axes, axes, video_arr.shape[1], video_arr.shape[2]
    )

    data_gen = DataGenerator(
        video=video_arr,
        start_frame=START_FRAME,
        end_frame=END_FRAME,
        expdf=exp,
        theodf=tho,
    )
    runner = AnimationRunner(
        subfigs=subfigs,
    )
    plt.title("Simple pendulum")
    fig.suptitle("Experimental vs theoretical results")
    ani = animation.FuncAnimation(
        fig, runner, data_gen, repeat=False, blit=False, interval=time_btw_frames
    )
    plt.tight_layout()
    ffwriter = animation.FFMpegWriter(fps=fps)
    ani.save("results_2.mp4", writer=ffwriter)
    # plt.show()


if __name__ == "__main__":
    main()
