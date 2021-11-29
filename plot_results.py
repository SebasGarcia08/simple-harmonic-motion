from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from video_processing import read_video_array
from typing import Dict, Tuple, Union, List
import pandas as pd
import numpy as np

START_FRAME = 142
END_FRAME = 268


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

    fig = plt.figure()
    gs = fig.add_gridspec(
        ncols=3,
        nrows=3,
        width_ratios=widths,
        height_ratios=heights,
    )
    video_plot = fig.add_subplot(gs[:, 0])
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

    def __call__(self) -> Tuple[np.ndarray, Result, Result]:
        while self.current_frame < self.end_frame:
            frame = self.video[self.current_frame]
            exp = self.get_result(self.expdf)
            theo = self.get_result(self.theodf)
            self.current_frame += 1
            return frame, exp, theo

    def get_result(self, df: pd.DataFrame) -> Result:
        result_args: Dict[str, float] = dict()
        columns: list[str] = df.columns
        for col in columns:
            result_args[col] = df.loc[self.current_frame, col]

        return Result(**result_args)


@dataclass
class AnimationRunner(object):
    video_axes: plt.Axes
    values_axes: Result

    def __post_init__(self):
        self.time: List[float] = []

    def __call__(self, data: Tuple[np.ndarray, Result, Result]):
        frame, exp, theo = data
        self.video_axes.set_array(frame)
        return self.video_axes


def main():
    video_arr, fps = read_video_array("data/raw/fisica_video.mp4")
    exp = pd.read_csv("data/processed/experimental_data.csv")
    tho = pd.read_csv("data/processed/theoretical_system.csv")
    print(video_arr.shape)

    fig, video_axes, axes = make_figures()
    video_axes.set_title("Video")
    init_img = np.random.rand(video_arr.shape[-2], video_arr.shape[-1])
    video_axes.imshow(init_img)

    data_gen = DataGenerator(video_arr, START_FRAME, END_FRAME, exp, tho)
    runner = AnimationRunner(video_axes=video_axes, values_axes=axes)

    ani = animation.FuncAnimation(
        fig, runner, data_gen, blit=True, interval=60, repeat=False
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
