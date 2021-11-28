import matplotlib.pyplot as plt
from video_processing import read_video_array

video, fps = read_video_array("data/fisica_video.mp4")
print(video.shape)
widths = [4, 4, 4]
heights = [1, 1, 1]
gs_kw = dict(
    ncols=3,
    nrows=3,
    width_ratios=widths,
    height_ratios=heights,
)

fig = plt.figure()
gs = fig.add_gridspec(
    ncols=3,
    nrows=3,
    width_ratios=widths,
    height_ratios=heights,
)
video = fig.add_subplot(gs[:, 0])
graphs = {
    (
        0,
        1,
    ): "angle",
    (
        1,
        1,
    ): "vx",
    (
        2,
        1,
    ): "ax",
    (
        0,
        2,
    ): "Em",
    (
        1,
        2,
    ): "Ek",
    (
        2,
        2,
    ): "Eu",
}
axes = {}

for rowi in range(3):
    for coli in range(1, 3):
        title = graphs[
            (
                rowi,
                coli,
            )
        ]
        axes[title] = fig.add_subplot(gs[rowi, coli])
        axes[title].set_title(title)

plt.tight_layout()
plt.show()
