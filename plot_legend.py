import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


# https://stackoverflow.com/a/42170161/754136
from matplotlib.legend_handler import HandlerLine2D


class SymHandler(HandlerLine2D):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        return super(SymHandler, self).create_artists(
            legend, orig_handle, xdescent, 0.6 * height, width, height, fontsize, trans
        )


sns.set_context("paper")
# sns.set_style('whitegrid', {'legend.frameon': True})
sns.set_style("darkgrid", {"legend.frameon": True})
plt.rcParams["axes.axisbelow"] = False
# plt.rcParams["axes.grid"] = False
plt.rcParams["grid.linestyle"] = "--"
# plt.rcParams['font.family'] = 'DejaVu Sans Mono'
plt.rcParams["font.family"] = "Bree Serif"
font_size = 12
plt.rcParams["font.size"] = font_size

my_labels = [
    "Oracle",
    "Rew. Model",
    "Sequential",
    "Joint",
    r"Ignore $\bot$",
    r"$\bot$ = 0",
]

fig, axs = plt.subplots(1, 1)
figl = plt.figure()
legend_handles = []
legend_labels = []
for lab in my_labels:
    axs.plot(1, 1, label=lab, lw=3)
    handles, labels = axs.get_legend_handles_labels()
    legend_handles.extend(handles)
    legend_labels.extend(labels)

# Remove duplicate legends
unique_legend = [
    (h, l)
    for i, (h, l) in enumerate(zip(legend_handles, legend_labels))  # noqa: E741
    if l not in legend_labels[:i]
]

leg = figl.legend(
    *zip(*unique_legend),
    handler_map={matplotlib.lines.Line2D: SymHandler()},
    handleheight=2.4,
    labelspacing=0.05,
    prop={"size": font_size - 4},
    # loc='upper left', bbox_to_anchor=(1, 1.5), # right, outside
    loc="center",
    bbox_to_anchor=(0.485, -0.5),  # below, outside
    ncol=len(my_labels),
    columnspacing=1.0,
)

plt.draw()

plt.savefig("legend.png", bbox_inches="tight", pad_inches=0, dpi=1500)
