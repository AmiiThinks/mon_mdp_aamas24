# ruff: noqa: F403, F405
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import seaborn as sns
import argparse
import importlib
import inspect
import sys

from src.plot_utils import *

sys.path.append("./configs/plots")

sns.set_context("paper")
# sns.set_style("whitegrid", {"legend.frameon": True})
sns.set_style("darkgrid", {"legend.frameon": True})
plt.rcParams["axes.axisbelow"] = False
# plt.rcParams["axes.grid"] = False
plt.rcParams["grid.linestyle"] = "--"
# plt.rcParams["font.family"] = "DejaVu Sans Mono"
plt.rcParams["font.family"] = "Bree Serif"
font_size = 12
plt.rcParams["font.size"] = font_size

# ------------------------------------------------------------------------------
# Note that this script tries to use the Bree Serif font. To install it, check
# where matplotlib fonts are saved by running
#    from matplotlib.font_manager import findfont, FontProperties
#    print(findfont(FontProperties(family=["sans-serif"])))
#
# Then, download Bree Serif and install it there.
#
# Finally, delete matplotlib cache. To find it, run
#     matplotlib.get_cachedir()
# ------------------------------------------------------------------------------


def plot(folder):
    n_seeds = 100
    in_paper = [
        "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor",
        "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor",
        "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor",
        "Gridworld-Medium-3x3-v0_mes50/iNMonitor_nm5",
        "Gridworld-Medium-3x3-v0_mes50/iLimitedTimeMonitor",
        "Gridworld-Medium-3x3-v0_mes50/iLimitedUseBonusMonitor_mb7",
        "Gridworld-Easy-3x3-Stochastic-v0_mes50/iStatelessBinaryMonitor",
        "Gridworld-Medium-3x3-Stochastic-v0_mes50/iStatelessBinaryMonitor",
        "Gridworld-Medium-3x3-Stochastic-v0_mes50/iToySwitchMonitor",
        "Gridworld-Medium-3x3-Stochastic-v0_mes50/iNMonitor_nm5",
        "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLimitedTimeMonitor",
        "Gridworld-Medium-3x3-Stochastic-v0_mes50/iLimitedUseBonusMonitor_mb7",
    ]
    max_algname_length = max([len(v) for v in alg_to_label.values()])
    latex_table = {alg: "" for alg in alg_to_label.keys()}

    for q0 in q_init_values:
        for mon in mon_to_label.keys():
            fig, axs = make_subplots(nrows=1, ncols=1, width_per_plot=2.6, height_per_plot=2)
            axs = axs[0][0]
            axs.set_prop_cycle(None)
            nothing_to_plot = True

            print(">>>", q0, mon)

            for alg in alg_to_label.keys():
                data_np = []
                seeds_completed = 0
                for seed in range(0, n_seeds):
                    try:
                        filename = os.path.join(folder, mon, q0, alg) + f"_test_{seed}.npy"
                        data_np.append(np.load(filename))
                        seeds_completed += 1
                    except Exception as e:  # noqa: F841
                        print(f"{alg} missing seed {seed}")
                        pass

                try:
                    data_np = np.stack(data_np)
                except Exception as e:
                    print(e)
                    continue

                error_shade_plot(
                    axs,
                    data_np,
                    testing_frequency,
                    smoothing_window=smoothing_window,
                    label=alg_to_label[alg],
                    lw=3,
                )
                # error_bar_plot(
                #     axs,
                #     data_np,
                #     testing_frequency,
                #     smoothing_window=smoothing_window,
                #     label=alg_to_label[alg],
                #     lw=1,
                #     elinewidth=1,
                #     capsize=1,
                # )
                nothing_to_plot = False

                steps_converged = np.zeros(data_np.shape[0]) * np.nan
                for j, d in enumerate(data_np):
                    for jj in range(len(d) - consecutive_steps_for_convergence):
                        try:
                            if np.all(d[jj:] >= mon_to_opt[mon]):
                                steps_converged[j] = jj * testing_frequency
                                break
                        except Exception as e:  # noqa: F841
                            pass

                n_converged = steps_converged.shape[0] - np.isnan(steps_converged).sum()
                mean_steps = np.nanmean(steps_converged)
                error_steps = 1.96 * np.nanstd(steps_converged) / np.sqrt(n_converged)

                space_pad = max_algname_length - len(alg_to_label[alg])
                stats_str = f"${mean_steps:,.0f} \pm {error_steps:,.0f}$ & {n_converged}/{seeds_completed}"  # fmt: skip
                stats_str = stats_str.replace("$nan \pm nan$", "---")
                print(alg_to_label[alg], " " * space_pad, stats_str)
                if mon in in_paper:
                    latex_table[alg] += " & " + stats_str

                # axs.set_title(mon_to_label[mon], fontsize=font_size+4)
                # axs.set_ylabel("Expected Return", fontsize=font_size)

                axs.tick_params(axis="x", labelsize=font_size - 2, pad=-2)
                axs.tick_params(axis="y", labelsize=font_size - 2, pad=y_tick_pad)

                axs.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                if mon_to_yticks[mon] is not None:
                    axs.yaxis.set_ticks(mon_to_yticks[mon])
                axs.set_ylim(mon_to_ylim[mon])

                axs.ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
                if mon_to_xticks[mon] is not None:
                    axs.xaxis.set_ticks(mon_to_xticks[mon])
                axs.set_xlim([0, mon_to_xlim[mon]])

                xlabel = "Training Steps (1e3)"  # depends on the format above
                axs.set_xlabel(xlabel, fontsize=font_size, labelpad=-23, loc="right")
                axs.xaxis.offsetText.set_visible(False)  # hide the exp notation

            if nothing_to_plot:
                continue

            plt.draw()
            savepath = os.path.join(folder, savedir)
            os.makedirs(savepath, exist_ok=True)
            savename = os.path.join(mon, q0).replace("\\", "_").replace("/", "_")
            savename = os.path.join(savepath, savename + ".png")
            plt.savefig(savename, bbox_inches="tight", pad_inches=0, dpi=1500)
            print()

        print("-----\n")
        for k, v in latex_table.items():
            print(alg_to_label[k] + v + "\n" + "\\\\" + "\n" + "\\hline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-f", "--folder")
    args = parser.parse_args()

    # https://stackoverflow.com/a/77350187/754136
    # inject config variables into the global namespace
    cfgmod = importlib.import_module(inspect.getmodulename(args.config))
    dicts = {k: v for k, v in inspect.getmembers(cfgmod) if not k.startswith("_")}
    globals().update(**dicts)

    plot(args.folder)
