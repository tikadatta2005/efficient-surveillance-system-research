import seaborn as sns
import matplotlib.pyplot as plt
import os
import math


def sub_plots(row, col, pos, data, x, y, x_label, y_label, title):
    # make subplots
    plt.subplot(row, col, pos)

    # show all plots
    for y_name in y:
        # line plot
        sns.lineplot(data=data, x=x, y=y_name, label=y_name, palette="colorblind")

    # show legend
    plt.legend()

    # show title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)



def save_plots(data, x, y, x_label, y_label, title, save_path):
    # make a canvas size
    plt.figure(figsize=(16, 10), facecolor="white")

    # show all plots
    for y_name in y:
        sns.lineplot(data=data, x=x, y=y_name, label=y_name)

    # show legend
    plt.legend()

    # show title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # file name
    name = title.replace(" ", "_").lower()

    # save figure
    plt.savefig(
        os.path.join(save_path, f"{name}.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    # close current figure
    plt.close()



def gen_line_charts(data, save_path, plot_name, prefix=[]):
    # make a canvas
    plt.figure(figsize=(24, 10), facecolor="white")

    # all chart y_names
    all_y_names = [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]

    # prefix based paired y_names
    paired = []

    # double loop
    for y_name in all_y_names:

        # add base y_name
        if len(prefix) == 0:
            paired.append([y_name])
            continue

        # pairs
        pair = []

        # secondary loop to add prefix
        for pref in prefix:
            pair.append(f"{pref}{y_name}")

        # append pair to paired
        paired.append(pair)


    # calculate rows for subplot
    row = math.ceil(len(paired) / 3)

    # loop pairs calling subplots
    for index, pairs in enumerate(paired):

        # sub plotter
        sub_plots(
            row,
            3,
            index + 1,
            data,
            x="epoch",
            y=pairs,
            title=f"{all_y_names[index].replace('_', ' ').title()} Curves",
            x_label="Epochs",
            y_label=all_y_names[index].replace('_', ' ').title()
        )


    # save all plot once
    plt.savefig(
        os.path.join(save_path, plot_name),
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    # show plot
    plt.show()

    # clear memory
    plt.close("all")


    # save individual
    for index, pair in enumerate(paired):

        save_plots(
            data,
            x="epoch",
            y=pair,
            x_label="Epochs",
            y_label=all_y_names[index].replace('_', ' ').title(),
            title=f"{all_y_names[index].replace('_', ' ').title()} Curves",
            save_path=save_path
        )


    # clear memory
    plt.close("all")