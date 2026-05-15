import marimo

__generated_with = "0.18.0"
app = marimo.App(width="columns")


@app.cell
def _():
    # Standard library imports
    import pickle

    # Third party
    import torch
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter
    from scipy import stats

    from hmdcpd import index, constants, visualization, states, iom, load, utils
    return FuncFormatter, index, np, pd, pickle, plt, sns, torch, visualization


@app.cell
def _(np, pd, plt, sns, torch):
    # Suppress sci notation
    # sns.set_style("whitegrid")
    # sns.set_style()
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    sns.despine()
    np.set_printoptions(suppress=True, linewidth=150)
    pd.set_option("display.float_format", lambda x: "%.5f" % x)
    torch.set_printoptions(sci_mode=False)
    plt.rcParams["figure.dpi"] = 200
    figsize = (3, 3)
    figsize_tuning = (2.5, 3)
    shortened_conditions = {
        "Hazard Rate": "HZ",
        "Contingency": "CT",
    }
    return (figsize,)


@app.cell
def _(index, np):
    M = 200 # Timesteps to use for loading data
    num_alphas = 11
    alphas = np.linspace(0, 1, num_alphas)

    # Base directory ford data
    dir_data_base = index.dir_data / "cache/interventions"
    assert dir_data_base.exists()
    dir_models = dir_data_base / "lstm"
    samples = np.load(str(dir_data_base / "samples.npy"), allow_pickle=True)
    targets = np.load(str(dir_data_base / "targets.npy"), allow_pickle=True)

    # Experiment IDs to use
    # dict_model_dir_datasets = {path.stem: path for path in dir_models.iterdir() if path.stem == "san-4604"}
    dict_model_dir_datasets = dict(sorted({path.stem: path for path in dir_models.iterdir()}.items()))

    # Ensure they exist
    for _, dir_dataset in dict_model_dir_datasets.items():
        assert dir_dataset.exists()

    exp_ids = [exp_id for exp_id, _ in dict_model_dir_datasets.items()]
    return alphas, dict_model_dir_datasets, dir_data_base, exp_ids


@app.cell
def _(dir_data_base, pd, pickle):
    # Load human dataset
    # dir_dataset_h = index.dir_data / "raw/bb_datasets/hbb_dataset_250324_115252_1185959417"
    with open(str(dir_data_base / "dataset_meta.pkl"), "rb") as f:
        dict_metadata = pickle.load(f)

    df_data = pd.read_csv(dir_data_base / "trial_meta.csv", index_col=0)
    return (df_data,)


@app.cell
def _(np):
    def window_samples(
        samples: np.ndarray,
        endpoints: np.ndarray,
        N: int,
    ) -> np.ndarray:
        b, T, f = samples.shape
        # create a (b, N) array of time‑indices for each batch
        t_idx = endpoints[:, None] + np.arange(-N, 0)  # shape (b, N)

        # create matching batch indices
        b_idx = np.arange(b)[:, None]                 # shape (b, 1)

        # fancy‑index into samples
        return samples[b_idx, t_idx, :]               # -> (b, N, f)
    return (window_samples,)


@app.cell
def _(alphas, df_data, dict_model_dir_datasets, np, pd, window_samples):
    def _process_model_predictions(dict_model_dir_datasets, df_data, alphas, N=24, stat="hz", skip=None):
        """
        Process model predictions for intervention plots.
    
        Parameters:
        -----------
        dict_model_dir_datasets : dict
            Dictionary mapping experiment IDs to base file paths
        df_data : pd.DataFrame
            DataFrame containing trial data with columns: 'color_entered', 'length', 
            'Hazard Rate', 'idx_time', 'Contingency'
        num_alphas : int
            Number of alpha values to use
        N : int, default=24
            Timesteps to extract for intervention plots
    
        Returns:
        --------
        dict_model_pred_dfs_melted : dict
            Dictionary mapping experiment IDs to melted prediction DataFrames
        """
        dict_model_preds = {}
        dict_model_pred_dfs_melted = {}
        num_alphas = len(alphas)
        name = [stat,]
        if skip is not None:
            name.append("hidden" if skip else "cell")
        name += [
            "centroid-interventions",
            str(num_alphas),
            "all-states",
            "alphas.npz"
        ]
        name = "-".join(name)
    
        for exp_id, base_name_file in dict_model_dir_datasets.items():
            print(exp_id)
            preds = np.load(str(base_name_file / name))["preds"]
            preds_list = []
            preds_cent_idx = []
            color_entered = df_data["color_entered"].values - 1

            preds_list = []
            for centroid_idx in range(2):
                # Get all alphas for this centroid
                centroid_preds = preds[:, centroid_idx]  # Shape: (11, 81, 409, 5)

                # Window sample each alpha
                windowed_preds = []
                for alpha_idx in range(num_alphas):
                    windowed = window_samples(
                        centroid_preds[alpha_idx],
                        df_data["length"].values,
                        N
                    )
                    windowed_preds.append(windowed)

                preds_list.append(np.stack(windowed_preds))

            preds_int = np.stack(preds_list)  # Shape: (2, 11, 81, N, 3)

            _, _, num_videos, timesteps, num_channels = preds_int.shape

            min_vals = preds_int.min(axis=-1, keepdims=True)
            max_vals = preds_int.max(axis=-1, keepdims=True)
            range_vals = max_vals - min_vals

            list_df_melted = []
            for i, preds_norm in enumerate(preds_int):
                print(preds_norm.shape)
                pred_same_color = preds_norm[
                    np.arange(num_alphas)[:, None, None],
                    np.arange(num_videos)[None, :, None],
                    np.arange(timesteps)[None, None, :],
                    color_entered[None, :, None]
                ]

                pred_same_color_reshaped = pred_same_color.reshape(-1, timesteps)

                # Create the DataFrame
                df_preds = pd.DataFrame(pred_same_color_reshaped)

                # Add columns for alpha values, sample sequences, and timesteps
                df_preds['Alpha'] = np.repeat(alphas, num_videos)
                df_preds['Video'] = list(range(num_videos)) * num_alphas
                df_preds['Hazard Rate'] = list(df_data["Hazard Rate"].values) * num_alphas
                df_preds['idx_time'] = list(df_data["idx_time"].values) * num_alphas
                df_preds['Contingency'] = list(df_data["Contingency"].values) * num_alphas
                df_preds['trial'] = list(df_data["trial"].values) * num_alphas
            
                df_preds_melted = df_preds.melt(
                    id_vars=[
                        'Alpha',
                        'Video',
                        'Hazard Rate',
                        'idx_time',
                        'Contingency',
                        'trial',
                    ],
                    var_name='Timestep',
                    value_name='Value',
                )
                df_preds_melted["Value"] = 1 - df_preds_melted["Value"]
                df_preds_melted["Centroid"] = i
                # df_preds_melted = df_preds_melted[df_preds_melted["idx_time"] == 2]

                list_df_melted.append(df_preds_melted)

            dict_model_pred_dfs_melted[exp_id] = pd.concat(list_df_melted)
    
        return dict_model_pred_dfs_melted

    dict_model_pred_dfs_melted_hz = _process_model_predictions(
        dict_model_dir_datasets=dict_model_dir_datasets,
        df_data=df_data,
        alphas=alphas,
        N=25  # Can override the default if needed
    )

    dict_model_pred_dfs_melted_straight_hz = {
        key: df[
            (df.trial == "Straight")
            & (df.idx_time == 2)
            ]
        for key, df in dict_model_pred_dfs_melted_hz.items()
    }

    bounce_idx = 16
    dict_model_pred_dfs_melted_bounce_hz = {
        key: df[
            (df.trial == "Bounce")
            & (df.Timestep >= bounce_idx)
            ]
        for key, df in dict_model_pred_dfs_melted_hz.items()
    }

    for _key, _df in dict_model_pred_dfs_melted_bounce_hz.items():
        dict_model_pred_dfs_melted_bounce_hz[_key].loc[:, "Timestep"] -= bounce_idx
    return (
        bounce_idx,
        dict_model_pred_dfs_melted_hz,
        dict_model_pred_dfs_melted_straight_hz,
    )


@app.cell
def _(
    FuncFormatter,
    dict_model_pred_dfs_melted_straight_hz,
    figsize,
    plt,
    sns,
):
    for _exp_id, df_preds_melted in dict_model_pred_dfs_melted_straight_hz.items():
        plt.figure(figsize=figsize)
        ax = sns.lineplot(
            data=df_preds_melted[
                # (df_preds_melted["Centroid"] == 0) &
                (df_preds_melted["Alpha"] == 0.0)
                # & (df_preds_melted["idx_time"] == 2)
            ],
            x="Timestep",
            y="Value",
            hue="Hazard Rate",
            # palette=palette,
            errorbar="ci",
        )
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
        # plt.ylim((0, 1.0))
        plt.ylabel("Probability Color Change")
        plt.title(f"{_exp_id.upper()} Grayzone Probability Color Change for High and Low Hz")
        # plt.title(""), #f"{_exp_id.upper()} Grayzone Probability Color Change for High and Low Hz")
        plt.legend(title="Hazard Rate")
        plt.show()
    return


@app.cell
def _(FuncFormatter, figsize, np, plt, sns, visualization):
    def plot_interventions(
        dict_model_pred_dfs_melted,
        dict_exp_alpha_mult,
        alphas,
        alpha_operator=lambda left, right: left <= right,
        alpha_comparison=0.0,
        figsize=figsize,
        stat="Hazard Rate",
        stat_cond="Low",
        hue="Alpha",
        title=None,
        reference=True,
        ref_cond="High",
        ref_palette="flare",
        x="Timestep",
        y="Value",
        ylabel="Probability Color Change",
        legend_loc="upper left",
        vline=None,
    ):
        if title:
            title_base = title
        else:
            title_base = (
                f"{stat_cond.title()} {stat.title()} Grayzone Probability Color Change with "
                "Varying Interventions"
            )

        stat_short = "Hz" if "hazard" in stat.lower() else "Cont"

        for exp_id, df_preds_melted in dict_model_pred_dfs_melted.items():
            print(exp_id)

            # if exp_id != "san-4604":
            #     continue
            alphas_to_plot = alphas[alpha_operator(alphas * -1, alpha_comparison)]
            palette = visualization.get_color_palette(
                sorted(alphas_to_plot, key=abs),
                (("viridis", len(alphas_to_plot)),),
                linspace_range=np.array((0.0, 1.1)),
            )

            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            # fig = plt.figure(figsize=figsize)
            df = df_preds_melted[
                    (df_preds_melted[stat] == stat_cond) &
                    (df_preds_melted["Centroid"] == dict_exp_alpha_mult[exp_id][stat_cond]) &
                    (alpha_operator(df_preds_melted[hue] * -1, alpha_comparison))
                ]
        
            # print(df)

            ax = sns.lineplot(
                data=df_preds_melted[
                    (df_preds_melted[stat] == stat_cond) &
                    (df_preds_melted["Centroid"] == dict_exp_alpha_mult[exp_id][stat_cond]) &
                    (alpha_operator(df_preds_melted[hue] * -1, alpha_comparison))
                ],
                x=x,
                y=y,
                hue=hue,
                palette=palette,
                errorbar="ci",
                ax=ax,
            )
            plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
            # plt.ylim((0, 1.0))
            plt.ylabel(ylabel)
            # plt.title(f"{exp_id.upper()} {title_base}")
            plt.title(" ") # {title_base}")

            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: abs(float(t[0]))))
            legend_title = f"{stat_cond.title()} {stat_short} {hue.title()}"
            legend_alphas = ax.legend(
                handles, 
                labels, 
                title=legend_title,
                bbox_to_anchor=(1, 1.016),
                loc=legend_loc or "upper right",
            )
            # Round the legend labels
            for text in legend_alphas.get_texts():
                label_value = float(text.get_text())
                text.set_text(f"{abs(label_value):.1f}")  # Format to 2 decimal places
            ax.add_artist(legend_alphas)

            fig.canvas.draw()
            legend_alphas.get_window_extent().width/100/figsize[0]
            # Plot reference
            sns.lineplot(
                data=df_preds_melted[
                    (df_preds_melted[stat] == ref_cond) &
                    (df_preds_melted[hue] == 0.0)
                ],
                x=x,
                y=y,
                color="red",
                errorbar="ci",
                ax=ax,
                legend=False,
                linewidth=2
            )
        
            # Create line handle
            from matplotlib.lines import Line2D
            ref_line = Line2D([0], [0], color='red', linewidth=2.5)
            fig.canvas.draw()
            # Make title same length as first legend title
            # Count characters and pad accordingly
            first_title_len = len(legend_title) + 3
            ref_title = f"{ref_cond.title()} {stat_short}"
            ref_title_len = len(ref_title)
        
            # Pad with non-breaking spaces for better control
            if first_title_len > ref_title_len:
                diff = first_title_len - ref_title_len
                # Use a combination of regular spaces and thin spaces
                ref_legend_title = " " * (diff//2) + ref_title + " " * (diff//2 + diff%2)
            else:
                ref_legend_title = ref_title


            offset = 0.08 + len(handles) * 0.045 + 0.03
        
            legend_ref = ax.legend(
                handles=[ref_line],
                labels=[""],
                bbox_to_anchor=(1, 1 - offset),
                loc="upper left",
                title=ref_legend_title,
                handlelength=2.5,
                borderpad=0.5,  # Match the first legend
                columnspacing=0,  # No column spacing needed
                handletextpad=0.5,  # Some padding even with no text
            )
            # Center the title
            legend_ref.get_title().set_ha('center')
    
            ax.add_artist(legend_ref)

            if vline:
                # Add a vertical dashed line
                ax.axvline(x=vline, color='gray', linestyle='--', linewidth=2, alpha=0.7)
            
                # Create a handle for the dashed line
                from matplotlib.lines import Line2D
                dashed_line = Line2D([0], [0], color='gray', linestyle='--', linewidth=2)
    
                ref_title = f"{ref_cond.title()} {stat_short}"
                ref_title_len = len(ref_title)
            
                # Pad with non-breaking spaces for better control
                vline_title = "Intervention"
                if first_title_len > ref_title_len:
                    diff = first_title_len - ref_title_len - 6
                    # Use a combination of regular spaces and thin spaces
                    vline_title_legend = " " * (diff//2) + vline_title + " " * (diff//2 + diff%2)
                else:
                    vline_title_legend = vline_title
    
                # print(len(legend_title), len(ref_legend_title))
            
                # Calculate position for third legend (below reference legend)
                # Previous offset + height of reference legend + gap
                offset_dashed = offset + 0.08 + 0.03 + 0.045 # Adjust as needed
            
                # Create the third legend
                legend_dashed = ax.legend(
                    handles=[dashed_line],
                    labels=[""],  # Empty label like the reference
                    bbox_to_anchor=(1, 1 - offset_dashed),
                    loc="upper left",
                    title=vline_title_legend,  # Or whatever title you want
                    handlelength=2.5,
                )
            
                # Center the title
                legend_dashed.get_title().set_ha('center')
        
                # Add to the plot
                ax.add_artist(legend_dashed)

            plt.show()
    return (plot_interventions,)


@app.cell
def _(
    alphas,
    bounce_idx,
    df_data,
    dict_model_dir_datasets,
    dict_model_pred_dfs_melted_straight_hz,
    exp_ids,
    figsize,
    np,
    pd,
    plot_interventions,
    window_samples,
):
    def process_model_predictions(dict_model_dir_datasets, df_data, alphas, N=24, stat="hz", skip=None):
        """
        Process model predictions for intervention plots.
    
        Parameters:
        -----------
        dict_model_dir_datasets : dict
            Dictionary mapping experiment IDs to base file paths
        df_data : pd.DataFrame
            DataFrame containing trial data with columns: 'color_entered', 'length', 
            'Hazard Rate', 'idx_time', 'Contingency'
        num_alphas : int
            Number of alpha values to use
        N : int, default=24
            Timesteps to extract for intervention plots
    
        Returns:
        --------
        dict_model_pred_dfs_melted : dict
            Dictionary mapping experiment IDs to melted prediction DataFrames
        """
        dict_model_preds = {}
        dict_model_pred_dfs_melted = {}
        num_alphas = len(alphas)
        name = [stat,]
        if skip is not None:
            name.append("hidden" if skip else "cell")
        name += [
            "centroid-interventions",
            str(num_alphas),
            # "all-states",
            "alphas.npz"
        ]
        name = "-".join(name)
    
        for exp_id, base_name_file in dict_model_dir_datasets.items():
            # print(exp_id)
            preds = np.load(str(base_name_file / name))["preds"]
            preds_list = []
            preds_cent_idx = []
            color_entered = df_data["color_entered"].values - 1

            preds_list = []
            for centroid_idx in range(2):
                # Get all alphas for this centroid
                centroid_preds = preds[:, centroid_idx]  # Shape: (11, 81, 409, 5)

                # Window sample each alpha
                windowed_preds = []
                for alpha_idx in range(num_alphas):
                    windowed = window_samples(
                        centroid_preds[alpha_idx],
                        df_data["length"].values,
                        N
                    )
                    windowed_preds.append(windowed)

                preds_list.append(np.stack(windowed_preds))

            preds_int = np.stack(preds_list)  # Shape: (2, 11, 81, N, 3)
            # print(preds_int.shape)

            _, _, num_videos, timesteps, num_channels = preds_int.shape

            min_vals = preds_int.min(axis=-1, keepdims=True)
            max_vals = preds_int.max(axis=-1, keepdims=True)
            range_vals = max_vals - min_vals
            # print(range_vals.shape)

            list_df_melted = []
            for i, preds_norm in enumerate(preds_int):
                # print(preds_norm.shape)
                pred_same_color = preds_norm[
                    np.arange(num_alphas)[:, None, None],
                    np.arange(num_videos)[None, :, None],
                    np.arange(timesteps)[None, None, :],
                    color_entered[None, :, None]
                ]

                # print(pred_same_color.sum(axis=-1).shape)
                pred_same_color_reshaped = pred_same_color.reshape(-1, timesteps)

                # Create the DataFrame
                df_preds = pd.DataFrame(pred_same_color_reshaped)

                # Add columns for alpha values, sample sequences, and timesteps
                df_preds['Alpha'] = np.repeat(alphas, num_videos)
                df_preds['Video'] = list(range(num_videos)) * num_alphas
                df_preds['Hazard Rate'] = list(df_data["Hazard Rate"].values) * num_alphas
                df_preds['idx_time'] = list(df_data["idx_time"].values) * num_alphas
                df_preds['Contingency'] = list(df_data["Contingency"].values) * num_alphas
                df_preds['trial'] = list(df_data["trial"].values) * num_alphas
            
                df_preds_melted = df_preds.melt(
                    id_vars=[
                        'Alpha',
                        'Video',
                        'Hazard Rate',
                        'idx_time',
                        'Contingency',
                        'trial',
                    ],
                    var_name='Timestep',
                    value_name='Value',
                )
                df_preds_melted["Value"] = 1 - df_preds_melted["Value"]
                df_preds_melted["Centroid"] = i
                # df_preds_melted = df_preds_melted[df_preds_melted["idx_time"] == 2]

                list_df_melted.append(df_preds_melted)

            dict_model_pred_dfs_melted[exp_id] = pd.concat(list_df_melted)

        return dict_model_pred_dfs_melted

    dict_exp_alpha_mult = {
        exp_id: {"Low": 1, "High": 0}
        for exp_id in exp_ids
    }

    # dict_exp_alpha_mult.update(
    #     {
    #         exp_id: {"Low": 0, "High": 1}
    #         for exp_id in [
    #             # "san-4604",
    #             "san-4606",
    #             # "san-4601",
    #             # "san-4603",
    #             # "san-4618",
    #             # "san-4617",
    #             # "san-4618",
    #         ]
    #     }
    # )
    # for exp_id,  mult in dict_exp_alpha_mult.items():
    #     print(exp_id, mult)
    timesteps_plot = 26
    timestep_intervention = timesteps_plot - 24

    _dict_model_pred_dfs_melted_hz = process_model_predictions(
        dict_model_dir_datasets=dict_model_dir_datasets,
        df_data=df_data,
        alphas=alphas,
        N=timesteps_plot,
    )

    for _exp_id, df in _dict_model_pred_dfs_melted_hz.items():
        df = df[
            (df.trial == "Straight")
            & (df.idx_time == 2)
        ]
        df['Type'] = (
            df['Hazard Rate'].apply(lambda x: 'L' if x == 'Low' else 'H') +
            '2' +
            df['Centroid'].apply(lambda x: 'L' if x == 0 else 'H')
        )
        df['Model'] = _exp_id
        dict_model_pred_dfs_melted_straight_hz[_exp_id] = df

    # bounce_idx = 16
    _dict_model_pred_dfs_melted_bounce_hz = {
        _exp_id: df[
            (df.trial == "Bounce")
            & (df.Timestep >= bounce_idx)
            ]
        for _exp_id, df in _dict_model_pred_dfs_melted_hz.items()
    }

    for _exp_id, df in _dict_model_pred_dfs_melted_bounce_hz.items():
        _dict_model_pred_dfs_melted_bounce_hz[_exp_id].loc[:, "Timestep"] -= bounce_idx

    plot_interventions(
        dict_model_pred_dfs_melted_straight_hz,
        dict_exp_alpha_mult,
        alphas,
        legend_loc="upper left",
        vline=timestep_intervention,
        figsize=figsize,
        title=None,
    )
    plot_interventions(
        dict_model_pred_dfs_melted_straight_hz,
        dict_exp_alpha_mult,
        alphas,
        stat_cond="High",
        ref_cond="Low",
        vline=timestep_intervention,
        figsize=figsize,
    )
    return


@app.cell
def _(dict_model_pred_dfs_melted_straight_hz):
    dict_model_pred_dfs_melted_straight_hz
    return


@app.cell
def _(dict_model_pred_dfs_melted_hz, dict_model_pred_dfs_melted_straight_hz):
    for _exp_id, _df in dict_model_pred_dfs_melted_hz.items():
        # print(len(_df))
        _df = _df[
            (_df.trial == "Straight")
            & (_df.idx_time == 2)
        ]
        # print(len(df))
        _df['Type'] = (
            _df['Hazard Rate'].apply(lambda x: 'L' if x == 'Low' else 'H') +
            '2' +
            _df['Centroid'].apply(lambda x: 'L' if x == 0 else 'H')
        )
        _df['Model'] = _exp_id
        # print(_df)
        dict_model_pred_dfs_melted_straight_hz[_exp_id] = _df
    dict_model_pred_dfs_melted_straight_hz[_exp_id]
    return


@app.cell
def _(
    FuncFormatter,
    dict_model_pred_dfs_melted_straight_hz,
    figsize,
    np,
    pd,
    plt,
    sns,
    visualization,
):
    palette = visualization.get_color_palette(
        ['L2L', 'L2H', 'H2L', 'H2H'],
        (("flare", 4),),
        linspace_range=np.array((0.0, 1.1)),
    )

    _df_concat = pd.concat([_df for _, _df in dict_model_pred_dfs_melted_straight_hz.items()])
    _df = _df_concat[
        (_df_concat["Timestep"] == 24)
    ]


    # for exp_id, df_preds in dict_model_preds.items():
    fig = plt.figure(figsize=figsize)
    # plt.title(f"{exp_id.upper()} {title_base}")
    _ax = sns.pointplot(
        _df,
        x="Alpha",
        y="Value",
        hue="Type",
        palette=palette,
    )
    plt.xlabel("Intervention Strength (Alpha)")
    plt.ylabel("Final Color Change Probability")
    _ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:,.1f}'.format(x*0.1)))
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
