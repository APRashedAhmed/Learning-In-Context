import marimo

__generated_with = "0.18.0"
app = marimo.App(width="columns")


@app.cell(column=0)
def _(mo):
    mo.md(r"""
    # Figure - Contingency
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import marimo as mo

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
    from sklearn.cluster import KMeans
    from scipy import stats
    from numpy.lib.stride_tricks import sliding_window_view
    from matplotlib.patches import Rectangle
    from matplotlib.patches import Rectangle as mpatches_Rectangle

    # %aimport hmdcpd.visualization
    # %aimport hmdcpd.states
    # %aimport hmdcpd.iom
    # %aimport hmdcpd.utils
    # %aimport hmdcpd.load
    from hmdcpd import index, constants, visualization, states, iom, load, utils
    return (
        index,
        mo,
        np,
        pd,
        pickle,
        plt,
        sliding_window_view,
        sns,
        stats,
        torch,
        visualization,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Plotting Style
    """)
    return


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
        "Hazard Rate": "Hz",
        "Contingency": "Cont",
    }
    return figsize, figsize_tuning, shortened_conditions


@app.cell
def _(mo):
    mo.md(r"""
    ## Loading Data
    """)
    return


@app.cell
def _(index, np, pd, pickle):
    dataset = "extended_dataset"
    model_name = "lstm"
    dir_data_base = index.dir_repo / f"data/cache/model_states/{dataset}"
    samples = np.load(str(dir_data_base / "samples.npy"), allow_pickle=True)
    targets = np.load(str(dir_data_base / "targets.npy"), allow_pickle=True)
    batch_size, timesteps, _ = samples.shape

    timestep_array = np.tile(np.arange(timesteps), batch_size).reshape(batch_size, timesteps)
    df_data = pd.read_csv(dir_data_base / "trial_meta.csv", index_col=0)
    with open(str(dir_data_base / "dataset_meta.pkl"), "rb") as f:
        dict_metadata = pickle.load(f)

    padding_value = dict_metadata["padding_value"]
    length = df_data["length"].values
    mask_valid = (timestep_array < length[:, None])[:, :, None]
    targets = np.where(mask_valid, targets, padding_value)

    path_models = {
        path.stem: path for path in (dir_data_base / model_name).iterdir() 
        if path.stem == "san-4604"
    }

    for _exp_id, path in path_models.items():
        print(_exp_id)
        assert path.exists()

    T = 16
    change_idx = 5
    color_change_index = 1
    return (
        T,
        change_idx,
        color_change_index,
        df_data,
        dir_data_base,
        model_name,
        padding_value,
        path_models,
        samples,
        targets,
    )


@app.cell
def _(np, path_models, stats):
    M = 250

    dict_model_data = {}
    dict_model_states = {}
    dict_model_first_states = {}
    # not_include = ["san-4615",]

    for _exp_id, _path_model in path_models.items():
        # if exp_id in not_include:
        #     continue
        model_data = dict_model_data[_exp_id] = np.load(
            str(_path_model),
            allow_pickle=True,
        )
        dict_model_states[_exp_id] = model_data["states"] #[np.arange(8192), -M:df_data["length"].values - 1]

        dict_model_first_states[_exp_id] = stats.zscore(
            np.concat(
                [
                    model_data["hiddens"][:, :M],
                    model_data["cells"][:, :M],
                ],
                axis=-1,
            ),
            axis=(0, 1),
            ddof=1,
        )
    return M, dict_model_data


@app.cell
def _():
    exp_no_cont = {
        "san-4602",
        "san-4605",
        "san-4616",
        "san-4617",
    }

    dict_model_stat_units = {
        'hz': {
            # 'san-4602': {
            #     1: 'Unit 1 - Hidden 1',
            #     17: 'Unit 17 - Cell 1'
            # },
            # 'san-4605': {
            #     8: 'Unit 8 - Hidden 8',
            #     24: 'Unit 24 - Cell 8'},
            'san-4604': {
                15: 'Unit 15 - Hidden 15',
                31: 'Unit 31 - Cell 15'},
            # 'san-4603': {
            #     6: 'Unit 6 - Hidden 6',
            #     22: 'Unit 22 - Cell 6'},
            # 'san-4606': {
            #     11: 'Unit 11 - Hidden 11',
            #     27: 'Unit 27 - Cell 11'},
            # 'san-4601': {
            #     1: 'Unit 1 - Hidden 1',
            #     17: 'Unit 17 - Cell 1'
            #     },
            #   'san-4615': {4: 'Unit 4 - Hidden 4', 20: 'Unit 20 - Cell 4'},
            #   'san-4616': {5: 'Unit 5 - Hidden 5', 21: 'Unit 21 - Cell 5'},
            #   'san-4618': {4: 'Unit 4 - Hidden 4', 20: 'Unit 20 - Cell 4'},
            #   'san-4617': {0: 'Unit 0 - Hidden 0', 16: 'Unit 16 - Cell 0'},
            },
        'cont': {
            'san-4604': {
                11: 'Unit 11 - Hidden 11',
                27: 'Unit 27 - Cell 11'
            },
        #     'san-4603': {
        #         0: 'Unit 0 - Hidden 0',
        #         16: 'Unit 16 - Cell 0',
        #     },
        #     'san-4606': {
        #         # 11: 'Unit 11 - Hidden 11',
        #         12: 'Unit 12 - Hidden 12',
        #         28: 'Unit 28 - Cell 12',
        #     },
        #     'san-4601': {
        #         5: 'Unit 5 - Hidden 5',
        #         21: 'Unit 21 - Cell 5'
        #     },
        #     'san-4615': {12: 'Unit 12 - Hidden 12', 28: 'Unit 28 - Cell 12'},
        #     'san-4618': {9: 'Unit 9 - Hidden 9', 25: 'Unit 25 - Cell 9'},
        }
    }

    exp_ids = list(dict_model_stat_units["hz"].keys())
    exp_ids
    return dict_model_stat_units, exp_ids, exp_no_cont


@app.cell
def _(dict_model_stat_units):
    dict_model_stat_units
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Single Unit Tuning
    """)
    return


@app.cell
def _():
    # def _():
    #     num_changes = {
    #         "Low": 3,
    #         "High": 8,
    #     } # int(min([max(order) for _, order in dict_model_order.items()])) + 1
    #     change_labels = {
    #         cond: [f"Change {val + 1}" for val in range(num)]
    #         for cond, num in num_changes.items()
    #     }
    #     # Get all unique change labels across conditions
    #     all_change_labels = []
    #     max_changes = max(num_changes.values())  # 12
    #     for i in range(max_changes):
    #         all_change_labels.append(f"Change {i + 1}")

    #     palette = visualization.get_color_palette(
    #         all_change_labels,
    #         (("viridis", max_changes),),
    #         linspace_range=np.array((0.0, 1.1)),
    #     )

    #     # for hz, (dict_model_change_order_dfs, dict_model_order) in dict_hz_model_change_order_dfs.items():
    #     plot_ordered_change_activity_rows(
    #         dict_hz_model_change_order_dfs,
    #         dict_model_stat_units_all["hz"],
    #         exp_ids=exp_ids,
    #         palette=palette,
    #         change_labels=change_labels,
    #         cond_order=["Low", "High"],
    #         figsize=figsize_tuning,
    #         change_idx=change_idx,
    #         stat="Hazard Rate",
    #         errorbar="se",
    #         title="", #"Ordered Random Color Changes Split by Hazard Rate",
    #     )
    #     plt.show()
    # _()
    return


@app.cell
def _(
    T,
    change_idx,
    df_data,
    dict_model_stat_units_all,
    dict_model_states_all,
    exp_no_cont,
    get_ordered_sliding_window,
    pd,
    samples,
    targets,
):
    dict_cont_model_change_order_dfs = {}

    for cont, df_cont in df_data.groupby("Contingency"):
        _dict_model_change_order_dfs = {}
        _dict_model_order = {}
        # for _exp_id, _states_first_M_zscore in dict_model_states.items():
        for _exp_id, _states_first_M_zscore in dict_model_states_all.items():
            if _exp_id in exp_no_cont:
                continue
            _states_win, _sample_win, _target_win, _order = get_ordered_sliding_window(
                _states_first_M_zscore,
                samples,
                targets,
                df_cont,
                T,
                2,
                change_idx=change_idx,
            )
            _dict_model_order[_exp_id] = _order
            _model_stat_units = dict_model_stat_units_all["cont"][_exp_id]
            _stat_units = list(_model_stat_units.keys())

            _unit_dfs = []
            for _i, _unit in enumerate(_stat_units):
                _unit_activity = _states_win[:, :, _unit]
                _df_unit = pd.DataFrame(_unit_activity)
                _df_unit = _df_unit.assign(**{
                    "unit": _unit,
                    "order": [f"Color Change {_j+1}" for _j in _order],
                })

                _unit_dfs.append(_df_unit)

            _dict_model_change_order_dfs[_exp_id] = pd.concat(_unit_dfs).melt(
                id_vars=[
                    'unit',
                    'order',
                ],
                var_name='Timestep',
                value_name='Value',
            )

        dict_cont_model_change_order_dfs[cont] = (_dict_model_change_order_dfs, _dict_model_order)
    return (dict_cont_model_change_order_dfs,)


@app.cell
def _(
    change_idx,
    dict_cont_model_change_order_dfs,
    dict_model_stat_units,
    exp_ids,
    exp_no_cont,
    figsize_tuning,
    np,
    plot_ordered_change_activity_rows,
    plt,
    visualization,
):
    def _():
        num_changes = {
            "Low": 3,
            "High": 7,
        } # int(min([max(order) for _, order in dict_model_order.items()])) + 1
        change_labels = {
            cond: [f"Color Change {val + 1}" for val in range(num)]
            for cond, num in num_changes.items()
        }
        # Get all unique change labels across conditions
        all_change_labels = []
        max_changes = max(num_changes.values())  # 12
        for i in range(max_changes):
            all_change_labels.append(f"Color Change {i + 1}")

        palette = visualization.get_color_palette(
            all_change_labels,
            (("viridis", max_changes),),
            linspace_range=np.array((0.0, 1.1)),
        )

        plot_ordered_change_activity_rows(
            dict_cont_model_change_order_dfs,
            dict_model_stat_units["cont"],
            exp_ids=[_exp_id for _exp_id in exp_ids if _exp_id not in exp_no_cont],
            palette=palette,
            change_labels=change_labels,
            cond_order=["Low", "High"],
            figsize=figsize_tuning,
            change_idx=change_idx,
            stat="Contingency",
            title="Ordered Contingent Color Changes Split by Contingency",
            label_vline="Wall Bounce",
        )
    _()
    plt.show()
    return


@app.cell
def _(
    T,
    change_idx,
    df_data,
    dict_model_stat_units_all,
    dict_model_states_all,
    exp_no_cont,
    np,
    pd,
    samples,
    sliding_window_view,
    targets,
):
    def get_ordered_sliding_window_bounce_no_change(states, samples, targets, df_selected, T, change_idx=None, padding_value=None):
        if change_idx is None:
            change_idx = T // 2
        # Create sliding windows
        states = states[df_selected.index]
        samples = samples[df_selected.index]
        targets = targets[df_selected.index]

        num_trials, timesteps, _ = states.shape
        states_win = sliding_window_view(states, window_shape=T, axis=1)
        sample_win = sliding_window_view(samples, window_shape=T, axis=1)
        target_win = sliding_window_view(targets, window_shape=T, axis=1)

        # # Filter out windows with padding if padding_value is specified
        # if padding_value is not None:
        #     # Check which windows don't contain padding
        #     no_padding_mask = ~(target_win == padding_value).any(axis=(2, 3))
        # else:
        #     # If no padding_value specified, all windows are valid
        #     no_padding_mask = np.ones((num_trials, timesteps - T + 1), dtype=bool)

        # print(target_win.shape)

        # Center around random color change AND exclude padded windows
        mask_center = (
            (target_win[:, :, -4, change_idx] == 1)
            & (target_win[:, :, -2, change_idx] == 0)
        )
    
        states_win = np.moveaxis(states_win[mask_center], [2,], [1,])
        sample_win = np.moveaxis(sample_win[mask_center], [2,], [1,])
        target_win = np.moveaxis(target_win[mask_center], [2,], [1,])

        # timestep_win = timestep_win[mask_center]
        order_counts = np.cumsum(mask_center, axis=1) - 1
        order_vector = order_counts[mask_center].tolist()

        return states_win, sample_win, target_win, order_vector


    dict_cont_no_change_model_change_order_dfs = {}

    for _cont, _df_cont in df_data.groupby("Contingency"):
        _dict_model_change_order_dfs = {}
        _dict_model_order = {}
        # for _exp_id, _states_first_M_zscore in dict_model_states.items():
        for _exp_id, _states_first_M_zscore in dict_model_states_all.items():
            if _exp_id in exp_no_cont:
                continue
            _states_win, _sample_win, _target_win, _order = get_ordered_sliding_window_bounce_no_change(
                _states_first_M_zscore,
                samples,
                targets,
                _df_cont,
                T,
                change_idx=change_idx,
            )
            _dict_model_order[_exp_id] = _order
            _model_stat_units = dict_model_stat_units_all["cont"][_exp_id]
            _stat_units = list(_model_stat_units.keys())

            _unit_dfs = []
            for _i, _unit in enumerate(_stat_units):
                _unit_activity = _states_win[:, :, _unit]
                _df_unit = pd.DataFrame(_unit_activity)
                _df_unit = _df_unit.assign(**{
                    "unit": _unit,
                    "order": [f"No Color Change {_j+1}" for _j in _order],
                })

                _unit_dfs.append(_df_unit)

            _dict_model_change_order_dfs[_exp_id] = pd.concat(_unit_dfs).melt(
                id_vars=[
                    'unit',
                    'order',
                ],
                var_name='Timestep',
                value_name='Value',
            )

        dict_cont_no_change_model_change_order_dfs[_cont] = (_dict_model_change_order_dfs, _dict_model_order)
    return (dict_cont_no_change_model_change_order_dfs,)


@app.cell
def _(
    change_idx,
    dict_cont_no_change_model_change_order_dfs,
    dict_model_stat_units,
    exp_ids,
    exp_no_cont,
    figsize_tuning,
    np,
    plot_ordered_change_activity_rows,
    plt,
    visualization,
):
    def _():
        num_changes = {
            "Low": 7,
            "High": 7,
        } # int(min([max(order) for _, order in dict_model_order.items()])) + 1
        change_labels = {
            cond: [f"No Color Change {val + 1}" for val in range(num)]
            for cond, num in num_changes.items()
        }
        # Get all unique change labels across conditions
        all_change_labels = []
        max_changes = max(num_changes.values())  # 12
        for i in range(max_changes):
            all_change_labels.append(f"No Color Change {i + 1}")

        palette = visualization.get_color_palette(
            all_change_labels,
            (("viridis", max_changes),),
            linspace_range=np.array((0.0, 1.1)),
        )
        print(palette, all_change_labels)

        plot_ordered_change_activity_rows(
            dict_cont_no_change_model_change_order_dfs,
            dict_model_stat_units["cont"],
            exp_ids=[_exp_id for _exp_id in exp_ids if _exp_id not in exp_no_cont],
            palette=palette,
            change_labels=change_labels,
            cond_order=["Low", "High"],
            figsize=figsize_tuning,
            change_idx=change_idx,
            stat="Contingency",
            title="Ordered Contingent Color Changes Split by Contingency",
            label_vline="Wall Bounce",
            bbox_to_anchor=(-1.4, 0.5)
        )
    _()
    plt.show()
    return


@app.cell
def _(dict_cont_no_change_model_change_order_dfs):
    dict_cont_no_change_model_change_order_dfs
    return


@app.cell(column=2)
def _(mo):
    mo.md(r"""
    # Ordered Change Activity
    """)
    return


@app.cell
def _(
    M,
    T,
    change_idx,
    color_change_index,
    df_data,
    dict_model_first_states_all,
    dict_model_stat_units_all,
    np,
    padding_value,
    pd,
    samples,
    sliding_window_view,
    targets,
):
    def get_ordered_sliding_window(states, samples, targets, df_selected, T, k, change_idx=None, padding_value=None):
        if change_idx is None:
            change_idx = T // 2
        # Create sliding windows
        states = states[df_selected.index]
        samples = samples[df_selected.index]
        targets = targets[df_selected.index]

        num_trials, timesteps, _ = states.shape
        states_win = sliding_window_view(states, window_shape=T, axis=1)
        sample_win = sliding_window_view(samples, window_shape=T, axis=1)
        target_win = sliding_window_view(targets, window_shape=T, axis=1)

        # # Filter out windows with padding if padding_value is specified
        # if padding_value is not None:
        #     # Check which windows don't contain padding
        #     no_padding_mask = ~(target_win == padding_value).any(axis=(2, 3))
        # else:
        #     # If no padding_value specified, all windows are valid
        #     no_padding_mask = np.ones((num_trials, timesteps - T + 1), dtype=bool)

        # print(target_win.shape)

        # Center around random color change AND exclude padded windows
        mask_center = (target_win[:, :, -k, change_idx] == 1) # & no_padding_mask
        states_win = np.moveaxis(states_win[mask_center], [2,], [1,])
        sample_win = np.moveaxis(sample_win[mask_center], [2,], [1,])
        target_win = np.moveaxis(target_win[mask_center], [2,], [1,])

        # timestep_win = timestep_win[mask_center]
        order_counts = np.cumsum(mask_center, axis=1) - 1
        order_vector = order_counts[mask_center].tolist()

        return states_win, sample_win, target_win, order_vector

    dict_hz_model_change_order_dfs = {}

    for hz, df_hz in df_data.groupby("Hazard Rate"):
        _dict_model_change_order_dfs = {}
        _dict_model_order = {}
        for _exp_id, _states_first_M_zscore in dict_model_first_states_all.items():
            _states_win, _sample_win, _target_win, _order = get_ordered_sliding_window(
                _states_first_M_zscore,
                samples[:, :M],
                targets[:, :M],
                df_hz,
                T,
                color_change_index,
                change_idx=change_idx,
                padding_value=padding_value,
            )
            _dict_model_order[_exp_id] = _order
            _model_stat_units = dict_model_stat_units_all["hz"][_exp_id]
            _stat_units = list(_model_stat_units.keys())

            _unit_dfs = []
            for i, _unit in enumerate(_stat_units):
                _unit_activity = _states_win[:, :, _unit]
                _df_unit = pd.DataFrame(_unit_activity)
                _df_unit = _df_unit.assign(**{
                    "unit": _unit,
                    "order": [f"Change {_i+1}" for _i in _order],
                })

                _unit_dfs.append(_df_unit)

            _dict_model_change_order_dfs[_exp_id] = pd.concat(_unit_dfs).melt(
                id_vars=[
                    'unit',
                    'order',
                ],
                var_name='Timestep',
                value_name='Value',
            )

        dict_hz_model_change_order_dfs[hz] = (_dict_model_change_order_dfs, _dict_model_order)
        # change_idcs = [np.where(order == i)[0] for i in range(max_N)]
        # dict_model_change_order_means[_exp_id] = np.stack([
        #     states_win[idcs].mean(axis=0) for idcs in change_idcs
        # ])
    return dict_hz_model_change_order_dfs, get_ordered_sliding_window


@app.cell
def _(dict_hz_model_change_order_dfs):
    dict_hz_model_change_order_dfs
    return


@app.cell(hide_code=True)
def _(T, exp_ids, plt, sns):
    def plot_ordered_change_activity(
        dict_cond_model_change_dfs,
        dict_model_stat_units,
        exp_ids=exp_ids,
        x="Timestep",
        y="Value",
        hue="order",
        cond_order=None,
        stat=None,
        palette=None,
        change_labels=None,
        figsize=(8, 6),
        title=None,
        change_idx=4,
        base_title=None,
        label_vline="Color Change",
    ):
        base_title = title
        if cond_order is None:
            cond_order = list(dict_cond_model_change_dfs.keys())
        num_conds = len(cond_order)
        for exp_id in exp_ids:
            print(exp_id, exp_ids)
            # df_model_change = dict_cond_model_change_dfs[exp_id]
            model_stat_units = dict_model_stat_units[exp_id]
            stat_units = list(model_stat_units.keys())

            fig = plt.figure(figsize=figsize)
            fig, axes = plt.subplots(
                len(stat_units),
                num_conds,
                figsize=figsize,
                # figsize=(figsize[0], figsize[1] * num_units),
                sharex='all',
                sharey='row',
            )
            for i, unit in enumerate(stat_units):
                for j, cond in enumerate(cond_order):
                    dict_cond, _ = dict_cond_model_change_dfs[cond]
                # for j, (cond, (dict_cond, _)) in enumerate(dict_cond_model_change_dfs.items()):
                    # Create figure with n_cond rows and 2 columns
                    df_model_change = dict_cond[exp_id]
                    if len(stat_units) == 1:
                        ax = axes[j]
                    else:
                        ax = axes[i, j]
                    ax.axvline(
                        change_idx, 
                        linestyle="--", 
                        color="gray", 
                        label=label_vline,
                    )

                    ax = sns.lineplot(
                        data=df_model_change[
                            (df_model_change["unit"] == unit) &
                            (df_model_change["order"].isin(change_labels[cond]))
                        ],
                        x=x,
                        y=y,
                        hue=hue,
                        palette=palette,
                        errorbar="ci",
                        ax=ax,
                        # legend=j == 0, #num_conds - 1,
                        legend=j == num_conds - 1,
                    )
                    # if j == 0:
                    #     # ax.set_ylabel(f"{model_stat_units[unit].split(" - ")[-1]} Mean Activity")
                    #     ax.set_ylabel(f"Mean {model_stat_units[unit].split(" ")[-2]} Unit Activity")
                    # else:
                    #     ax.set_ylabel("")
                    ax.set_ylabel(f"Mean {model_stat_units[unit].split(" ")[-2]} Unit Activity")

                    if i == 0:
                        name_cond = cond if stat is None else f"{cond} {stat}"
                        ax.set_title(f"{name_cond} Activity")

                    if j == num_conds - 1:
                    # if j == 0:
                        # sns.move_legend(
                        #     ax,
                        #     "center left",
                        #     frameon=False,
                        #     bbox_to_anchor=(-0.2, 0.5),
                        #     fancybox=True,
                        #     # bbox_to_anchor=(1, 1),
                        #     title=None,
                        # )
                        legend = ax.legend(
                            loc='center left',
                            bbox_to_anchor=(-0.4575, 0.5),  # Negative x puts it to the left
                            # bbox_to_anchor=(1, 0.5),  # Negative x puts it to the left
                            frameon=True,
                            fancybox=True,
                               # shadow=True,
                               # title='Functions',
                               # title_fontsize=12
                        )
                        # Center align the legend text
                        for text in legend.get_texts():
                            text.set_ha('center')  # Center horizontally
                    plt.xticks(range(0, T, 5))

            if base_title is None:
                title = exp_id
            else:
                title = " - ".join([base_title, exp_id])

            # fig.suptitle(title, fontsize=14)
            plt.tight_layout()
            # plt.tight_layout(h_pad=2.0)
        return fig, axes
        # plt.tight_layout()
        # plt.show()
    return


@app.cell(hide_code=True)
def _(T, exp_ids, plt, sns):
    def plot_ordered_change_activity_separate(
        dict_cond_model_change_dfs,
        dict_model_stat_units,
        exp_ids=exp_ids,
        x="Timestep",
        y="Value",
        hue="order",
        cond_order=None,
        stat=None,
        palette=None,
        change_labels=None,
        figsize=(8, 6),
        title=None,
        change_idx=4,
        base_title=None,
        label_vline="Color Change",
    ):
        base_title = title
        if cond_order is None:
            cond_order = list(dict_cond_model_change_dfs.keys())
        num_conds = len(cond_order)

        all_figs = []
        all_axes = []
        plot_metadata = []  # Store info about which row each plot belongs to

        for exp_id in exp_ids:
            print(exp_id, exp_ids)
            model_stat_units = dict_model_stat_units[exp_id]
            stat_units = list(model_stat_units.keys())

            # STEP 1: Generate all plots first
            for i, unit in enumerate(stat_units):
                for j, cond in enumerate(cond_order):
                    dict_cond, _ = dict_cond_model_change_dfs[cond]
                    df_model_change = dict_cond[exp_id]

                    # Create individual figure
                    fig, ax = plt.subplots(1, 1, figsize=figsize) # (figsize[0]/num_conds, figsize[1]/len(stat_units)))

                    ax.axvline(
                        change_idx, 
                        linestyle="--", 
                        color="gray", 
                        label=label_vline,
                    )

                    ax = sns.lineplot(
                        data=df_model_change[
                            (df_model_change["unit"] == unit) &
                            (df_model_change["order"].isin(change_labels[cond]))
                        ],
                        x=x,
                        y=y,
                        hue=hue,
                        palette=palette,
                        errorbar="ci",
                        ax=ax,
                        legend=None,
                        # legend=j == num_conds - 1,
                    )

                    ax.set_ylabel(f"{model_stat_units[unit].split(' ')[-2]} Unit Activity")

                    name_cond = cond if stat is None else f"{cond} {stat}"
                    plot_title = f"{name_cond} Activity"
                    if base_title is None:
                        full_title = f"{exp_id} - {plot_title}"
                    else:
                        full_title = f"{base_title} - {exp_id} - {plot_title}"

                    # ax.set_title(full_title)

                    # if j == num_conds - 1:
                    #     legend = ax.legend(
                    #         loc='center left',
                    #         bbox_to_anchor=(1.05, 0.5),
                    #         frameon=True,
                    #         fancybox=True,
                    #     )
                    #     for text in legend.get_texts():
                    #         text.set_ha('center')

                    ax.set_xticks(range(0, T, 5))

                    all_figs.append(fig)
                    all_axes.append(ax)
                    plot_metadata.append({'row': i, 'col': j, 'exp_id': exp_id})

            # STEP 2: Extract actual limits from rendered plots
            # Get shared x-limits (across ALL plots)
            x_min, x_max = float('inf'), float('-inf')
            for ax in all_axes:
                xlim = ax.get_xlim()
                x_min = min(x_min, xlim[0])
                x_max = max(x_max, xlim[1])

            # Get shared y-limits per row
            y_limits_per_row = {}
            for idx, (ax, metadata) in enumerate(zip(all_axes, plot_metadata)):
                row = metadata['row']
                ylim = ax.get_ylim()

                if row not in y_limits_per_row:
                    y_limits_per_row[row] = [ylim[0], ylim[1]]
                else:
                    y_limits_per_row[row][0] = min(y_limits_per_row[row][0], ylim[0])
                    y_limits_per_row[row][1] = max(y_limits_per_row[row][1], ylim[1])

            # STEP 3: Apply shared limits to all plots
            for ax, metadata in zip(all_axes, plot_metadata):
                row = metadata['row']
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_limits_per_row[row][0], y_limits_per_row[row][1])

            # Redraw all figures to apply the new limits
            for fig in all_figs:
                fig.tight_layout()
                fig.canvas.draw()

        return all_figs, all_axes
    return


@app.cell
def _(T, exp_ids, plt, shortened_conditions, sns):
    def plot_ordered_change_activity_rows(
        dict_cond_model_change_dfs,
        dict_model_stat_units,
        exp_ids=exp_ids,
        x="Timestep",
        y="Value",
        hue="order",
        cond_order=None,
        stat=None,
        palette=None,
        change_labels=None,
        figsize=(8, 6),  # Size per subplot
        title=None,
        change_idx=4,
        base_title=None,
        label_vline="Color Change",
        errorbar="ci",
        short=True,
        legend_width_inches=3.5,  # Space to allocate for legend
        bbox_to_anchor=(-1.25, 0.5),
    ):
        base_title = title
        if cond_order is None:
            cond_order = list(dict_cond_model_change_dfs.keys())
        num_conds = len(cond_order)

        all_figs = []
        all_axes = []

        for exp_id in exp_ids:
            # print(exp_id, exp_ids)
            model_stat_units = dict_model_stat_units[exp_id]
            stat_units = list(model_stat_units.keys())

            for i, unit in enumerate(stat_units):
                # Calculate total figure width: subplot widths + legend space
                total_width = figsize[0] * num_conds - 1 + (legend_width_inches if num_conds > 1 else 0)

                # Create figure with calculated total width
                fig = plt.figure(figsize=(total_width, figsize[1]))

                # Use GridSpec for precise spacing control
                if num_conds > 1:
                    # Width ratios: each subplot gets figsize[0], legend space gets legend_width_inches
                    # width_ratios = [figsize[0]] + [legend_width_inches] + [figsize[0]] * (num_conds - 1)
                    width_ratios = [figsize[0]] + [legend_width_inches] + [figsize[0]] * (num_conds - 1)
                    gs = fig.add_gridspec(1, num_conds + 1, width_ratios=width_ratios, 
                                         hspace=0, wspace=0)

                    # Create subplots, skipping the middle space (index 1)
                    axes = []
                    axes.append(fig.add_subplot(gs[0, 0]))
                    for j in range(1, num_conds):
                        axes.append(fig.add_subplot(gs[0, j + 1], sharey=axes[0], sharex=axes[0]))
                else:
                    gs = fig.add_gridspec(1, 1)
                    axes = [fig.add_subplot(gs[0, 0])]

                for j, cond in enumerate(cond_order):
                    dict_cond, _ = dict_cond_model_change_dfs[cond]
                    df_model_change = dict_cond[exp_id]

                    ax = axes[j]

                    ax.axvline(
                        change_idx, 
                        linestyle="--", 
                        color="gray", 
                        label=label_vline,
                    )

                    ax = sns.lineplot(
                        data=df_model_change[
                            (df_model_change["unit"] == unit) &
                            (df_model_change["order"].isin(change_labels[cond]))
                        ],
                        x=x,
                        y=y,
                        hue=hue,
                        palette=palette,
                        errorbar=errorbar,
                        ax=ax,
                        legend=j == num_conds - 1,
                    )
                    ax.set_ylabel(f"{model_stat_units[unit].split(' ')[-2]} Unit Activity")

                    stat_title = stat if not short else shortened_conditions[stat]
                    name_cond = cond if stat is None else f"{cond} {stat_title}"
                    ax.set_title(f"{name_cond} Trials")

                    if j == num_conds - 1:
                        ax.set_ylabel("") 
                        legend = ax.legend(
                            loc='center left',
                            bbox_to_anchor=bbox_to_anchor,
                            frameon=True,
                            # fancybox=True,
                        )
                        for text in legend.get_texts():
                            text.set_ha('center')
                    else:
                        ax.set_ylabel(f"{model_stat_units[unit].split(' ')[-2]} Unit Activity")

                    plt.xticks(range(0, T, 5))

                plt.tight_layout(pad=1.0)

                all_figs.append(fig)
                all_axes.append(axes)

        return all_figs, all_axes
    return (plot_ordered_change_activity_rows,)


@app.cell
def _():
    return


@app.cell(column=3)
def _():
    dict_model_stat_units_all = {
        'hz': {
        #     'san-4602': {
        #         1: 'Unit 1 - Hidden 1',
        #         17: 'Unit 17 - Cell 1'
        #     },
        #     'san-4605': {
        #         8: 'Unit 8 - Hidden 8',
        #         24: 'Unit 24 - Cell 8'},
        #     'san-4604': {
        #         15: 'Unit 15 - Hidden 15',
        #         31: 'Unit 31 - Cell 15'},
        #     'san-4603': {
        #         6: 'Unit 6 - Hidden 6',
        #         22: 'Unit 22 - Cell 6'},
        #     'san-4606': {
        #         11: 'Unit 11 - Hidden 11',
        #         27: 'Unit 27 - Cell 11'},
        #     'san-4601': {
        #         1: 'Unit 1 - Hidden 1',
        #         17: 'Unit 17 - Cell 1'
        #         },
        #       'san-4615': {4: 'Unit 4 - Hidden 4', 20: 'Unit 20 - Cell 4'},
        #       'san-4616': {5: 'Unit 5 - Hidden 5', 21: 'Unit 21 - Cell 5'},
        #       'san-4618': {4: 'Unit 4 - Hidden 4', 20: 'Unit 20 - Cell 4'},
        #       'san-4617': {0: 'Unit 0 - Hidden 0', 16: 'Unit 16 - Cell 0'},
        },
        'cont': {
            'san-4604': {
                11: 'Unit 11 - Hidden 11',
                27: 'Unit 27 - Cell 11'
            },
            'san-4603': {
                0: 'Unit 0 - Hidden 0',
                16: 'Unit 16 - Cell 0',
            },
            'san-4606': {
                # 11: 'Unit 11 - Hidden 11',
                12: 'Unit 12 - Hidden 12',
                28: 'Unit 28 - Cell 12',
            },
            'san-4601': {
                5: 'Unit 5 - Hidden 5',
                21: 'Unit 21 - Cell 5'
            },
            'san-4615': {12: 'Unit 12 - Hidden 12', 28: 'Unit 28 - Cell 12'},
            'san-4618': {9: 'Unit 9 - Hidden 9', 25: 'Unit 25 - Cell 9'},
        }
    }

    exp_ids_all = list(dict_model_stat_units_all["hz"].keys())
    exp_ids_all
    return (dict_model_stat_units_all,)


@app.cell
def _(M, dict_model_data, dir_data_base, exp_no_cont, model_name, np, stats):
    dict_model_data_all = {}
    dict_model_states_all = {}
    dict_model_first_states_all = {}
    path_models_all = {
        path.stem: path for path in (dir_data_base / model_name).iterdir()
    }

    for _exp_id, _path_model in path_models_all.items():
        print(_exp_id)
        if _exp_id in exp_no_cont:
            continue
        _model_data = dict_model_data[_exp_id] = np.load(
            str(_path_model),
            allow_pickle=True,
        )
        dict_model_states_all[_exp_id] = _model_data["states"] #[np.arange(8192), -M:df_data["length"].values - 1]

        dict_model_first_states_all[_exp_id] = stats.zscore(
            np.concat(
                [
                    _model_data["hiddens"][:, :M],
                    _model_data["cells"][:, :M],
                ],
                axis=-1,
            ),
            axis=(0, 1),
            ddof=1,
        )
    return dict_model_first_states_all, dict_model_states_all


@app.cell
def _(
    M,
    df_data,
    dict_model_first_states_all,
    dict_model_stat_units_all,
    np,
    samples,
    targets,
):
    def get_activity_difference_around_criterion(states, samples, targets, df_selected, tau):
        # Select relevant data
        states = states[df_selected.index]
        samples = samples[df_selected.index]
        targets = targets[df_selected.index]

        num_trials, timesteps, num_features = states.shape

        # Find where criterion is met
        # Assuming targets has shape (num_trials, timesteps, k_dim, feature_dim)
        criterion_mask = (targets[:, :, -4:-2] == 1).any(axis=-1) # & (samples[:, :, 2:] != 127).any(axis=-1)

        # Get indices where criterion is met
        trial_indices, time_indices = np.where(criterion_mask)
        # print(trial_indices.shape)

        # Filter out cases where we can't look tau steps before or after
        valid_mask = (time_indices >= tau) & (time_indices < timesteps - tau)
        trial_indices = trial_indices[valid_mask]
        time_indices = time_indices[valid_mask]

        # Calculate differences
        states_before = states[trial_indices, time_indices - tau]
        states_after = states[trial_indices, time_indices + tau]
        diff_states = states_after - states_before

        # Store criterion indices for reference
        criterion_indices = list(zip(trial_indices, time_indices))

        return diff_states, criterion_indices


    def get_activity_difference_during_zero_criterion(states, targets, df_selected, tau):
        from numpy.lib.stride_tricks import as_strided

        # Select relevant data
        states = states[df_selected.index]
        targets = targets[df_selected.index]

        num_trials, timesteps, num_features = states.shape

        # Find where criterion is 0
        zero_mask = (targets[:, :, -4:] == 0).all(axis=-1)

        # Use as_strided for maximum efficiency (same as sliding_window_view but more direct)
        stride_time = zero_mask.strides[1]
        zero_windows = as_strided(
            zero_mask,
            shape=(num_trials, timesteps - tau + 1, tau),
            strides=(zero_mask.strides[0], stride_time, stride_time)
        )

        # Find valid sequences (all tau positions are True)
        valid_sequences = zero_windows.all(axis=2)

        # Get flat indices for efficient extraction
        trial_indices, window_indices = np.where(valid_sequences)

        if len(trial_indices) == 0:
            # No valid sequences found
            return np.empty((0, num_features)), []

        # Vectorized extraction of start and end states
        # We can reshape states for easier indexing
        flat_idx_start = trial_indices * timesteps + window_indices
        flat_idx_end = trial_indices * timesteps + (window_indices + tau - 1)

        states_flat = states.reshape(-1, num_features)
        states_start = states_flat[flat_idx_start]
        states_end = states_flat[flat_idx_end]

        # Calculate differences
        diff_states = (states_end - states_start) / (tau - 1)

        # Create sequence indices
        sequence_indices = list(zip(trial_indices.tolist(), window_indices.tolist()))

        print(f"Found {len(sequence_indices)} sequences of {tau} consecutive zeros")

        return diff_states, sequence_indices

    # color_change_index_bounce = 4
    tau_change = 5
    tau_no_change = 2
    # change_idx = 5
    dict_model_change_state_diffs = {}

    for _exp_id, _states_first_M_zscore in dict_model_first_states_all.items():
        print(_exp_id)
        _model_stat_units = dict_model_stat_units_all["hz"][_exp_id]
        _stat_units = list(_model_stat_units.keys())

        diff_states_change, _ = get_activity_difference_around_criterion(
            _states_first_M_zscore,
            samples[:, :M],
            targets[:, :M],
            df_data,
            tau_change,
            # color_change_index_bounce,
            # change_idx=change_idx,
            # padding_value=padding_value,
        )
        # dict_model_change_state_diffs[_exp_id] = {
        #     unit: diff_states_change[:, unit] for unit in _stat_units
        # }

        _diff_states_no_change, _ = get_activity_difference_during_zero_criterion(
            _states_first_M_zscore,
            # samples[:, :M],
            targets[:, :M],
            df_data,
            tau_no_change,
            # change_idx=change_idx,
            # padding_value=padding_value,
        )

        dict_model_change_state_diffs[_exp_id] = {
            "hidden" if i == 0 else "cell": {
                "step": np.abs(diff_states_change[:, unit]).mean(),
                "decay": np.abs(_diff_states_no_change[:, unit]).mean(),
            }
            for i, unit in enumerate(_stat_units)
        }
    return (
        dict_model_change_state_diffs,
        get_activity_difference_during_zero_criterion,
        tau_change,
        tau_no_change,
    )


@app.cell
def _(dict_model_change_state_diffs, pd):
    df_model_change_state_diffs = pd.DataFrame.from_dict(
        {_exp_id: pd.json_normalize(_exp_data).iloc[0] 
         for _exp_id, _exp_data in dict_model_change_state_diffs.items()}, 
        orient='index'
    )
    df_model_change_state_diffs.columns = [
        '_'.join(col.split('.')) for col in df_model_change_state_diffs.columns
    ]  # Flatten column names
    df_model_change_state_diffs
    return (df_model_change_state_diffs,)


@app.cell
def _(df_model_change_state_diffs, figsize, plt, sns):
    def plot_decay_vs_step(df, figsize=(12, 5),
                           point_size=100, alpha=0.7, colors=None):
        figs = []
        axes = []
        x_col_type = "step"
        y_col_type = "decay"


        max_x = max([df[f'{_state}_{x_col_type}'].max() for _state in ["hidden", "cell"]]) * 1.1
        max_y = max([df[f'{_state}_{y_col_type}'].max() for _state in ["hidden", "cell"]]) * 1.1

        for _state in ["hidden", "cell"]: 
            # Calculate the maximum value across both x and y for this state
            x_col = f'{_state}_{x_col_type}'
            y_col = f'{_state}_{y_col_type}'

            # max_x = max(df[x_col].max(), df[y_col].max())

            # Plot
            fig, ax = plt.subplots(figsize=figsize)
            sns.scatterplot(
                data=df, 
                x=x_col, 
                y=y_col,
                ax=ax,
                s=point_size,
                alpha=alpha,
                color=colors,
                edgecolor='white',
                linewidth=0.5,
            )
            ax.set_xlabel(f'Step Size')
            ax.set_ylabel(f'Activity Decay')
            ax.set_title(f'All {_state.title()} Unit\nActivity Profiles')

            # Set limits: min=0, max=same for both axes
            ax.set_xlim(0, max_x)
            ax.set_ylim(0, max_y)

            # # Optional: Add diagonal reference line
            # ax.plot([0, max_value], [0, max_value], 'k--', alpha=0.3, linewidth=1)

            # # Optional: Make the plot square
            # ax.set_aspect('equal', adjustable='box')

            plt.tight_layout()
            plt.show()

            figs.append(fig)
            axes.append(ax)

        return figs, axes

    # # Usage example:
    # figs, axes = plot_decay_vs_step(
    #     df_model_change_state_diffs,
    #     figsize=(6, 6),  # Square figure works better with equal axes
    # )

    # Usage example:
    fig, axes = plot_decay_vs_step(
        df_model_change_state_diffs,
        figsize=figsize,
    )
    plt.show()
    return (plot_decay_vs_step,)


@app.cell
def _(
    M,
    df_data,
    dict_model_first_states_all,
    dict_model_stat_units_all,
    get_activity_difference_during_zero_criterion,
    np,
    samples,
    targets,
    tau_change,
    tau_no_change,
):
    def get_activity_difference_around_criterion_no_change(states, samples, targets, df_selected, tau):
        # Select relevant data
        states = states[df_selected.index]
        samples = samples[df_selected.index]
        targets = targets[df_selected.index]

        num_trials, timesteps, num_features = states.shape

        # Find where criterion is met
        # Assuming targets has shape (num_trials, timesteps, k_dim, feature_dim)
        criterion_mask = (
            (targets[:, :, -4] == 1)
            & (targets[:, :, -2] == 0)
        )
    
        # Get indices where criterion is met
        trial_indices, time_indices = np.where(criterion_mask)
        # print(trial_indices.shape)

        # Filter out cases where we can't look tau steps before or after
        valid_mask = (time_indices >= tau) & (time_indices < timesteps - tau)
        trial_indices = trial_indices[valid_mask]
        time_indices = time_indices[valid_mask]

        # Calculate differences
        states_before = states[trial_indices, time_indices - tau]
        states_after = states[trial_indices, time_indices + tau]
        diff_states = states_after - states_before

        # Store criterion indices for reference
        criterion_indices = list(zip(trial_indices, time_indices))

        return diff_states, criterion_indices
    
    dict_model_no_change_bounce_state_diffs = {}

    for _exp_id, _states_first_M_zscore in dict_model_first_states_all.items():
        print(_exp_id)
        _model_stat_units = dict_model_stat_units_all["hz"][_exp_id]
        _stat_units = list(_model_stat_units.keys())

        _diff_states_no_change_bounce, _ = get_activity_difference_around_criterion_no_change(
            _states_first_M_zscore,
            samples[:, :M],
            targets[:, :M],
            df_data,
            tau_change,
            # color_change_index_bounce,
            # change_idx=change_idx,
            # padding_value=padding_value,
        )
        # dict_model_change_state_diffs[_exp_id] = {
        #     unit: diff_states_change[:, unit] for unit in _stat_units
        # }

        _diff_states_no_change, _ = get_activity_difference_during_zero_criterion(
            _states_first_M_zscore,
            # samples[:, :M],
            targets[:, :M],
            df_data,
            tau_no_change,
            # change_idx=change_idx,
            # padding_value=padding_value,
        )

        dict_model_no_change_bounce_state_diffs[_exp_id] = {
            "hidden" if i == 0 else "cell": {
                "step": np.abs(_diff_states_no_change_bounce[:, unit]).mean(),
                "decay": np.abs(_diff_states_no_change[:, unit]).mean(),
            }
            for i, unit in enumerate(_stat_units)
        }
    return (dict_model_no_change_bounce_state_diffs,)


@app.cell
def _(
    dict_model_no_change_bounce_state_diffs,
    figsize,
    pd,
    plot_decay_vs_step,
    plt,
):
    df_model_no_change_bounce_state_diffs = pd.DataFrame.from_dict(
        {_exp_id: pd.json_normalize(_exp_data).iloc[0] 
         for _exp_id, _exp_data in dict_model_no_change_bounce_state_diffs.items()}, 
        orient='index'
    )
    df_model_no_change_bounce_state_diffs.columns = [
        '_'.join(col.split('.')) for col in df_model_no_change_bounce_state_diffs.columns
    ]  # Flatten column names
    df_model_no_change_bounce_state_diffs

    # Usage example:
    plot_decay_vs_step(
        df_model_no_change_bounce_state_diffs,
        figsize=figsize,
    )
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
