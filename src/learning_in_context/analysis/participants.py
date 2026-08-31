"""Participant-response pipeline for the human bouncing-ball task.

Turns the raw jsPsych exports of the online experiment into the three
artifacts the behavioral figures consume:

- ``participant_stats_filtered.parquet`` — one row per surviving participant,
  carrying per-trial-type accuracy, response time and confidence summaries.
- ``participant_cwc.parquet`` — long table of per-trial confidence-weighted
  choice (CWC) for those participants.
- ``participant_counts.json`` — cohort bookkeeping: participants loaded,
  survivors after each filter stage, and the final sample size.

The pipeline runs in one direction: load the trial metadata, relabel catch
trials whose color settles too early, load every participant's responses
against that metadata, summarize each participant, apply six exclusion
filters in order, and compute CWC for whoever is left.

Run it as a module to write the artifacts::

    python -m learning_in_context.analysis.participants \
        --output-dir data/cache/participants \
        --raw-dir data/raw/hbb_v3_2_2 \
        --participant-dataset-dir data/cache/model_states/participant_dataset
"""

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COL_PARTICIPANT_ID = 'Participant ID'
COL_VIDEO_ID = 'Video ID'
COL_BLOCK = 'Dataset Block'
COL_BLOCK_VIDEO = 'Dataset Block Video'

BALL_COLORS = ('red', 'green', 'blue')
IDX_TO_COLOR = {idx + 1: color for idx, color in enumerate(BALL_COLORS)}
COLOR_TO_IDX = {color: idx + 1 for idx, color in enumerate(BALL_COLORS)}

TRIAL_TYPES = ('catch', 'straight', 'bounce', 'nonwall')
RESPONSE_COLUMNS = (
    'rt',
    'response',
    'trial_index',
    'internal_node_id',
    'time_elapsed',
    'correct_response',
    'participant_id',
    'study_id',
    'slider_start',
)

# Statuses kept when reading the recruitment-platform demographics export, and
# the status excluded from analysis outright.
VALID_PARTICIPANT_STATUS = ('APPROVED', 'AWAITING REVIEW', 'TIMED-OUT', 'RETURNED')
EXCLUDED_PARTICIPANT_STATUS = 'RETURNED'

# Catch trials whose ball color stops changing this many timesteps (or fewer)
# before the video ends give the answer away, so they are scored separately.
MIN_CATCH_TIMESTEPS = 8
CATCH_FAST_LABEL = 'Catch-Fast'

# Exclusion-filter cutoffs.
MIN_TRIAL_FRACTION = 0.8
MAX_MISSED_FRACTION = 0.2
RMS_SLIDER_CUTOFF = 0.2
VAR_RMS_SLIDER_CUTOFF = 0.03
SLIDER_VAR_CUTOFF = 0.0
SLIDER_MEDIAN_CUTOFF = 0.0
CATCH_ACCURACY_CUTOFF = 0.8

# Adjusted-confidence spread below which a participant's confidence reports
# carry no usable signal and every trial is scored at full confidence.
CONFIDENCE_RANGE_CUTOFF = 70

ARTIFACT_STATS = 'participant_stats_filtered.parquet'
ARTIFACT_CWC = 'participant_cwc.parquet'
ARTIFACT_COUNTS = 'participant_counts.json'


# ---------------------------------------------------------------------------
# Video-indexed helpers
# ---------------------------------------------------------------------------

def filter_video_indexed_data(
    df_video_indexed_data: pd.DataFrame,
    df_filter: pd.DataFrame,
    two_way: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict a video-indexed frame to the videos present in another.

    Args:
        df_video_indexed_data: Frame indexed by ``Video ID`` to restrict.
        df_filter: Frame indexed by ``Video ID`` supplying the videos to keep.
        two_way: Also restrict ``df_filter`` to the same intersection and
            return both frames, aligned row for row.

    Returns:
        The restricted data frame, or both restricted frames when ``two_way``.

    Raises:
        ValueError: If either frame is not indexed by ``Video ID``.
    """
    if df_video_indexed_data.index.name != COL_VIDEO_ID:
        raise ValueError(
            f"df_video_indexed_data must have an index named '{COL_VIDEO_ID}', "
            f"got '{df_video_indexed_data.index.name}'"
        )
    if df_filter.index.name != COL_VIDEO_ID:
        raise ValueError(
            f"df_filter must have an index named '{COL_VIDEO_ID}', "
            f"got '{df_filter.index.name}'"
        )

    df_filtered = df_video_indexed_data.loc[
        df_video_indexed_data.index.intersection(df_filter.index)
    ]

    if not two_way:
        return df_filtered

    return df_filtered, df_filter.loc[df_filter.index.intersection(df_filtered.index)]


# ---------------------------------------------------------------------------
# Response and confidence utilities
# ---------------------------------------------------------------------------

def response_accuracy_vector(
    df: pd.DataFrame,
    column: str = 'response',
    df_comparison: pd.DataFrame | None = None,
    column_comparison: str = 'correct_response',
) -> pd.Series:
    """Per-trial correctness of a response column against the correct answer.

    Args:
        df: Frame holding the participant's responses.
        column: Response column in ``df``.
        df_comparison: Frame holding the correct answers; defaults to ``df``.
        column_comparison: Correct-answer column in ``df_comparison``.

    Returns:
        Boolean series, one entry per trial.

    Raises:
        ValueError: If either column is missing.
    """
    if df_comparison is None:
        df_comparison = df

    if column not in df.columns:
        raise ValueError(f"The input DataFrame must contain '{column}'")
    if column_comparison not in df_comparison.columns:
        raise ValueError(
            f"The comparison DataFrame must contain '{column_comparison}'"
        )

    return df[column].values == df_comparison[column_comparison]


def response_accuracy(*args, **kwargs) -> float:
    """Mean of :func:`response_accuracy_vector`, ignoring missing responses."""
    return response_accuracy_vector(*args, **kwargs).mean(skipna=True)


def compute_confidence(
    df: pd.DataFrame,
    column: str = 'slider_end',
    column_comparison: str = 'slider_start',
    average: bool = False,
    median: bool = False,
    var: bool = False,
) -> float | np.ndarray:
    """Confidence reports rescaled from slider units to the unit interval.

    Args:
        df: Frame holding the confidence slider columns.
        column: Column holding the slider's final position.
        column_comparison: Column holding the slider's starting position.
        average: Return the mean confidence instead of the per-trial values.
        median: Return the median confidence.
        var: Return the variance of the confidences.

    Returns:
        The requested reduction, or the per-trial confidences when no
        reduction is requested.

    Raises:
        ValueError: If a slider column is missing, or more than one reduction
            is requested.
    """
    if column not in df.columns:
        raise ValueError(f"The input DataFrame must contain '{column}'")
    if column_comparison not in df.columns:
        raise ValueError(f"The input DataFrame must contain '{column_comparison}'")
    if sum(bool(val) for val in (average, median, var)) > 1:
        raise ValueError(
            "Cannot have 'average', 'median', or 'var' equal True at same time"
        )

    values = df[column].dropna()

    if average:
        return values.values.mean() / 100
    if median:
        return values.median() / 100
    if var:
        return (values.values / 100).var()
    return values.values / 100


def infer_video_internal_node_id_from_response(node_id: str) -> str:
    """Map a response trial's jsPsych node id to its stimulus trial's node id.

    The stimulus and the response it elicited share a timeline position and
    differ only in the first component of the node id's final segment.
    """
    parts = node_id.split('-')
    parts[-1] = '.'.join(['1', parts[-1].split('.')[-1]])
    return '-'.join(parts)


def infer_block_from_internal_node_id(node_id: str) -> int:
    """Read the experiment block index out of a jsPsych node id."""
    return int(node_id.split('-')[2].split('.')[0])


def ms_to_min(ms):
    """Convert milliseconds to whole minutes, rounded to nearest."""
    return np.round(ms / 1000 / 60).astype(int)


# ---------------------------------------------------------------------------
# Trial metadata
# ---------------------------------------------------------------------------

def load_trial_metadata(
    path_dataset: Path,
    col_block: str = COL_BLOCK,
    col_video: str = COL_BLOCK_VIDEO,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the experiment's trial metadata table and dataset parameters.

    Args:
        path_dataset: Dataset directory holding ``trial_meta.csv``,
            ``dataset_meta.pkl`` and the ``videos/`` tree.
        col_block: Column naming each trial's block.
        col_video: Column naming each trial's video within its block.

    Returns:
        The trial metadata indexed by ``Video ID``, and the dataset parameter
        dictionary.
    """
    path_trial_metadata = path_dataset / 'trial_meta.csv'
    path_dataset_metadata = path_dataset / 'dataset_meta.pkl'

    df_metadata = pd.read_csv(str(path_trial_metadata), index_col=0)
    with open(str(path_dataset_metadata), 'rb') as f:
        dict_metadata = pickle.load(f)

    # Where each trial's rendered video and its sampled trajectory live
    df_metadata['Dir Video'] = df_metadata.apply(
        lambda row: f'{path_dataset}/videos/block_{row[col_block]}/video_{row[col_video]}',
        axis=1,
    )
    df_metadata['final_color_response'] = (
        df_metadata['Final Color'].map(COLOR_TO_IDX).values
    )

    # Ordered condition labels, derived from the two generative parameters
    hazard_rates = np.sort(df_metadata['PCCNVC'].unique())
    contingencies = np.sort(df_metadata['PCCOVC'].unique())
    df_metadata['Hazard Rate'] = pd.Categorical(
        df_metadata['PCCNVC'].apply(
            lambda hz: (
                'Low' if np.isclose(hz, hazard_rates[0])
                else 'High' if np.isclose(hz, hazard_rates[1])
                else 'Unknown'
            )
        ),
        categories=['Low', 'High'],
    )
    df_metadata['Contingency'] = pd.Categorical(
        df_metadata['PCCOVC'].apply(
            lambda cont: (
                'Low' if np.isclose(cont, contingencies[0])
                else 'Medium' if np.isclose(cont, contingencies[1])
                else 'High' if np.isclose(cont, contingencies[2])
                else 'Unknown'
            )
        ),
        categories=['Low', 'Medium', 'High'],
    )

    # +1 when the correct answer is a color change, -1 when it is not
    df_metadata['correct_coded'] = 1
    df_metadata.loc[
        df_metadata['color_entered'] == df_metadata['correct_response'], 'correct_coded'
    ] = -1

    df_metadata = df_metadata.rename(columns={'idx': 'idx_trial'})
    df_metadata.index.name = COL_VIDEO_ID

    return df_metadata, dict_metadata


def compute_catch_timesteps(
    df_trial_metadata: pd.DataFrame,
    trial_column: str = 'trial',
    video_dir_column: str = 'Dir Video',
) -> pd.Series:
    """Count how long each catch trial's ball color held its final value.

    For every catch trial, reads the sampled trajectory and returns the number
    of timesteps between the last color change and the end of the video. Small
    counts mean the answer was visible well before the video ended.

    Args:
        df_trial_metadata: Trial metadata indexed by ``Video ID``; rows other
            than catch trials are ignored when a trial-type column is present.
        trial_column: Column naming each trial's type.
        video_dir_column: Column holding each trial's video directory.

    Returns:
        Timesteps since the color settled, indexed by ``Video ID``.
    """
    df_catch = df_trial_metadata
    if trial_column in df_catch.columns:
        df_catch = df_catch[df_catch[trial_column] == 'Catch']

    timesteps = {}
    for video_id, dir_video in df_catch[video_dir_column].items():
        path_samples = sorted(Path(dir_video).glob('*_samples.csv'))[0]
        colors = pd.read_csv(path_samples)[['r', 'g', 'b']].to_numpy()

        final_color = colors[-1]
        last_idx = len(colors) - 1
        idx_differing = np.where((colors[:-1] != final_color).any(axis=1))[0]
        timesteps[video_id] = last_idx - idx_differing[-1]

    series = pd.Series(timesteps, dtype=int)
    series.index.name = df_trial_metadata.index.name
    return series


def reclassify_fast_catch_trials(
    df_trial_metadata: pd.DataFrame,
    catch_timesteps: pd.Series,
    min_timesteps: int = MIN_CATCH_TIMESTEPS,
    trial_column: str = 'trial',
) -> pd.DataFrame:
    """Relabel catch trials whose ball color settled too early to score.

    A catch trial whose color stops changing ``min_timesteps`` or fewer steps
    before the video ends shows the participant the answer, so it is relabeled
    ``Catch-Fast`` and drops out of catch-trial accuracy.

    Args:
        df_trial_metadata: Trial metadata indexed by ``Video ID``.
        catch_timesteps: Timesteps since the color settled, per catch trial.
        min_timesteps: Largest settle time still counted as too fast.
        trial_column: Column naming each trial's type.

    Returns:
        A copy of the metadata with the fast catch trials relabeled.

    Raises:
        ValueError: If the trial-type column is missing.
    """
    if trial_column not in df_trial_metadata.columns:
        raise ValueError(
            f"df_trial_metadata must contain a '{trial_column}' column"
        )

    idx_fast = catch_timesteps.index[catch_timesteps <= min_timesteps]
    idx_fast = df_trial_metadata.index.intersection(idx_fast)

    df_reclassified = df_trial_metadata.copy()
    df_reclassified.loc[idx_fast, trial_column] = CATCH_FAST_LABEL
    return df_reclassified


# ---------------------------------------------------------------------------
# jsPsych response loader
# ---------------------------------------------------------------------------

def load_participant_responses(
    participant_id: str,
    path_participant_data_all: Path,
    df_dataset_metadata: pd.DataFrame,
    validate: bool = True,
    columns_to_keep: tuple[str, ...] = RESPONSE_COLUMNS,
    dir_videos: Path = Path('videos'),
    trial_types: tuple[str, ...] = TRIAL_TYPES,
    col_block: str = COL_BLOCK,
    col_video: str = COL_BLOCK_VIDEO,
) -> dict[str, Any]:
    """Read one participant's jsPsych export into video-indexed responses.

    The export interleaves stimulus, response and confidence trials. Each
    response is paired with the confidence report that followed it and with
    the video that preceded it, then keyed by the ``Video ID`` that video
    carries in the experiment metadata.

    Args:
        participant_id: Participant directory name under the responses root.
        path_participant_data_all: Root directory of participant exports.
        df_dataset_metadata: Trial metadata indexed by ``Video ID``.
        validate: Check the recorded answers against the video filenames.
        columns_to_keep: jsPsych fields carried through to the response frame.
        dir_videos: Base directory the recorded video paths are rebuilt
            against; only the block and video components of those paths are
            read back, so the base need not exist.
        trial_types: Trial types split out into their own response frames.
        col_block: Column naming each trial's block.
        col_video: Column naming each trial's video within its block.

    Returns:
        Dictionary with the participant's device, the raw trial table, the
        per-block response frames, and a ``responses`` mapping of trial type to
        video-indexed responses.
    """
    dict_participant_data = {}

    # Exports accumulate one file per session; the largest holds the full run
    path_json = max(
        (path_participant_data_all / participant_id).iterdir(),
        key=lambda path: (path.stat().st_size, path.name),
    )
    with open(path_json) as f:
        json_data = json.load(f)

    df_all_trials = pd.DataFrame(json_data['trials'])
    dict_participant_data['device'] = json_data['app_platform']
    dict_participant_data['all_trials'] = df_all_trials

    # Response trials carry the correct answer; confidence trials carry a
    # starting slider position
    df_confidence = df_all_trials.loc[~df_all_trials['slider_start'].isna()]
    df_response = df_all_trials.loc[~df_all_trials['correct_response'].isna()]
    df_response = df_response[list(columns_to_keep)].copy()
    df_confidence = df_confidence[list(columns_to_keep)]

    # The confidence report immediately follows the response it qualifies
    for idx in df_response.index:
        if idx + 1 in df_confidence.index:
            df_response.at[idx, 'slider_start'] = df_confidence.at[idx + 1, 'slider_start']
            df_response.at[idx, 'slider_end'] = df_confidence.at[idx + 1, 'response']

    # Recover the video each response refers to from its stimulus trial
    video_node_ids = df_response['internal_node_id'].apply(
        infer_video_internal_node_id_from_response
    )
    paths_video = df_all_trials[
        df_all_trials['internal_node_id'].isin(video_node_ids)
    ].stimulus.apply(
        lambda stimulus: str(
            (dir_videos / '/'.join(stimulus[0].split('/')[2:]))
            .resolve(strict=False)
            .absolute()
        )
    ).to_list()
    df_response['Path Video'] = paths_video

    def _video_index(path: str) -> int:
        stem = Path(path).parent.stem
        return -2 if stem == 'walkthrough' else -1 if stem == 'examples' else int(
            stem.split('_')[-1]
        )

    def _block_index(path: str) -> int:
        stem = Path(path).parent.stem
        if stem == 'walkthrough':
            return -2
        if stem == 'examples':
            return -1
        return int(Path(path).parent.parent.stem.split('_')[-1])

    df_response[col_video] = df_response['Path Video'].apply(_video_index)
    df_response[col_block] = df_response['Path Video'].apply(_block_index)

    # Colors are recorded zero-indexed and used one-indexed everywhere else
    for column in ('rt', 'response', 'correct_response'):
        if column not in columns_to_keep:
            continue
        df_response[column] = df_response[column].astype(pd.Int64Dtype())
        if column in ('response', 'correct_response'):
            df_response[column] = df_response[column] + 1

    if validate:
        if not all(
            value in IDX_TO_COLOR if pd.notna(value) else True
            for value in df_response.response.unique()
        ):
            raise ValueError(
                f"Participant '{participant_id}' recorded a response outside "
                f'the valid colors {tuple(IDX_TO_COLOR.values())}'
            )
        colors_shown = [Path(path).stem.split('_')[-1] for path in paths_video]
        colors_correct = [IDX_TO_COLOR[value] for value in df_response.correct_response]
        if colors_shown != colors_correct:
            raise ValueError(
                f"Participant '{participant_id}' has recorded answers that "
                'disagree with the videos they were shown'
            )

    # Split the practice phases off from the experiment blocks
    idx_walkthrough = [
        i for i, path in enumerate(paths_video) if 'walkthrough' in path.split('/')
    ]
    idx_examples = [
        i for i, path in enumerate(paths_video) if 'examples' in path.split('/')
    ]
    idx_blocks = [
        i for i in range(len(paths_video)) if i not in idx_walkthrough + idx_examples
    ]

    dict_responses = {
        'tutorial': df_response.iloc[idx_walkthrough],
        'practice': df_response.iloc[idx_examples],
    }

    # Re-key the experiment responses by Video ID
    df_blocks = df_response.iloc[idx_blocks].merge(
        df_dataset_metadata.reset_index()[[col_block, col_video, COL_VIDEO_ID]],
        on=[col_block, col_video],
        how='left',
    )
    df_blocks = df_blocks.set_index(COL_VIDEO_ID).sort_index()
    df_blocks = df_blocks.drop(columns=['Path Video'])
    df_blocks['Block Shown'] = df_blocks['internal_node_id'].apply(
        infer_block_from_internal_node_id
    )

    dict_participant_data['blocks'] = [
        data.sort_values(by='trial_index')
        for _, data in df_blocks.groupby('Block Shown', sort=True)
    ]

    # Trials the participant did not answer in time
    dict_responses['missed'] = df_blocks.loc[df_blocks.response.isna()]

    for trial_type in trial_types:
        dict_responses[trial_type.lower()] = filter_video_indexed_data(
            df_blocks,
            df_dataset_metadata[df_dataset_metadata['trial'] == trial_type.title()],
        )

    # Everything the participant was scored on, catch trials excluded
    dict_responses['experiment'] = filter_video_indexed_data(
        df_blocks,
        df_dataset_metadata[~df_dataset_metadata['trial'].str.contains('Catch')],
    )

    dict_participant_data['responses'] = dict_responses
    return dict_participant_data


def load_all_participant_responses(
    path_participant_responses: Path,
    df_dataset_metadata: pd.DataFrame,
    shuffle: bool = False,
    sort: bool = True,
    skip_errors: bool = True,
    skip_iphone: bool = False,
    **kwargs,
) -> dict[str, dict[str, Any]]:
    """Read every participant export under a responses directory.

    Args:
        path_participant_responses: Root directory of participant exports.
        df_dataset_metadata: Trial metadata indexed by ``Video ID``.
        shuffle: Visit participants in random order.
        sort: Visit participants in sorted order.
        skip_errors: Skip participants whose export cannot be read instead of
            raising.
        skip_iphone: Drop participants who took the task on an iPhone, whose
            video playback differs from every other device.
        **kwargs: Forwarded to :func:`load_participant_responses`.

    Returns:
        Mapping of participant id to loaded participant data.

    Raises:
        ValueError: If both ``shuffle`` and ``sort`` are requested.
    """
    if shuffle and sort:
        raise ValueError("Cannot have both 'shuffle' and 'sort' set to True.")

    participant_ids = [path.stem for path in path_participant_responses.iterdir()]
    if shuffle:
        random.shuffle(participant_ids)
    if sort:
        participant_ids = sorted(participant_ids)

    dict_all_participants = {}
    for participant_id in participant_ids:
        try:
            participant_data = load_participant_responses(
                participant_id,
                path_participant_responses,
                df_dataset_metadata,
                **kwargs,
            )
        except Exception as exc:
            if not skip_errors:
                raise
            print(
                f"Loading participant '{participant_id}' raised "
                f'{type(exc).__name__}: {exc}. Skipping'
            )
            continue

        if skip_iphone and participant_data['device'] == 'iPhone':
            print(f"Participant '{participant_id}' used an iPhone. Skipping")
            continue

        dict_all_participants[participant_id] = participant_data

    return dict_all_participants


def load_all_participant_demographics(
    path_demographics: Path,
    valid_participant_status: tuple[str, ...] | None = VALID_PARTICIPANT_STATUS,
    col_participant_id: str = 'Participant id',
) -> pd.DataFrame:
    """Read the recruitment-platform demographics export.

    Args:
        path_demographics: CSV exported by the recruitment platform.
        valid_participant_status: Submission statuses to keep; ``None`` keeps
            every row.
        col_participant_id: Participant-id column in the export.

    Returns:
        Demographics indexed by ``Participant ID``, with parsed timestamps and
        task durations in minutes and milliseconds.
    """
    df_demographics = pd.read_csv(path_demographics).set_index(col_participant_id)
    df_demographics.index.name = COL_PARTICIPANT_ID

    if valid_participant_status is not None:
        df_demographics = df_demographics[
            df_demographics['Status'].isin(valid_participant_status)
        ]
    df_demographics = df_demographics.copy()

    df_demographics['Started at'] = pd.to_datetime(
        df_demographics['Started at'], errors='coerce'
    )
    df_demographics['Completed at'] = pd.to_datetime(
        df_demographics['Completed at'], utc=True, errors='coerce'
    )
    # Submissions that were never completed have no recorded duration
    time_taken = df_demographics['Time taken']
    df_demographics['Time taken (min)'] = (time_taken / 60).round().astype('Int64')
    df_demographics['Time taken (ms)'] = (time_taken * 1000).round().astype('Int64')

    # A participant with more than one submission keeps only the approved ones
    duplicated = df_demographics.index[df_demographics.index.duplicated(keep=False)]
    if len(duplicated):
        df_demographics = df_demographics[
            ~df_demographics.index.isin(duplicated)
            | (df_demographics['Status'] == 'APPROVED')
        ]

    return df_demographics


# ---------------------------------------------------------------------------
# Per-participant summary statistics
# ---------------------------------------------------------------------------

def compute_initial_stats_participant(
    participant_data: dict[str, Any],
    participant_demographics: pd.Series,
    responses_to_summarize: tuple[str, ...] = (
        'tutorial',
        'practice',
        'catch',
        'experiment',
        'straight',
        'bounce',
        'nonwall',
    ),
) -> dict[str, Any]:
    """Summarize one participant's session, timings and per-trial-type scores.

    Args:
        participant_data: Output of :func:`load_participant_responses`.
        participant_demographics: That participant's demographics row.
        responses_to_summarize: Response frames to score.

    Returns:
        Flat mapping of statistic name to value.
    """
    stats: dict[str, Any] = {}
    dict_responses = participant_data['responses']
    df_all_trials = participant_data['all_trials']

    # Time spent on the between-block debrief screens
    df_debrief = df_all_trials[df_all_trials['task'] == 'Block Debrief']
    stats['Block Debrief Time'] = ms_to_min(df_debrief['rt'].sum())

    time_elapsed_ms = df_all_trials.iloc[-1]['time_elapsed']
    stats['Duration Task'] = ms_to_min(time_elapsed_ms)
    stats['Duration Prolific'] = participant_demographics['Time taken (min)']

    stats['Task Start Date'] = pd.to_datetime(df_all_trials.iloc[-1]['start_date'])
    stats['Task Completed Date'] = stats['Task Start Date'] + pd.to_timedelta(
        time_elapsed_ms, unit='ms'
    )
    stats['Prolific Start Date'] = participant_demographics['Started at']
    stats['Prolific Completed Date'] = participant_demographics['Completed at']

    duration_prolific = (
        stats['Prolific Completed Date'] - stats['Prolific Start Date']
    ) / pd.Timedelta(minutes=1)
    stats['Duration Prolific Computed'] = (
        round(duration_prolific) if pd.notna(duration_prolific) else duration_prolific
    )
    stats['Prolific to Task Start'] = (
        stats['Task Start Date'] - stats['Prolific Start Date']
    )
    stats['Task End to Prolific'] = (
        stats['Prolific Completed Date'] - stats['Task Completed Date']
    )

    stats['Experiment Trials Present'] = len(dict_responses['experiment'])

    for key, df in dict_responses.items():
        if key not in responses_to_summarize or not len(df):
            continue
        name = key.title()
        stats[f'Accuracy {name}'] = response_accuracy(df)
        stats[f'RT {name}'] = np.round(df.rt.mean()).astype(int)
        stats[f'Average Confidence {name}'] = compute_confidence(df, average=True)
        stats[f'Median Confidence {name}'] = compute_confidence(df, median=True)
        stats[f'Var Confidence {name}'] = compute_confidence(df, var=True)
        stats[f'Proportion Answered Confidence {name}'] = (
            1 - df['slider_end'].isna().mean()
        )

    return stats


def compute_initial_stats_all_participants(
    data_all_participants: dict[str, dict[str, Any]],
    participant_demographics: pd.DataFrame,
    columns_time_reformat: tuple[str, ...] = (
        'Prolific to Task Start',
        'Task End to Prolific',
    ),
    skip_errors: bool = True,
    num_trials: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Summarize every participant that has both responses and demographics.

    Args:
        data_all_participants: Mapping of participant id to loaded data.
        participant_demographics: Demographics indexed by ``Participant ID``.
        columns_time_reformat: Timedelta columns rendered as ``HH:MM:SS``.
        skip_errors: Skip participants whose summary raises instead of
            propagating the error.
        num_trials: Trial count the presence percentage is taken against;
            defaults to the largest observed.
        **kwargs: Forwarded to :func:`compute_initial_stats_participant`.

    Returns:
        One row per summarized participant, indexed by ``Participant ID``.
    """
    stats_all = {}
    for participant_id, participant_data in data_all_participants.items():
        if participant_id not in participant_demographics.index:
            continue
        try:
            stats_all[participant_id] = compute_initial_stats_participant(
                participant_data,
                participant_demographics.loc[participant_id],
                **kwargs,
            )
        except Exception as exc:
            if not skip_errors:
                raise
            print(
                f"Computing stats for participant '{participant_id}' raised "
                f'{type(exc).__name__}: {exc}. Skipping'
            )

    df_stats = pd.DataFrame.from_dict(stats_all, orient='index')

    for column in columns_time_reformat:
        df_stats[column] = df_stats[column].apply(
            lambda value: (
                f'{value.components.hours:02}:{value.components.minutes:02}:'
                f'{value.components.seconds:02}'
                if pd.notna(value)
                else value
            )
        )

    if num_trials is None:
        num_trials = df_stats['Experiment Trials Present'].max()
    df_stats['Percent Trials Present'] = np.round(
        (df_stats['Experiment Trials Present'].values / num_trials) * 100
    ).astype(int)

    return df_stats.rename_axis(COL_PARTICIPANT_ID)


def add_missed_counts(
    df_participant_stats: pd.DataFrame,
    data_all_participants: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Add the count of trials each participant failed to answer in time."""
    df_stats = df_participant_stats.copy()
    df_stats['Num Missed'] = [
        len(data_all_participants[participant_id]['responses']['missed'])
        for participant_id in df_stats.index
    ]
    return df_stats


def add_slider_stats(
    df_participant_stats: pd.DataFrame,
    data_all_participants: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Add summaries of how each participant moved the confidence slider.

    ``RMS Slider Change`` measures how far the slider was moved from its
    starting position, ``Var RMS Slider Change`` how much that movement varied
    across trials, and ``RMS Slider End`` where the slider tended to end up.
    """
    df_stats = df_participant_stats.copy()

    slider_change, slider_change_var, slider_end = [], [], []
    for participant_id in df_stats.index:
        responses = data_all_participants[participant_id]['responses']['experiment']
        start = responses['slider_start'] / 100
        end = responses['slider_end'] / 100
        change_squared = (end - start) ** 2

        slider_end.append(np.sqrt((end ** 2).mean()))
        slider_change.append(np.sqrt(change_squared.mean()))
        slider_change_var.append(np.sqrt(change_squared).var())

    df_stats['RMS Slider Change'] = slider_change
    df_stats['Var RMS Slider Change'] = slider_change_var
    df_stats['RMS Slider End'] = slider_end
    return df_stats


# ---------------------------------------------------------------------------
# Exclusion filters
# ---------------------------------------------------------------------------

def filter_by_status(
    data_all_participants: dict[str, dict[str, Any]],
    participant_demographics: pd.DataFrame,
    excluded_status: str = EXCLUDED_PARTICIPANT_STATUS,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Drop participants who returned their submission.

    Args:
        data_all_participants: Mapping of participant id to loaded data.
        participant_demographics: Demographics indexed by ``Participant ID``.
        excluded_status: Submission status to exclude.

    Returns:
        The participant data and demographics, both restricted to the
        participants who did not return their submission.
    """
    kept = participant_demographics.index[
        participant_demographics['Status'] != excluded_status
    ]
    kept_set = set(kept)
    data_filtered = {
        participant_id: participant_data
        for participant_id, participant_data in data_all_participants.items()
        if participant_id in kept_set
    }
    return data_filtered, participant_demographics.loc[kept]


def filter_by_completeness(
    df_participant_stats: pd.DataFrame,
    total_trials: int,
    min_fraction: float = MIN_TRIAL_FRACTION,
) -> pd.DataFrame:
    """Keep participants who completed more than a fraction of the trials."""
    if 'Experiment Trials Present' not in df_participant_stats.columns:
        raise ValueError(
            "df_participant_stats must contain an 'Experiment Trials Present' column"
        )
    return df_participant_stats[
        df_participant_stats['Experiment Trials Present'] > total_trials * min_fraction
    ]


def filter_by_missed(
    df_participant_stats: pd.DataFrame,
    total_trials: int,
    max_fraction: float = MAX_MISSED_FRACTION,
) -> pd.DataFrame:
    """Keep participants who missed fewer than a fraction of the trials."""
    if 'Num Missed' not in df_participant_stats.columns:
        raise ValueError("df_participant_stats must contain a 'Num Missed' column")
    return df_participant_stats[
        df_participant_stats['Num Missed'] < total_trials * max_fraction
    ]


def filter_by_slider_use(
    df_participant_stats: pd.DataFrame,
    rms_cutoff: float = RMS_SLIDER_CUTOFF,
    var_cutoff: float = VAR_RMS_SLIDER_CUTOFF,
) -> pd.DataFrame:
    """Keep participants who actually used the confidence slider.

    Reporting confidence means both moving the slider away from its random
    starting position and moving it by different amounts on different trials.
    A participant has to clear both cutoffs: leaving the slider alone, or
    dragging it the same distance every trial, are both ways of answering
    without reporting confidence.
    """
    used_slider = (df_participant_stats['RMS Slider Change'] >= rms_cutoff) & (
        df_participant_stats['Var RMS Slider Change'] >= var_cutoff
    )
    return df_participant_stats[used_slider]


def filter_by_slider_values(
    df_participant_stats: pd.DataFrame,
    var_cutoff: float = SLIDER_VAR_CUTOFF,
    median_cutoff: float = SLIDER_MEDIAN_CUTOFF,
) -> pd.DataFrame:
    """Drop participants whose confidence reports never varied.

    Excludes anyone with no variance in reported confidence, and anyone who
    parked the slider at either extreme of its range for the whole experiment.
    """
    excluded = (
        (df_participant_stats['Var Confidence Experiment'] <= var_cutoff)
        | (df_participant_stats['RMS Slider End'] >= 1.0 - median_cutoff)
        | (df_participant_stats['RMS Slider End'] <= median_cutoff)
    )
    return df_participant_stats[~excluded]


def filter_by_catch_accuracy(
    df_participant_stats: pd.DataFrame,
    cutoff: float = CATCH_ACCURACY_CUTOFF,
) -> pd.DataFrame:
    """Keep participants who scored at least ``cutoff`` on the catch trials."""
    if 'Accuracy Catch' not in df_participant_stats.columns:
        raise ValueError("df_participant_stats must contain an 'Accuracy Catch' column")
    return df_participant_stats[df_participant_stats['Accuracy Catch'] >= cutoff]


# ---------------------------------------------------------------------------
# Confidence-weighted choice
# ---------------------------------------------------------------------------

def compute_participant_cwc(
    participant_responses: pd.DataFrame,
    df_trial_metadata: pd.DataFrame,
    confidence_cutoff: float = CONFIDENCE_RANGE_CUTOFF,
) -> pd.DataFrame:
    """Score one participant's responses as confidence-weighted choices.

    Each response becomes a signed choice — ``-1`` for predicting the color the
    ball entered the occluder with, ``+1`` for predicting a change — scaled by
    how confident the participant was. Confidence is the slider's final
    position with its dependence on the slider's random starting position
    regressed out, then normalized across the participant's own trials. A
    participant whose adjusted confidence barely varies is scored at full
    confidence throughout, so their choices still count.

    Args:
        participant_responses: One participant's video-indexed responses.
        df_trial_metadata: Trial metadata indexed by ``Video ID``.
        confidence_cutoff: Adjusted-confidence spread below which confidence
            reports carry no usable signal.

    Returns:
        The responses restricted to trials present in both frames, with
        ``Choice``, ``Confidence Adjusted``, ``Confidence``, ``CWC`` and
        ``correct_coded`` columns added.
    """
    responses, metadata = filter_video_indexed_data(
        participant_responses.dropna(), df_trial_metadata, two_way=True
    )
    responses = responses.copy()

    # Predicting the color the ball entered with is "no change", anything else
    # is a predicted change
    responses['Choice'] = 1
    responses.loc[responses['response'] == metadata['color_entered'], 'Choice'] = -1
    responses['correct_response_coded'] = metadata['correct_coded']
    responses['correct_coded'] = (
        responses['correct_response_coded'] == responses['Choice']
    )

    # Regress out the slider's random starting position, keeping the original
    # scale by recentering on the mean final position
    slider = responses[['slider_end', 'slider_start']].dropna().to_numpy()
    slope, intercept, *_ = linregress(slider[:, 1], slider[:, 0])
    final = responses['slider_end'].to_numpy(dtype=float)
    start = responses['slider_start'].to_numpy(dtype=float)
    confidence = final - (slope * start + intercept) + final.mean()
    responses['Confidence Adjusted'] = confidence

    confidence_range = np.sqrt(confidence.max() ** 2 - confidence.min() ** 2)
    if confidence_range > confidence_cutoff:
        low = np.quantile(confidence, 0.02)
        high = np.quantile(confidence, 0.98)
        confidence_normalized = ((confidence - low) / (high - low)).clip(0, 1)
    else:
        confidence_normalized = np.ones_like(confidence)

    responses['Confidence'] = confidence_normalized
    responses['CWC'] = responses['Choice'] * confidence_normalized
    return responses


def participant_cwc_by_hazard(
    participant_cwc: dict[str, pd.DataFrame],
    df_straight: pd.DataFrame,
    hazard_column: str = 'Hazard Rate',
    position_column: str = 'idx_time',
) -> pd.DataFrame:
    """Average each participant's CWC by hazard rate and grayzone position.

    The buckets come from the full experiment metadata, so every participant is
    grouped the same way and a participant's rows do not depend on which other
    participants were analyzed alongside them. A bucket a participant has no
    trials in yields a missing CWC rather than a dropped row.

    Args:
        participant_cwc: Mapping of participant id to scored responses.
        df_straight: Straight-trial metadata indexed by ``Video ID``.
        hazard_column: Column naming each trial's hazard rate.
        position_column: Column naming where in the grayzone the trial ended.

    Returns:
        Long table with one row per participant, hazard rate and position.
    """
    rows = []
    for participant_id, responses in participant_cwc.items():
        for hazard_rate, df_hazard in df_straight.groupby(hazard_column, observed=True):
            for position, df_position in df_hazard.groupby(
                position_column, observed=True
            ):
                matched = filter_video_indexed_data(responses, df_position)
                rows.append({
                    'Hazard Rate': hazard_rate,
                    'Grayzone Position': position,
                    COL_PARTICIPANT_ID: participant_id,
                    'CWC': matched['CWC'].mean(),
                })
    return pd.DataFrame(rows)


def participant_cwc_by_contingency(
    participant_cwc: dict[str, pd.DataFrame],
    df_bounce: pd.DataFrame,
    contingency_column: str = 'Contingency',
) -> pd.DataFrame:
    """Average each participant's CWC by contingency on the bounce trials.

    Args:
        participant_cwc: Mapping of participant id to scored responses.
        df_bounce: Bounce-trial metadata indexed by ``Video ID``.
        contingency_column: Column naming each trial's contingency.

    Returns:
        Long table with one row per participant and contingency.
    """
    rows = []
    for participant_id, responses in participant_cwc.items():
        for contingency, df_contingency in df_bounce.groupby(
            contingency_column, observed=True
        ):
            matched = filter_video_indexed_data(responses, df_contingency)
            rows.append({
                'Contingency': contingency,
                COL_PARTICIPANT_ID: participant_id,
                'CWC': matched['CWC'].mean(),
            })

    df_cwc = pd.DataFrame(rows)
    df_cwc['Trial'] = 'Bounce'
    return df_cwc


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _dataset_version(raw_dir: Path) -> str:
    """Recover the task version from a raw dataset directory name."""
    name = raw_dir.name
    prefix = 'hbb_'
    return name[len(prefix):] if name.startswith(prefix) else name


def run_participant_pipeline(
    raw_dir: Path,
    participant_dataset_dir: Path,
) -> dict[str, Any]:
    """Load, filter and score the participant cohort.

    Args:
        raw_dir: Raw dataset directory holding the participant response
            exports and the demographics CSV.
        participant_dataset_dir: Dataset directory holding the trial metadata
            and the rendered videos.

    Returns:
        Dictionary with the filtered participant statistics
        (``participant_stats``), the long per-trial CWC table
        (``participant_cwc``), and the cohort ``counts``.
    """
    raw_dir = Path(raw_dir)
    participant_dataset_dir = Path(participant_dataset_dir)
    version = _dataset_version(raw_dir)

    df_metadata, _ = load_trial_metadata(participant_dataset_dir)

    # Catch trials that give the answer away are scored separately, so this has
    # to happen before any participant's responses are keyed against the
    # metadata
    catch_timesteps = compute_catch_timesteps(df_metadata)
    df_metadata = reclassify_fast_catch_trials(df_metadata, catch_timesteps)
    num_catch_fast = int((df_metadata['trial'] == CATCH_FAST_LABEL).sum())

    data_all = load_all_participant_responses(
        raw_dir / f'hbb_participant_responses_{version}',
        df_metadata,
        dir_videos=participant_dataset_dir / 'videos',
    )
    total_loaded = len(data_all)

    demographics = load_all_participant_demographics(
        raw_dir / f'hbb_demographics_{version}.csv'
    )

    stage_counts: dict[str, int] = {}

    data_all, demographics = filter_by_status(data_all, demographics)
    df_stats = compute_initial_stats_all_participants(data_all, demographics)
    df_stats['Status'] = demographics.loc[df_stats.index]['Status']
    stage_counts['status'] = len(df_stats)

    # The trial count the cohort was meant to see, taken from the cohort itself
    total_trials = int(df_stats['Experiment Trials Present'].mode().iloc[0])

    df_stats = filter_by_completeness(df_stats, total_trials)
    stage_counts['completeness'] = len(df_stats)

    df_stats = add_missed_counts(df_stats, data_all)
    df_stats = filter_by_missed(df_stats, total_trials)
    stage_counts['missed'] = len(df_stats)

    df_stats = add_slider_stats(df_stats, data_all)
    df_stats = filter_by_slider_use(df_stats)
    stage_counts['slider_use'] = len(df_stats)

    df_stats = filter_by_slider_values(df_stats)
    stage_counts['slider_values'] = len(df_stats)

    df_stats = filter_by_catch_accuracy(df_stats)
    stage_counts['catch_accuracy'] = len(df_stats)

    df_stats = df_stats.sort_index()

    participant_cwc = {
        participant_id: compute_participant_cwc(
            data_all[participant_id]['responses']['experiment'], df_metadata
        )
        for participant_id in df_stats.index
    }

    cwc_columns = [
        'Choice',
        'Confidence Adjusted',
        'Confidence',
        'CWC',
        'correct_response_coded',
        'correct_coded',
    ]
    df_cwc = pd.concat(
        {
            participant_id: responses[cwc_columns]
            for participant_id, responses in participant_cwc.items()
        },
        names=[COL_PARTICIPANT_ID],
    ).reset_index()
    df_cwc = df_cwc.sort_values([COL_PARTICIPANT_ID, COL_VIDEO_ID]).reset_index(
        drop=True
    )

    counts = {
        'total_loaded': int(total_loaded),
        'stage_counts': {name: int(count) for name, count in stage_counts.items()},
        'final_n': int(len(df_stats)),
        'num_trials': int(total_trials),
        'num_catch_fast': num_catch_fast,
    }

    return {
        'participant_stats': df_stats,
        'participant_cwc': df_cwc,
        'counts': counts,
    }


def write_participant_stats(
    output_dir: Path,
    raw_dir: Path,
    participant_dataset_dir: Path,
) -> dict[str, Any]:
    """Run the participant pipeline and write its three artifacts.

    Args:
        output_dir: Directory the artifacts are written to.
        raw_dir: Raw dataset directory holding the participant response
            exports and the demographics CSV.
        participant_dataset_dir: Dataset directory holding the trial metadata
            and the rendered videos.

    Returns:
        The cohort counts, as written to ``participant_counts.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_participant_pipeline(raw_dir, participant_dataset_dir)

    results['participant_stats'].to_parquet(output_dir / ARTIFACT_STATS)
    results['participant_cwc'].to_parquet(output_dir / ARTIFACT_CWC, index=False)
    with open(output_dir / ARTIFACT_COUNTS, 'w') as f:
        json.dump(results['counts'], f, indent=2)

    return results['counts']


def main():
    """Command-line entry point for the participant-response pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory the participant artifacts are written to',
    )
    parser.add_argument(
        '--raw-dir',
        type=Path,
        required=True,
        help='Raw dataset directory with participant responses and demographics',
    )
    parser.add_argument(
        '--participant-dataset-dir',
        type=Path,
        required=True,
        help='Dataset directory with the trial metadata and rendered videos',
    )
    args = parser.parse_args()

    counts = write_participant_stats(
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        participant_dataset_dir=args.participant_dataset_dir,
    )

    print(f"Participant artifacts written to {args.output_dir}")
    print(f"  Participants loaded: {counts['total_loaded']}")
    for name, count in counts['stage_counts'].items():
        print(f"  Remaining after {name} filter: {count}")
    print(f"  Final sample: {counts['final_n']}")


if __name__ == '__main__':
    main()
