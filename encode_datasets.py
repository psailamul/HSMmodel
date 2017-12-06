import os
import argparse
import numpy as np
from db import db
from config import Allen_Brain_Observatory_Config as Config
from declare_datasets import declare_allen_datasets as DA

"""
def load_npzs(data_dicts):
    """Load cell data from an npz."""
    cell_fields = [
        'neural_trace',
        'stim_template',
        'stim_table',
        'ROI_mask',
        'other_recording',
        'RF_info'
    ]
    data_files = []
    missing_keys = []
    for d in data_dicts:
        df = {}
        it_check = []
        cell_data = [np.load(d['output_dict'])]
        for k, v in cell_data.iteritems():
            df[k] = v  # Load data
        data_files += [df]
        for k in cell_fields.keys():
            if k not in df.keys():
                it_check += [k]
        missing_keys += [it_check]
    remove_keys = np.unique(missing_keys)
    if remove_keys is not None:
        for idx, d in enumerate(data_files):
            it_d = {k: v for k, v in d.iteritems() if k not in remove_keys}
            data_files[idx] = it_d
    return data_files
"""

def prepare_data_for_tf_records():
    pass


def package_dataset(config, dataset_info, output_directory):
    """Query and package."""
    dataset_instructions = dataset_info['cross_ref']
    if 'coordinates' in dataset_instructions:
        data_dicts = db.get_cells_by_rf(dataset_info['rf_coordinate_range'])
    else:
        # Incorporate more queryies and eventually allow inner-joining on them.
        raise RuntimeError('Other instructions are not yet implemented.')
    data_files = load_npzs(data_dicts)
    prepped_data = prepare_data_for_tf_records(data_files)


def main(experiment_name, output_directory=None):
    """Pull desired experiment cells and encode as tfrecords."""
    config = Config()
    da = DA()[experiment_name]
    if output_directory is None:
        output_directory = config.tf_record_output
    package_dataset(
        config=config,
        dataset_info=da,
        output_directory=output_directory)
