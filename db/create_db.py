"""
Populate the db and optionally initialize it.
"""
import db
import argparse
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from ops import helper_funcs
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from config import Allen_Brain_Observatory_Config as Config
from utils import logger
from utils import py_utils


def session_filters(config, cells_ID_list):
    """Find cells that present in all sessions"""
    masterkey = {k: v for k, v in config.session.iteritems()
                 if v in cells_ID_list.keys()}
    reference_session = masterkey[masterkey.keys()[0]]
    reference_cells = cells_ID_list[reference_session]
    flist = {}  # List of cells that found in all three sessions
    for cell in reference_cells:
        checks = []
        for k in masterkey.keys()[1:]:
            v = cells_ID_list[masterkey[k]]
            checks += [cell in v]
        flist[cell] = all(checks)
    return flist


def add_None_to_rfdict(Ref_dict, FLAG_ON=False, FLAG_OFF=False):
    list_keys_on = [
        'on_distance',
        'on_area',
        'on_overlap',
        'on_height',
        'on_center_x',
        'on_center_y',
        'on_width_x',
        'on_width_y',
        'on_rotation'
    ]
    list_keys_off = [
        'off_distance',
        'off_area',
        'off_overlap',
        'off_height',
        'off_center_x',
        'off_center_y',
        'off_width_x',
        'off_width_y',
        'off_rotation'
    ]
    if not FLAG_ON:
        for key in list_keys_on:
            Ref_dict[key] = None

    if not FLAG_OFF:
        for key in list_keys_off:
            Ref_dict[key] = None
    return Ref_dict


def get_all_pointers(config, cid, sess, stim):
    sess_type = sess['session_type']
    if stim in config.available_stims:
        stim_template = "%s%s_template.pkl" % (
            config.stimulus_template_loc, stim)
    else:
        stim_template = None
    pointer_list = {
        'fluorescence_trace': "%s%s_%s_traces.npz" % (
            config.fluorescence_traces_loc, cid, sess_type),
        'stim_table': "%s%s_%s_%s_table.npz" % (
            config.stim_table_loc, sess['id'], sess_type, stim),
        'other_recording': "%s%s_related_traces.npz" % (
            config.specimen_recording_loc, sess['id']),
        'ROImask': "%s%s_%s_ROImask.npz" % (
            config.ROIs_mask_loc, cid, sess_type),
        'stim_template': stim_template,
    }
    return pointer_list


def get_stim_list_boolean(boc, config, this_stim, output_dict):

    for stim in boc.get_all_stimuli():
        if stim in config.sess_with_number.keys():
            stim = config.sess_with_number[stim]
        output_dict[stim] = False
    output_dict[this_stim] = True
    return output_dict


def filter_dict(cell_rf_dict):
    for k, v in cell_rf_dict.iteritems():
        if isinstance(v, np.ndarray):
            if v.size == 0:
                v = None
            elif v.size == 1:
                v = float(v)
            else:
                v = float(v.ravel()[0])
            cell_rf_dict[k] = v
    return cell_rf_dict


def process_cell(
        boc,
        config,
        exp_session,
        RFinfo_this_exp,
        recorded_cells_list):
    # Get common cells
    cells_ID_list = {}
    for sess in exp_session:
        tmp = boc.get_ophys_experiment_data(sess['id'])
        cells_ID_list[sess['session_type']] = tmp.get_cell_specimen_ids()
    common_cells = session_filters(config, cells_ID_list)
    common_cell_id = []
    for cell_specimen_id, session_filter in common_cells.iteritems():
        if session_filter:
            common_cell_id += [cell_specimen_id]
    ###############################################
    # get RF and cell DB for each cell
    for cell_id in common_cell_id:
        this_cell_rf = RFinfo_this_exp[cell_id]
        for rf in this_cell_rf:
            if rf['lsn_name'] in config.pick_main_RF:
                represent_RF = rf
        cell_rf_dict = represent_RF.copy()
        cell_rf_dict = add_None_to_rfdict(
            cell_rf_dict,
            FLAG_ON=cell_rf_dict['found_on'],
            FLAG_OFF=cell_rf_dict['found_off'])
        cell_rf_dict = filter_dict(cell_rf_dict)
        list_of_cell_stim_dicts = []
        for session in exp_session:
            data_set = boc.get_ophys_experiment_data(session['id'])
            for stimulus in data_set.list_stimuli():
                output_pointer = os.path.join(
                    config.output_pointer_loc,
                    '%s_%s_%s.npy' % (
                        cell_id,
                        session['session_type'],
                        stimulus)
                    )
                all_pointers = get_all_pointers(
                    config=config,
                    cid=cell_id,
                    sess=session,
                    stim=stimulus)
                if stimulus in config.session_name_for_RF:
                    for rf in this_cell_rf:
                        if rf['lsn_name'] == stimulus:
                            rf_from_this_stim = rf.copy()
                    np.savez(
                        output_pointer,
                        neural_trace=all_pointers['fluorescence_trace'],
                        stim_template=all_pointers['stim_template'],
                        stim_table=all_pointers['stim_table'],
                        ROImask=all_pointers['ROImask'],
                        other_recording=all_pointers['other_recording'],
                        RF_info=rf_from_this_stim)
                else:
                    np.savez(
                        output_pointer,
                        neural_trace=all_pointers['fluorescence_trace'],
                        stim_template=all_pointers['stim_template'],
                        stim_table=all_pointers['stim_table'],
                        ROImask=all_pointers['ROImask'],
                        other_recording=all_pointers['other_recording'])
                it_stim_dict = {
                    'cell_specimen_id': cell_id,
                    'session': session['session_type'],
                    'cell_output_npy': output_pointer}
                it_stim_dict = get_stim_list_boolean(
                    boc=boc,
                    config=config,
                    this_stim=stimulus,
                    output_dict=it_stim_dict)
                list_of_cell_stim_dicts += [it_stim_dict]
        recorded_cells_list[cell_id] = {
          'cell_rf_dict': cell_rf_dict,
          'list_of_cell_stim_dicts': list_of_cell_stim_dicts
        }
        db.add_cell_data(
                cell_rf_dict,
                list_of_cell_stim_dicts)
    return recorded_cells_list


def populate_db(config, boc, log, timestamp, start_exp=None, end_exp=None):
    """Populate DB with cell information."""
    df = pd.read_csv(config.all_exps_csv)
    exp_con_ids = np.asarray(df['experiment_container_id'])
    if start_exp is None:
        start_exp = 0
    if end_exp is None:
        end_exp = len(exp_con_ids)
    idx_range = np.arange(start_exp, end_exp)

    # ---> list of exp / then dict of cell name
    RFs_info = helper_funcs.load_object(
        os.path.join(
            config.RF_info_loc,
            "all_cells_RF_info_08_30_17.pkl"))
    recorded_cells_list = {}
    log.info('Updating DB with experiment containers.')
    for idx in tqdm(
            idx_range,
            desc="Data from the experiment",
            total=len(idx_range)):
        exps = exp_con_ids[idx]
        exp_session = boc.get_ophys_experiments(
            experiment_container_ids=[exps])
        recorded_cells_list = process_cell(
            boc,
            config,
            exp_session,
            RFs_info[idx],
            recorded_cells_list)
    fname = "Recorded_cells_list_for_db_%s.pkl" % timestamp
    helper_funcs.save_object(recorded_cells_list, fname)
    log.info('Saved data to: %s.' % fname)


def main(initialize_database, start_exp=None, end_exp=None):
    config = Config()
    boc = BrainObservatoryCache(manifest_file=config.manifest_file)
    timestamp = py_utils.timestamp()
    log_file = os.path.join(config.log_dir, timestamp)
    log = logger.get(log_file)
    if initialize_database:
        db.initialize_database()
        log.info('Initialized DB.')
    populate_db(config, boc, log, timestamp, start_exp, end_exp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initialize",
        dest="initialize_database",
        action='store_true',
        help='Initialize database.')
    parser.add_argument(
        "--start",
        dest="start_exp",
        type=int,
        default=None,
        help='Start with this experiment.')
    parser.add_argument(
        "--end",
        dest="end_exp",
        type=int,
        default=None,
        help='End with this experiment.')
    main(**vars(parser.parse_args()))
