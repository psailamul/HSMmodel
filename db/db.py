#!/usr/bin/env python
import sys
import json
import sshtunnel
import argparse
import psycopg2
import psycopg2.extras
import psycopg2.extensions
import credentials
import numpy as np
from config import Config
from ops import hp_opt_utils
sshtunnel.DAEMON = True  # Prevent hanging process due to forward thread
main_config = Config()


class db(object):
    def __init__(self, config):
        """Init global variables."""
        self.status_message = False
        self.db_schema_file = 'db/db_schema.txt'
        # Pass config -> this class
        for k, v in config.items():
            setattr(self, k, v)

    def __enter__(self):
        """Enter method."""
        if main_config.db_ssh_forward:
            forward = sshtunnel.SSHTunnelForwarder(
                credentials.machine_credentials()['ssh_address'],
                ssh_username=credentials.machine_credentials()['username'],
                ssh_password=credentials.machine_credentials()['password'],
                remote_bind_address=('127.0.0.1', 5432))
            forward.start()
            self.forward = forward
            self.pgsql_port = forward.local_bind_port
        else:
            self.forward = None
            self.pgsql_port = ''
        pgsql_string = credentials.postgresql_connection(str(self.pgsql_port))
        self.pgsql_string = pgsql_string
        self.conn = psycopg2.connect(**pgsql_string)
        self.conn.set_isolation_level(
            psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        self.cur = self.conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method."""
        if exc_type is not None:
            print exc_type, exc_value, traceback
            self.close_db(commit=False)
        else:
            self.close_db()
        if main_config.db_ssh_forward:
            self.forward.close()
        return self

    def close_db(self, commit=True):
        """Commit changes and exit the DB."""
        self.conn.commit()
        self.cur.close()
        self.conn.close()

    def experiment_fields(self):
        """Dict of fields in exp & hp_combo_history tables. DEPRECIATED."""
        return {
            'experiment_name': ['experiments', 'hp_combo_history'],
            'model_struct': ['experiments', 'hp_combo_history'],
            'loss_function': ['experiments', 'hp_combo_history'],
            'regularization_type': ['experiments', 'hp_combo_history'],
            'regularization_strength': ['experiments', 'hp_combo_history'],
            'optimizer': ['experiments', 'hp_combo_history'],
            'lr': ['experiments', 'hp_combo_history'],
            'dataset': ['experiments', 'hp_combo_history'],
            'regularization_type_domain': ['experiments', 'hp_combo_history'],
            'regularization_strength_domain': ['experiments', 'hp_combo_history'],
            'optimizer_domain': ['experiments', 'hp_combo_history'],
            'lr_domain': ['experiments', 'hp_combo_history'],
            'timesteps': ['experiments', 'hp_combo_history'],
            'timesteps_domain': ['experiments', 'hp_combo_history'],
            'u_t_domain': ['experiments', 'hp_combo_history'],
            'q_t_domain': ['experiments', 'hp_combo_history'],
            't_t_domain': ['experiments', 'hp_combo_history'],
            'p_t_domain': ['experiments', 'hp_combo_history'],
            'u_t': ['experiments', 'hp_combo_history'],
            'q_t': ['experiments', 'hp_combo_history'],
            't_t': ['experiments', 'hp_combo_history'],
            'p_t': ['experiments', 'hp_combo_history'],
            'hp_optim': ['experiments', 'hp_combo_history'],
            'hp_multiple': ['experiments', 'hp_combo_history'],
            'hp_current_iteration': ['experiments', 'hp_combo_history'],
            'experiment_iteration': ['experiments', 'hp_combo_history']
        }

    def fix_namedict(self, namedict, table):
        """Insert empty fields in dictionary where keys are absent."""
        experiment_fields = self.experiment_fields()
        for idx, entry in enumerate(namedict):
            for k, v in experiment_fields.iteritems():
                if k == 'experiment_iteration':
                    # Initialize iterations at 0
                    entry[k] = 0
                elif k not in entry.keys():
                    entry[k] = None
            namedict[idx] = entry
        return namedict

    def recreate_db(self, run=False):
        """Initialize the DB from the schema file."""
        if run:
            db_schema = open(self.db_schema_file).read().splitlines()
            for s in db_schema:
                t = s.strip()
                if len(t):
                    self.cur.execute(t)

    def return_status(
            self,
            label,
            throw_error=False):
        """
        General error handling and status of operations.
        ::
        label: a string of the SQL operation (e.g. 'INSERT').
        throw_error: if you'd like to terminate execution if an error.
        """
        if label in self.cur.statusmessage:
            print 'Successful %s.' % label
        else:
            if throw_error:
                raise RuntimeError('%s' % self.cur.statusmessag)
            else:
                'Encountered error during %s: %s.' % (
                    label, self.cur.statusmessage
                    )

    def populate_db(self, namedict):
        """
        Add a combination of parameter_dict to the db.
        ::
        experiment_name: name of experiment to add
        parent_experiment: linking a child (e.g. clickme) -> parent (ILSVRC12)
        """
        namedict = self.fix_namedict(namedict, 'experiments')
        self.cur.executemany(
            """
            INSERT INTO experiments
            (
            experiment_name,
            model_struct,
            loss_function,
            regularization_type,
            regularization_strength,
            optimizer,
            lr,
            dataset,
            regularization_type_domain,
            regularization_strength_domain,
            optimizer_domain,
            lr_domain,
            timesteps,
            timesteps_domain,
            u_t_domain,
            q_t_domain,
            t_t_domain,
            p_t_domain,
            u_t,
            q_t,
            t_t,
            p_t,
            hp_optim,
            hp_multiple,
            hp_current_iteration,
            experiment_iteration
            )
            VALUES
            (
            %(experiment_name)s,
            %(model_struct)s,
            %(loss_function)s,
            %(regularization_type)s,
            %(regularization_strength)s,
            %(optimizer)s,
            %(lr)s,
            %(dataset)s,
            %(regularization_type_domain)s,
            %(regularization_strength_domain)s,
            %(optimizer_domain)s,
            %(lr_domain)s,
            %(timesteps)s,
            %(timesteps_domain)s,
            %(u_t_domain)s,
            %(q_t_domain)s,
            %(t_t_domain)s,
            %(p_t_domain)s,
            %(u_t)s,
            %(q_t)s,
            %(t_t)s,
            %(p_t)s,
            %(hp_optim)s,
            %(hp_multiple)s,
            %(hp_current_iteration)s,
            %(experiment_iteration)s
            )
            """,
            namedict)
        self.cur.execute(
            """
            UPDATE experiments
            SET experiment_link=_id
            WHERE experiment_name=%(experiment_name)s
            """,
            namedict[0])
        if self.status_message:
            self.return_status('INSERT')

    def get_parameters(self, experiment_name=None, random=True):
        """Pull parameters DEPRECIATED."""
        if experiment_name is not None:
            exp_string = """experiment_name='%s' and""" % experiment_name
        else:
            exp_string = """"""
        if random:
            rand_string = """ORDER BY random()"""
        else:
            rand_string = """"""
        self.cur.execute(
            """
            SELECT * from experiments h
            WHERE %s NOT EXISTS (
                SELECT 1
                FROM in_process i
                WHERE h._id = i.experiment_id
                )
            %s
            """ % (
                exp_string,
                rand_string
                )
        )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def get_parameters_and_reserve(self, experiment_name=None, random=True):
        """Pull parameters and update the in process table."""
        if experiment_name is not None:
            exp_string = """experiment_name='%s' and""" % experiment_name
        else:
            exp_string = """"""
        if random:
            rand_string = """ORDER BY random()"""
        else:
            rand_string = """"""
        self.cur.execute(
            """
            INSERT INTO in_process (experiment_id, experiment_name)
            (SELECT _id, experiment_name FROM experiments h
            WHERE %s NOT EXISTS (
                SELECT 1
                FROM in_process i
                WHERE h._id = i.experiment_id
                )
            %s LIMIT 1)
            RETURNING experiment_id
            """ % (
                exp_string,
                rand_string,
                )
        )
        self.cur.execute(
            """
            SELECT * FROM experiments
            WHERE _id=%(_id)s
            """,
            {
                '_id': self.cur.fetchone()['experiment_id']
            }
        )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def list_experiments(self):
        """List all experiments."""
        self.cur.execute(
            """
            SELECT distinct(experiment_name) from experiments
            """
        )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchall()

    def update_in_process(self, experiment_id, experiment_name):
        """Update the in_process table."""
        self.cur.execute(
            """
             INSERT INTO in_process
             VALUES
             (%(experiment_id)s, %(experiment_name)s)
            """,
            {
                'experiment_id': experiment_id,
                'experiment_name': experiment_name
            }
        )
        if self.status_message:
            self.return_status('INSERT')

    def get_performance(self, experiment_name):
        """Get experiment performance."""
        self.cur.execute(
            """
            SELECT * FROM performance AS P
            INNER JOIN experiments ON experiments._id = P.experiment_id
            WHERE P.experiment_name=%(experiment_name)s
            """,
            {
                'experiment_name': experiment_name
            }
        )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchall()

    def remove_experiment(self, experiment_name):
        """Delete an experiment from all tables."""
        self.cur.execute(
            """
            DELETE FROM experiments WHERE experiment_name=%(experiment_name)s;
            DELETE FROM performance WHERE experiment_name=%(experiment_name)s;
            DELETE FROM in_process WHERE experiment_name=%(experiment_name)s;
            """,
            {
                'experiment_name': experiment_name
            }
        )
        if self.status_message:
            self.return_status('DELETE')

    def reset_in_process(self):
        """Reset in process table."""
        self.cur.execute(
            """
            DELETE FROM in_process
            """
        )
        if self.status_message:
            self.return_status('DELETE')

    def update_performance(self, namedict):
        """Update performance in database."""
        self.cur.execute(
            """
            INSERT INTO performance
            (
            experiment_id,
            experiment_name,
            summary_dir,
            ckpt_file,
            training_loss,
            validation_loss,
            time_elapsed,
            training_step
            )
            VALUES
            (
            %(experiment_id)s,
            %(experiment_name)s,
            %(summary_dir)s,
            %(ckpt_file)s,
            %(training_loss)s,
            %(validation_loss)s,
            %(time_elapsed)s,
            %(training_step)s
            )
            RETURNING _id""",
            namedict
            )
        if self.status_message:
            self.return_status('SELECT')


def get_experiment_name():
    """Get names of experiments."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        param_dict = db_conn.get_parameters()
    if param_dict is None:
        print 'No remaining experiments to run.'
        sys.exit(1)
    return param_dict['experiment_name']


def get_parameters(experiment_name, log, random=False):
    """Get parameters for a given experiment."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        param_dict = db_conn.get_parameters_and_reserve(
            experiment_name=experiment_name,
            random=random)
        log.info('Using parameters: %s' % json.dumps(param_dict, indent=4))
        if param_dict is not None:
            experiment_id = param_dict['_id']
            # db_conn.update_in_process(
            #     experiment_id=experiment_id,
            #     experiment_name=experiment_name)
        else:
            experiment_id = None
    if param_dict is None:
        raise RuntimeError('This experiment is complete.')
    return param_dict, experiment_id


def initialize_database():
    """Initialize and recreate the database."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.recreate_db(run=True)
        db_conn.return_status('CREATE')


def reset_in_process():
    """Reset the in_process table."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.reset_in_process()
    print 'Cleared the in_process table.'


def list_experiments():
    """List all experiments in the database."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        experiments = db_conn.list_experiments()
    return experiments


def update_performance(
        experiment_id,
        experiment_name,
        summary_dir,
        ckpt_file,
        training_loss,
        validation_loss,
        time_elapsed,
        training_step):
    """Update performance table for an experiment."""
    config = credentials.postgresql_connection()
    perf_dict = {
        'experiment_id': experiment_id,
        'experiment_name': experiment_name,
        'summary_dir': summary_dir,
        'ckpt_file': ckpt_file,
        'training_loss': training_loss,
        'validation_loss': validation_loss,
        'time_elapsed': time_elapsed,
        'training_step': training_step,
    }
    with db(config) as db_conn:
        db_conn.update_performance(perf_dict)


def get_performance(experiment_name):
    """Get performance for an experiment."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        perf = db_conn.get_performance(experiment_name=experiment_name)
    return perf


def query_hp_hist(exp_params, eval_on='validation_loss', init_top=1e10):
    """Query an experiment's history of hyperparameters and performance."""
    config = credentials.postgresql_connection()
    domain_param_map = hp_opt_utils.hp_opt_dict()
    experiment_name = exp_params['experiment_name']
    with db(config) as db_conn:
        perf = db_conn.get_performance(experiment_name=experiment_name)
        if len(perf) == 0:
            # And set hp history to initial values.
            hp_history = {}
            for k, v in domain_param_map.iteritems():
                if exp_params[k] is not None:  # If searching this domain.
                    hp_history[v] = exp_params[v]

            # First step of hp-optim. Requires > 1 entry for X/Y.
            perf = [
                [init_top + ((np.random.rand() - 0.5) * init_top / 10)],
                [init_top + ((np.random.rand() - 0.5) * init_top / 10)]
            ]
            hp_history = [
                hp_history,
                hp_history
            ]
        else:
            # Sort performance by time elapsed
            times = [x['time_elapsed'] for x in perf]
            time_idx = np.argsort(times)
            perf = [perf[idx][eval_on] for idx in time_idx]

            # Sort hp parameters by history
            times = [x['experiment_iteration'] for x in exp_params]
            time_idx = np.argsort(times)
            hp_history = [exp_params[idx] for idx in time_idx]
    return perf, hp_history


def main(
        initialize_db,
        reset_process=False):
    """Test the DB."""
    if reset_process:
        reset_in_process()
    if initialize_db:
        print 'Initializing database.'
        initialize_database()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset_process",
        dest="reset_process",
        action='store_true',
        help='Reset the in_process table.')
    parser.add_argument(
        "--initialize",
        dest="initialize_db",
        action='store_true',
        help='Recreate your database.')
    args = parser.parse_args()
    main(**vars(args))
