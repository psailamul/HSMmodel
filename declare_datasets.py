class declare_allen_datasets():
    """Class for declaring datasets to be encoded as tfrecords."""
    def __getitem__(self, name):
        """Method for addressing class methods."""
        return getattr(self, name)

    def __contains__(self, name):
        """Method for checking class contents."""

    def __init__(self):
        """Global variables for all datasets."""
        pass

    def add_globals(self, exp):
        """Add attributes to this class."""
        for k, v in self.globals().iteritems():
            exp[k] = v
        return exp

    def all_neurons(self):
        """Pull data from all neurons."""
        exp_dict = {
            'experiment_name': 'all_neurons',
            'rf_coordinate_range': {  # Get all
                'x_min': -10000,
                'x_max': 10000,
                'y_min': -10000,
                'y_max': 10000,
            }
        }
        return exp_dict
