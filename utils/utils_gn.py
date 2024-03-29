import numpy as np
import os
import pickle
import h5py
from sklearn.preprocessing import StandardScaler
from functools import reduce
from matplotlib import cm
from utils import utils_models, utils_noah, utils_ivc, utils_dgrd
from datetime import datetime
import importlib
import hashlib
from config.definitions import ROOT_DIR

importlib.reload(utils_models)
importlib.reload(utils_noah)
importlib.reload(utils_ivc)
importlib.reload(utils_dgrd)


def test_set_check(identifier, test_ratio, hash):
    """
    Function that checks if a sample belongs to a test set.

    Args:
    ----
        identifier:  identifier in the dataset
        test_ratio:  fraction of test set
        hash:        hash of the identifier

    Returns:
    -------
            boolean corresponding to whether the hash of the identify <= test_ratio * 256
    """
    return hash(identifier.encode()).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, hash=hashlib.md5):
    """
    Function to split data into train and test set.

    Args:
    ----
         data:        data to be split
         test_ratio:  fraction of test set

    Returns:
    -------
            train, test splits
    """
    np.random.seed(42)

    ids = np.array(list(data.keys()))
    # Shuffle the ids
    shuffled_indices = np.random.permutation(len(ids))
    ids = ids[shuffled_indices]

    in_test_set = [test_set_check(id_, test_ratio, hash) for id_ in ids]
    ids_test = np.asarray(list(data.keys()))[in_test_set]

    return {k: data[k] for k in ids if k not in ids_test}, {k: data[k] for k in ids_test}


class FeatureTransformation:
    """
    Class that transforms raw battery data into features that can be fed into ml models.
    """
    __slots__ = ['n', 'step_size', 'sc', 'selected_feature_names']

    def __init__(self, n=None, step_size=1):
        self.n = int(n)
        self.step_size = step_size
        self.sc = StandardScaler()
        self.selected_feature_names = None

    def fit_transform(self, data, targets, with_eol):
        df = utils_ivc.ccv_features(data_dict=data, step_size=self.step_size, n=self.n).join(
            utils_dgrd.create_knee_elbow_data(data, with_eol)[targets])
        df_features_only = df.drop(targets, axis=1)

        # Perform feature scaling
        self.sc = self.sc.fit(df_features_only.values)
        return self.sc.transform(df_features_only.values), df[targets].values

    def transform(self, data):
        df = utils_ivc.ccv_features(
            data_dict=data, step_size=self.step_size, n=self.n)
        return self.sc.transform(df.values)


def time_monitor(initial_time=None):
    """
    This function monitors time from the start of a process to the end of the process
    """
    if not initial_time:
        initial_time = datetime.now()
        return initial_time
    else:
        thour, temp_sec = divmod(
            (datetime.now() - initial_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)

        return '%ih %imin and %ss.' % (thour, tmin, round(tsec, 2))


def load_data(filename, batch_num, num_cycles=None):
    """
    This function loads the downloaded matlab file into a dictionary

    Args:
        filename:     string with the path of the data file
        batch_num: index of this batch
        num_cycles:   number of cycles to be loaded

    Returns a dictionary with data for each cell in the batch
    """

    # read the matlab file
    f = h5py.File(filename, 'r')
    batch = f['batch']

    # get the number of cells in this batch
    num_cells = batch['summary'].shape[0]

    # initialize a dictionary to store the result
    batch_dict = {}

    summary_features = ["IR", "QCharge", "QDischarge", "Tavg", "Tmin",
                        "Tmax", "chargetime", "cycle"]
    cycle_features = ["I", "Qc", "Qd", "Qdlin",
                      "T", "Tdlin", "V", "discharge_dQdV", "t"]

    for i in range(num_cells):

        # decide how many cycles will be loaded
        if num_cycles is None:
            loaded_cycles = f[batch['cycles'][i, 0]]['I'].shape[0]
        else:
            loaded_cycles = min(
                num_cycles, f[batch['cycles'][i, 0]]['I'].shape[0])

        if i % 10 == 0:
            print(f"* {i} cells loaded ({loaded_cycles} cycles)")

        # initialise a dictionary for this cell
        cell_dict = {
            "cycle_life": f[batch["cycle_life"][i, 0]][()]
            if batch_num != 3
            else f[batch["cycle_life"][i, 0]][()] + 1,
            "charge_policy": f[batch["policy_readable"][i, 0]][()]
            .tobytes()[::2]
            .decode(),
            "summary": {},
        }

        for feature in summary_features:
            cell_dict["summary"][feature] = np.hstack(
                f[batch['summary'][i, 0]][feature][0, :].tolist())

        # for the cycle data
        cell_dict["cycle_dict"] = {}

        for j in range(loaded_cycles):
            cell_dict["cycle_dict"][str(j + 1)] = {}
            for feature in cycle_features:
                cell_dict["cycle_dict"][str(j + 1)][feature] = np.hstack(
                    (f[f[batch['cycles'][i, 0]][feature][j, 0]][()]))

        # converge into the batch dictionary
        batch_dict[f'b{batch_num}c{i}'] = cell_dict

    return batch_dict


def load_and_save_dict_data(num_cycles=None, option=1):
    """
    This function load and save downloaded matlab files as pickle files.

    Args:
    ----
         num_cycles:  number of cycles to load
         option:      1: to load all batches in one pickle file, 2: to load each batch and save it in a pickle file separately
    """

    # paths for data file with each batch of cells
    mat_filenames = {
        "batch1": os.path.join(f"{ROOT_DIR}", "data", "2017-05-12_batchdata_updated_struct_errorcorrect.mat"),
        "batch2": os.path.join(f"{ROOT_DIR}", "data", "2017-06-30_batchdata_updated_struct_errorcorrect.mat"),
        "batch3": os.path.join(f"{ROOT_DIR}", "data", "2018-04-12_batchdata_updated_struct_errorcorrect.mat"),
        "batch4": os.path.join(f"{ROOT_DIR}", "data", "2018-08-28_batchdata_updated_struct_errorcorrect.mat"),
        "batch5": os.path.join(f"{ROOT_DIR}", "data", "2018-09-02_batchdata_updated_struct_errorcorrect.mat"),
        "batch6": os.path.join(f"{ROOT_DIR}", "data", "2018-09-06_batchdata_updated_struct_errorcorrect.mat"),
        "batch7": os.path.join(f"{ROOT_DIR}", "data", "2018-09-10_batchdata_updated_struct_errorcorrect.mat"),
        "batch8": os.path.join(f"{ROOT_DIR}", "data", "2019-01-24_batchdata_updated_struct_errorcorrect.mat")
    }


    start = time_monitor()
    print("Loading batch 1 data...")
    batch1 = load_data(mat_filenames["batch1"], 1, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 2 data...")
    batch2 = load_data(mat_filenames["batch2"], 2, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 3 data...")
    batch3 = load_data(mat_filenames["batch3"], 3, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 4 data...")
    batch4 = load_data(mat_filenames["batch4"], 4, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 5 data...")
    batch5 = load_data(mat_filenames["batch5"], 5, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 6 data...")
    batch6 = load_data(mat_filenames["batch6"], 6, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 7 data...")
    batch7 = load_data(mat_filenames["batch7"], 7, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 8 data...")
    batch8 = load_data(mat_filenames["batch8"], 8, num_cycles=num_cycles)
    print(time_monitor(start))

    print(f"* {len(batch1.keys())} cells loaded in batch 1")
    print(f"* {len(batch2.keys())} cells loaded in batch 2")
    print(f"* {len(batch3.keys())} cells loaded in batch 3")
    print(f"* {len(batch4.keys())} cells loaded in batch 4")
    print(f"* {len(batch5.keys())} cells loaded in batch 5")
    print(f"* {len(batch6.keys())} cells loaded in batch 6")
    print(f"* {len(batch7.keys())} cells loaded in batch 7")
    print(f"* {len(batch8.keys())} cells loaded in batch 8")

    # there are four cells from batch1 that carried into batch2, we'll remove the data from batch2 and put it with
    # the correct cell from batch1
    b2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
    b1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
    add_len = [662, 981, 1060, 208, 482]

    # append data to batch 1
    for i, bk in enumerate(b1_keys):
        batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]

        for j in batch1[bk]['summary'].keys():
            if j == 'cycle':
                batch1[bk]['summary'][j] = np.hstack(
                    (batch1[bk]['summary'][j],
                     batch2[b2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
            else:
                batch1[bk]['summary'][j] = np.hstack(
                    (batch1[bk]['summary'][j],
                     batch2[b2_keys[i]]['summary'][j]))

        last_cycle = len(batch1[bk]['cycle_dict'].keys())

        # useful when all cycles loaded
        if num_cycles is None:
            for j, jk in enumerate(batch2[b2_keys[i]]['cycle_dict'].keys()):
                batch1[bk]['cycle_dict'][str(
                    last_cycle + j)] = batch2[b2_keys[i]]['cycle_dict'][jk]
    '''
    The authors exclude cells that:
        * do not reach 80% capacity (batch 1)
        * were carried into batch2 but belonged to batch 1 (batch 2)
        * noisy channels (batch 3)
    '''

    exc_cells = {"batch1": ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"],
                 "batch2": ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"],
                 "batch3": ["b3c37", "b3c2", "b3c23", "b3c32", "b3c38", "b3c39"]}

    for c in exc_cells["batch1"]:
        del batch1[c]

    for c in exc_cells["batch2"]:
        del batch2[c]

    for c in exc_cells["batch3"]:
        del batch3[c]

    # exclude the first cycle from all cells because this data was not part of the first batch of cells
    batches = [batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8]
    for batch in batches:
        for cell in batch.keys():
            del batch[cell]['cycle_dict']['1']

    for batch in batches:
        for cell in batch.keys():
            assert '1' not in batch[cell]['cycle_dict'].keys()

    for batch in batches:
        for cell in batch.keys():
            for feat in batch[cell]['summary'].keys():
                batch[cell]['summary'][feat] = np.delete(
                    batch[cell]['summary'][feat], 0)

    if num_cycles is None:
        filename_suffix = 'all.pkl'
    else:
        filename_suffix = f'{str(num_cycles)}cycles.pkl'

    if option == 1:

        # combine all batches in one dictionary
        data_dict = {**batch1, **batch2, **batch3, **
                     batch4, **batch5, **batch6, **batch7, **batch8}

        # save the dict as a pickle file
        dump_data(
            data=data_dict,
            path=f"{ROOT_DIR}/data",
            fname=f"data_{filename_suffix}"
        )

    elif option == 2:
        # save the batch separately
        for i, batch in zip(('1', '2', '3', '4', '5', '6', '7', '8'), batches):
            dump_data(
                data=batch,
                path=f"{ROOT_DIR}/data",
                fname=f"batch{i}_{filename_suffix}"
            )


def read_data(fname, folder="data"):
    # Load pickle data
    with open(os.path.join(folder, fname), "rb") as fp:
        df = pickle.load(fp)

    return df


def scaler(X):
    """
    A function that performs standard scaling of an input data.

    Argument:
             X:  the data to be scaled
    Returns:
            scaled data
    """
    scaler = StandardScaler()

    return scaler.fit_transform(X)


def dict_of_colours(data_dict):
    """
    This function returns a dictionary of colors which correspond to the EOL of cells
    """

    # get the eol of cells and normalize it
    eol = utils_noah.cycle_life(data_dict)['cycle_life']
    eol = (eol - eol.min()) / (eol.max() - eol.min())

    # define the colour map and map it to the normalized eol
    cmap = cm.get_cmap('viridis')
    colours = cmap(eol)

    return dict(zip(data_dict.keys(), colours))


def read_data(path, fname):
    # load pickle data
    with open(os.path.join(path, fname), "rb") as fp:
        data = pickle.load(fp)

    return data


def dump_data(data, path, fname):
    with open(os.path.join(path, fname), "wb") as fp:
        pickle.dump(data, fp)
