import os
import argparse
import re
import resource
import subprocess
import time
from time import sleep
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import sparse

from core.active_passive_net.classifier.vendor.network import build_real_filler_extractor_network
from core.joiner.utils import generate_shapes
from core.joiner.vendor.network import shift_matrix
from core.unshifter.vendor.network import unshift_matrix
from demo.peano_mtwa_net import encode as encode_number


class MemoryMonitor:
    def __init__(self):
        self.keep_measuring = True

    def measure_usage(self, is_snapshot=True):
        max_usage = 0
        while self.keep_measuring:
            max_usage = max(
                max_usage,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            if is_snapshot:
                self.keep_measuring = False
            else:
                sleep(0.1)
        self.keep_measuring = True
        kilo = (2 ** 10)
        kb = max_usage / kilo
        mb = kb / kilo
        return mb


def input_data():
    fillers = np.array([
        [7, 0, 0, 0, 0],  # A
        [0, 4, 0, 0, 0],  # V
        [0, 0, 2, 0, 0],  # P
        [0, 0, 0, 5, 0],  # Aux
        [0, 0, 0, 0, 3],  # by
    ])
    roles = np.array([
        [10, 0],  # r_0
        [0, 5],  # r_1
    ])
    dual_basic_roles_case_1 = np.linalg.inv(roles)
    return fillers, roles, dual_basic_roles_case_1


def create_encoded_random_tree(depth):
    fillers, roles, _ = input_data()
    return encode_number(1, depth, roles, fillers)


def create_encoded_random_tree_numpy(depth):
    fillers, roles, _ = input_data()

    single_role_shape = roles[0].shape
    single_filler_shape = fillers[0].shape
    fillers_shapes = generate_shapes(max_tree_depth=depth,
                                     role_shape=single_role_shape,
                                     filler_shape=single_filler_shape)

    # this is the mocked version
    # print('Warning: Generating fake tensor instead of real encoding')
    # mocked_tree = [np.ones(shape) for shape in fillers_shapes]
    # return flattenize_per_tensor_representation(mocked_tree)

    # this is the true version
    filler_len = fillers_shapes[0][1]
    max_depth = len(fillers_shapes)
    new_left = fillers[0]
    new_right = fillers[1]
    for level_index in range(1, max_depth):
        left_shift_input = shift_matrix(roles[0], filler_len, level_index, name=None)
        right_shift_input = shift_matrix(roles[1], filler_len, level_index, name=None)
        joint = left_shift_input.dot(new_left) + right_shift_input.dot(new_right)
        new_left = np.append(np.zeros((filler_len, )), joint)
        new_right = new_left

    return new_left


def create_encoded_random_tree_scipy(depth):
    fillers, roles, _ = input_data()

    single_role_shape = roles[0].shape
    single_filler_shape = fillers[0].shape
    fillers_shapes = generate_shapes(max_tree_depth=depth,
                                     role_shape=single_role_shape,
                                     filler_shape=single_filler_shape)

    filler_len = fillers_shapes[0][1]
    max_depth = len(fillers_shapes)
    new_left = fillers[0]
    new_right = fillers[1]
    for level_index in range(1, max_depth):
        print(f'Loading sparse shift left matrix from level {level_index}')
        left_shift_input = sparse.load_npz(shift_sparse_path(level_index, 'left'))
        print(f'Loading sparse shift right matrix from level {level_index}')
        right_shift_input = sparse.load_npz(shift_sparse_path(level_index, 'right'))

        joint = left_shift_input.dot(new_left) + right_shift_input.dot(new_right)
        new_left = np.append(np.zeros((filler_len, )), joint)
        new_right = new_left

    return new_left


def decode_most_nested_element(tree, depth):
    fillers, roles, dual_roles = input_data()
    mocked_shape = np.arange(depth)
    keras_decode_patient = build_real_filler_extractor_network(roles=dual_roles,
                                                               fillers=fillers,
                                                               tree_shape=mocked_shape,
                                                               role_extraction_order=[1 for _ in range(depth)],
                                                               stop_level=0)

    extracted = keras_decode_patient.predict_on_batch([
        tree
    ])
    return extracted


def unshift_sparse_path(level_index):
    _, tmp_root = tmp_folder_path()
    return os.path.join(tmp_root, f'unshift_sparse_level_from_{level_index}.npz')


def shift_sparse_path(level_index, side):
    _, tmp_root = tmp_folder_path()
    return os.path.join(tmp_root, f'shift_sparse_level_to_{level_index}_{side}.npz')


def dump_array_as_sparse(path, array):
    print('Dumping for sparsification experiment')
    index_list = np.argwhere(array != 0)
    transposed_indexes = tuple(index_list.T)
    print(f'{array.shape} '
          f'-> {array.size} '
          f'-> {len(index_list)} '
          f'-> compression in {array.size / len(index_list)} times')

    row_indexes, column_indexes = transposed_indexes
    sparse_data = array[transposed_indexes]
    sparse_matrix = sparse.coo_matrix((sparse_data, (row_indexes, column_indexes)), shape=array.shape)
    sparse.save_npz(path, sparse_matrix)


def dump_unshift_matrixes_till(depth):
    fillers, roles, dual_roles = input_data()
    max_depth = depth - 1
    stop_level = 0
    filler_len = fillers[0].shape[0]
    role_extraction_order = [1 for _ in range(max_depth)]

    for level_index, role_index in zip(range(max_depth, stop_level - 1, -1), role_extraction_order):
        left_shift_input = unshift_matrix(roles[role_index], filler_len, level_index)
        dump_array_as_sparse(unshift_sparse_path(level_index), left_shift_input)


def dump_shift_matrixes_till(depth):
    fillers, roles, _ = input_data()
    filler_len = len(fillers)

    def _dump(role_index, mode):
        matrix = shift_matrix(roles[role_index], filler_len, level_index, name=None)
        path = shift_sparse_path(level_index, mode)
        dump_array_as_sparse(path, matrix)

    for level_index in range(1, depth):
        _dump(0, 'left')
        _dump(1, 'right')


def decode_most_nested_element_numpy(tree, depth):
    fillers, roles, dual_roles = input_data()
    current_input = tree
    max_depth = depth - 1
    stop_level = 0
    filler_len = fillers[0].shape[0]
    role_extraction_order = [1 for _ in range(max_depth)]

    for level_index, role_index in zip(range(max_depth, stop_level - 1, -1), role_extraction_order):
        left_shift_input = unshift_matrix(roles[role_index], filler_len, level_index)
        current_input = left_shift_input.dot(current_input[filler_len:])
    return current_input


def decode_most_nested_element_scipy(tree, depth):
    fillers, roles, dual_roles = input_data()
    current_input = tree
    max_depth = depth - 1
    stop_level = 0
    filler_len = fillers[0].shape[0]
    role_extraction_order = [1 for _ in range(max_depth)]

    for level_index, role_index in zip(range(max_depth, stop_level - 1, -1), role_extraction_order):
        print(f'Loading sparse matrix from level {level_index}')
        left_shift_input = sparse.load_npz(unshift_sparse_path(level_index))
        current_input = left_shift_input.dot(current_input[filler_len:])
    return current_input


def run_task(task, *args):
    monitor = MemoryMonitor()
    on_start = monitor.measure_usage(is_snapshot=True)

    with ThreadPoolExecutor() as executor:
        mem_thread = executor.submit(monitor.measure_usage)
        try:
            fn_thread = executor.submit(task, *args)
            result = fn_thread.result()
        finally:
            monitor.keep_measuring = False
            max_usage = mem_thread.result()
            monitor.keep_measuring = True

        on_peak = max_usage

    on_termination = monitor.measure_usage(is_snapshot=True)

    print(f'[START] {on_start}')
    print(f'[PEAK] {on_peak}')
    print(f'[FINISH] {on_termination}')

    return result


def tmp_folder_path():
    project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    tmp_artifacts_root = os.path.join(project_root, 'tmp')
    return project_root, tmp_artifacts_root


def process_task(depth, task, backend):
    print(f'Processing task {task} for depth {depth}')
    venv_path = os.path.join('venv', 'bin', 'python')
    project_root, tmp_artifacts_root = tmp_folder_path()
    python_executable = os.path.join(project_root, venv_path)

    options = [
        python_executable, __file__,
        '--depth', str(depth),
        '--mode', task,
        '--backend', backend,
        '--artifact', os.path.join(tmp_artifacts_root, f'test_{depth}_{backend}.npy')
    ]
    os.makedirs(tmp_artifacts_root, exist_ok=True)
    res = subprocess.run(options, capture_output=True)
    output = res.stdout.decode().strip().split('\n')
    pattern = re.compile(r'^\[(?P<type>START|PEAK|FINISH)\]\s(?P<value>\d+\.\d+)')
    values = {}
    try:
        if re.search('Error:', res.stdout.decode().strip()):
            raise AttributeError('stderr of the process is not empty')
        for line in output:
            match = re.search(pattern, line)
            if not match:
                continue
            values[f'{task}_{match.groupdict()["type"]}'] = float(match.groupdict()['value'])
    except AttributeError as e:
        print('Unable to parse output of the task')
        raise AttributeError from e
    return values


def process_depth(depth, backend):
    before_e = time.time()
    encode_values = process_task(depth, 'encode', backend)
    after_e = time.time() - before_e
    before_d = time.time()
    decode_values = process_task(depth, 'decode', backend)
    after_d = time.time() - before_d
    results = {
        **encode_values,
        'encode_time': after_e,
        **decode_values,
        'decode_time': after_d,
        'depth': depth
    }
    return results


def main(args):
    gen_col = [f'{t}_{stage}' for t in ['encode', 'decode'] for stage in ['START', 'PEAK', 'FINISH']]
    columns = gen_col + ['depth', 'encode_time', 'decode_time']
    df = pd.DataFrame(columns=columns)

    upper_bound = 12
    if args.backend == 'nn' and upper_bound is None:
        upper_bound = 12
    elif args.backend in ('numpy', 'scipy') and upper_bound is None:
        upper_bound = 30
    lower_bound = 2

    if args.backend == 'scipy':
        dump_shift_matrixes_till(upper_bound)
        dump_unshift_matrixes_till(upper_bound)

    steps = range(lower_bound, upper_bound) if upper_bound else (upper_bound, )
    for i, depth in enumerate(steps):
        values = process_depth(depth, args.backend)
        df.loc[i] = [values[col] for col in columns]

    _, tmp_artifacts_root = tmp_folder_path()
    df.to_csv(os.path.join(tmp_artifacts_root, f'test_{args.backend}.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--artifact", required=False, type=str)
    parser.add_argument("--backend", default='numpy', choices={'nn', 'numpy', 'scipy'})
    parser.add_argument("--mode", default='main', choices={'main', 'encode', 'decode'})

    args = parser.parse_args()
    print(args)

    if args.mode == 'main':
        main(args)
    elif args.mode == 'encode' and args.backend == 'nn':
        tree = run_task(create_encoded_random_tree, args.depth)
        np.save(args.artifact, tree)
    elif args.mode == 'encode' and args.backend == 'numpy':
        tree = run_task(create_encoded_random_tree_numpy, args.depth)
        np.save(args.artifact, tree)
    elif args.mode == 'encode' and args.backend == 'scipy':
        tree = run_task(create_encoded_random_tree_scipy, args.depth)
        np.save(args.artifact, tree)
    elif args.mode == 'decode' and args.backend == 'nn':
        tree = np.load(args.artifact)
        run_task(decode_most_nested_element, tree, args.depth)
    elif args.mode == 'decode' and args.backend == 'numpy':
        tree = np.load(args.artifact)
        run_task(decode_most_nested_element_numpy, tree, args.depth)
    elif args.mode == 'decode' and args.backend == 'scipy':
        tree = np.load(args.artifact)
        run_task(decode_most_nested_element_scipy, tree, args.depth)
