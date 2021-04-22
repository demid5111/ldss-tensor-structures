import numpy as np

from core.active_passive_net.utils import elementary_join
from core.joiner.utils import generate_shapes
from core.joiner.vendor.network import build_tree_joiner_network
from core.model_2_tuple.core import Model2Tuple
from core.model_2_tuple.decoder.vendor.network import build_decode_model_2_tuple_network
from core.utils import flattenize_per_tensor_representation


class FillerFactory:
    alpha_precision = 1  # save only 1 sign after the floating point
    weight_precision = 1  # save only 1 sign after the floating point

    @staticmethod
    def get_encoding_indexes():
        model_2_tuple = Model2Tuple(term_index=0, alpha=.0, linguistic_scale_size=5)
        index, alpha, weight = FillerFactory._to_tpr_fillers(model_2_tuple)

        encoding_order = ['index', 'alpha', 'weight']
        encoding_ranges = {
            'index': [0, len(index)],
            'alpha': [len(index), len(index) + len(alpha)],
            'weight': [len(index) + len(alpha), len(index) + len(alpha) + len(weight)]
        }
        return encoding_ranges, encoding_order

    @staticmethod
    def from_model_2_tuple(model_2_tuple: Model2Tuple):
        index, alpha, weight = FillerFactory._to_tpr_fillers(model_2_tuple)

        ranges, order = FillerFactory.get_encoding_indexes()
        empty_filler = np.zeros(FillerFactory.get_filler_size())

        filler_index = np.copy(empty_filler)
        filler_index[ranges['index'][0]:ranges['index'][1]] = index

        filler_alpha = np.copy(empty_filler)
        filler_alpha[ranges['alpha'][0]:ranges['alpha'][1]] = alpha

        filler_weight = np.copy(empty_filler)
        filler_weight[ranges['weight'][0]:ranges['weight'][1]] = weight

        return filler_index, filler_alpha, filler_weight

    @staticmethod
    def decode_tpr(tpr_index: np.array, tpr_alpha: np.array, tpr_weight: np.array):
        index_vector = FillerFactory._extract_filler_from_full_filler(tpr_index, 'index')
        alpha_vector = FillerFactory._extract_filler_from_full_filler(tpr_alpha, 'alpha')
        weight_vector = FillerFactory._extract_filler_from_full_filler(tpr_weight, 'weight')

        value_position = np.where(index_vector > 0)[0][0]
        term_index = int(index_vector[value_position])

        alpha_position = np.where(alpha_vector != 0)
        if alpha_position[0]:
            alpha_position = alpha_position[0][0]
            alpha = float(alpha_vector[alpha_position]) / (FillerFactory.alpha_precision * 10)
        else:
            alpha = .0

        value_position = np.where(weight_vector != 0)
        if value_position:
            value_position = value_position[0][0]
            weight = float(weight_vector[value_position]) / (FillerFactory.weight_precision * 10)
        else:
            weight = .0

        return term_index, alpha, weight

    @staticmethod
    def get_filler_size():
        ranges, order = FillerFactory.get_encoding_indexes()
        last_package = order[-1]
        return ranges[last_package][-1]

    @staticmethod
    def _extract_filler_from_full_filler(full_filler, component_name):
        ranges, order = FillerFactory.get_encoding_indexes()
        return full_filler[ranges[component_name][0]:ranges[component_name][1]]

    @staticmethod
    def _to_tpr_fillers(model_2_tuple: Model2Tuple):
        """returns f_i filler"""
        index = [0 for _ in range(model_2_tuple.linguistic_scale_size)]
        index[model_2_tuple.term_index] = model_2_tuple.term_index

        # alpha is in [-0.5, 0.5]
        # [-5 -4 -3 -2 -1 0 1 2 3 4 5]
        alpha = [0 for _ in range(11)]
        rounded_value = round(model_2_tuple.alpha * FillerFactory.alpha_precision * 10)
        rounded_value_index = rounded_value + 5
        alpha[rounded_value_index] = rounded_value

        # alpha is in [-0.5, 0.5]
        # [0 1 2 3 4 5 6 7 8 9]
        weight = [0 for _ in range(10)]
        rounded_value = round(model_2_tuple.weight * FillerFactory.weight_precision * 10)
        weight[rounded_value] = rounded_value

        return index, alpha, weight


def encode_model_2_tuple(model_2_tuple: Model2Tuple) -> np.array:
    roles = np.array([
        [1, 0, 0],  # r_i
        [0, 1, 0],  # r_alpha
        [0, 0, 1],  # r_w
    ])

    filler_index, filler_alpha, filler_weight = FillerFactory.from_model_2_tuple(model_2_tuple)

    MAX_TREE_DEPTH = 2
    SINGLE_ROLE_SHAPE = roles[0].shape
    SINGLE_FILLER_SHAPE = filler_index.shape

    fillers_shapes = generate_shapes(max_tree_depth=MAX_TREE_DEPTH,
                                     role_shape=SINGLE_ROLE_SHAPE,
                                     filler_shape=SINGLE_FILLER_SHAPE)

    keras_joiner = build_tree_joiner_network(roles=roles, fillers_shapes=fillers_shapes)

    fillers = np.array([filler_index, filler_alpha, filler_weight])
    model_2_tuple_encoded = elementary_join(joiner_network=keras_joiner,
                                            input_structure_max_shape=fillers_shapes,
                                            basic_roles=roles,
                                            basic_fillers=fillers,
                                            subtrees=(
                                                filler_index,
                                                filler_alpha,
                                                filler_weight
                                            ))
    return model_2_tuple_encoded


def decode_model_2_tuple_tpr(mta_result_encoded: np.array):
    roles = np.array([
        [1, 0, 0],  # r_i
        [0, 1, 0],  # r_alpha
        [0, 0, 1],  # r_w
    ])
    dual_roles = np.linalg.inv(roles)
    filler_len = FillerFactory.get_filler_size()
    MAX_TREE_DEPTH = 2

    keras_decoder = build_decode_model_2_tuple_network(filler_len=filler_len, dual_roles=dual_roles,
                                                       max_depth=MAX_TREE_DEPTH)

    flattened_model_2_tuple_tpr = flattenize_per_tensor_representation(mta_result_encoded)

    filler_index, filler_alpha, filler_weight = keras_decoder.predict_on_batch([
        *flattened_model_2_tuple_tpr
    ])

    term_index, alpha, weight = FillerFactory.decode_tpr(filler_index, filler_alpha, filler_weight)

    return Model2Tuple(term_index=term_index, alpha=alpha, weight=weight)


def aggregate_and_check(model_2_tuple_a, model_2_tuple_b):
    if model_2_tuple_a.linguistic_scale_size != model_2_tuple_b.linguistic_scale_size:
        raise ValueError('2-tuple elements should be from the single scale!')

    aggregation = model_2_tuple_a.to_number() * model_2_tuple_a.weight + \
                  model_2_tuple_b.to_number() * model_2_tuple_b.weight
    mta_result = Model2Tuple.from_number(beta=aggregation,
                                         linguistic_scale_size=model_2_tuple_a.linguistic_scale_size)

    mta_result_encoded = encode_model_2_tuple(mta_result)
    decoded_2_tuple = decode_model_2_tuple_tpr(mta_result_encoded)

    if mta_result != decoded_2_tuple:
        raise ValueError('Encoding is working with information loss!')


def main():
    print('Converting 2-tuple to TPR')
    linguistic_scale_size = 5
    first_tuple = Model2Tuple(term_index=3, alpha=0, linguistic_scale_size=linguistic_scale_size)
    second_tuple = Model2Tuple(term_index=2, alpha=0, linguistic_scale_size=linguistic_scale_size)

    aggregate_and_check(first_tuple, second_tuple)
    print('Converting 2-tuple to TPR and back works with no information loss!')


if __name__ == '__main__':
    main()
