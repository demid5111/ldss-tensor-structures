import numpy as np

from .core import Model2Tuple


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

        value_position = np.where(index_vector > 0)
        if len(value_position[0]) > 0:
            value_position = value_position[0][0]
            term_index = int(index_vector[value_position])
        else:
            term_index = 0

        alpha_position = np.where(alpha_vector != 0)
        if len(alpha_position[0]) > 0:
            alpha_position = alpha_position[0][0]
            alpha = float(alpha_vector[alpha_position]) / (FillerFactory.alpha_precision * 10)
        else:
            alpha = .0

        weight_position = np.where(weight_vector != 0)
        if len(weight_position[0]) > 0:
            weight_position = weight_position[0][0]
            weight = float(weight_vector[weight_position]) / (FillerFactory.weight_precision * 10)
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
