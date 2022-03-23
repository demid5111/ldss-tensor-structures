import itertools

import matplotlib.pyplot as plt
import os
import pandas as pd


def get_img_folder_path():
    project_root = os.path.dirname(os.path.realpath(__file__))
    tmp_artifacts_root = os.path.join(project_root, 'img')
    return tmp_artifacts_root


def create_chart(task, comparison_criteria, nn_data, numpy_data, scipy_data, language='ru'):
    font_settings = {
        'fontname': 'Times New Roman',
        'fontsize': 12
    }

    i8n_mapping = {
        'memory': {
            'ru': 'Использованная память, Мбайт',
            'en': 'Peak memory, Mb'
        },
        'time': {
            'ru': 'Время выполнения, сек',
            'en': 'Time elapsed, sec'
        },
        'x_axis': {
            'ru': 'Глубина структуры',
            'en': 'Tree depth'
        }
    }

    sample_ticks = scipy_data['depth'].tolist()
    if comparison_criteria == 'memory':
        y_title = i8n_mapping['memory'][language]

        filtered_data_df = pd.DataFrame({
            'x': sample_ticks,
            'nn': nn_data[f'{task}_FINISH'] - nn_data[f'{task}_START'],
            'numpy': numpy_data[f'{task}_FINISH'] - numpy_data[f'{task}_START'],
            'scipy': scipy_data[f'{task}_FINISH'] - scipy_data[f'{task}_START'],
        })
    else:  # it is time
        y_title = i8n_mapping['time'][language]

        filtered_data_df = pd.DataFrame({
            'x': sample_ticks,
            'nn': nn_data[f'{task}_time'],
            'numpy': numpy_data[f'{task}_time'],
            'scipy': scipy_data[f'{task}_time'],
        })
    plt.close()
    plt.plot('x', 'nn', data=filtered_data_df, marker='o', color='black',
             linewidth=2, label='Keras')
    plt.plot('x', 'numpy', data=filtered_data_df, marker='s', color='black',
             linestyle='dashed', linewidth=2, label='NumPy')
    plt.plot('x', 'scipy', data=filtered_data_df, marker='^', color='black',
             linestyle='dotted', linewidth=2, label='SciPy')
    plt.yscale('log')
    plt.ylabel(y_title, **font_settings)
    plt.xlabel(i8n_mapping['x_axis'][language], **font_settings)
    plt.xticks(sample_ticks, **font_settings)
    plt.yticks(**font_settings)

    # task_title = f'{task.title()} task'
    # task_title = 'Задача кодирования структуры' if task == 'encode' else 'Задача декодирования структуры'
    # plt.title(task_title)
    plt.legend()
    return plt


def save_chart(plot, task, comparison_criteria, language='ru'):
    path_template = os.path.join(get_img_folder_path(), f'{task}_{comparison_criteria}_{language}.{{}}')
    os.makedirs(get_img_folder_path(), exist_ok=True)
    plot.savefig(path_template.format('eps'), format='eps', dpi=600)


def main():
    nn_df = pd.read_csv('./data/test_nn.csv', dtype={'depth': 'int16'})
    numpy_df = pd.read_csv('./data/test_numpy.csv', dtype={'depth': 'int16'})
    scipy_df = pd.read_csv('./data/test_scipy.csv', dtype={'depth': 'int16'})

    language = 'ru'
    for task, criteria in itertools.product(('encode', 'decode'), ('memory', 'time')):
        plot = create_chart(task=task,
                            comparison_criteria=criteria,
                            nn_data=nn_df,
                            numpy_data=numpy_df,
                            scipy_data=scipy_df,
                            language=language)
        save_chart(plot, task=task, comparison_criteria=criteria, language=language)


if __name__ == '__main__':
    main()
