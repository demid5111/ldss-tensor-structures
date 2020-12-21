import itertools

import matplotlib.pyplot as plt
import os
import pandas as pd




def get_img_folder_path():
    project_root = os.path.dirname(os.path.realpath(__file__))
    tmp_artifacts_root = os.path.join(project_root, 'img')
    return tmp_artifacts_root


def create_chart(task, comparison_criteria, nn_data, numpy_data, scipy_data):
    font_settings = {
        'fontname': 'Times New Roman',
        'fontsize': 12
    }
    
    sample_ticks = scipy_data['depth'].tolist()
    if comparison_criteria == 'memory':
        y_title = 'Memory used, Mb'
        chart_title = 'peak memory usage'

        filtered_data_df = pd.DataFrame({
            'x': sample_ticks,
            'nn': nn_data[f'{task}_FINISH'] - nn_data[f'{task}_START'],
            'numpy': numpy_data[f'{task}_FINISH'] - numpy_data[f'{task}_START'],
            'scipy': scipy_data[f'{task}_FINISH'] - scipy_data[f'{task}_START'],
        })
    else:  # it is time
        y_title = 'Wall time, seconds'
        chart_title = 'time elapsed'

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
    plt.xlabel('Depth of structure', **font_settings)
    plt.xticks(sample_ticks, **font_settings)
    plt.yticks(**font_settings)
    plt.title(f'{task.title()} task, {chart_title}')
    plt.legend()
    return plt


def save_chart(plot, task, comparison_criteria):
    path_template = os.path.join(get_img_folder_path(), f'{task}_{comparison_criteria}.{{}}')
    os.makedirs(get_img_folder_path(), exist_ok=True)
    plot.savefig(path_template.format('eps'), format='eps', dpi=600)


def main():
    nn_df = pd.read_csv('./data/test_nn.csv', dtype={'depth': 'int16'})
    numpy_df = pd.read_csv('./data/test_numpy.csv', dtype={'depth': 'int16'})
    scipy_df = pd.read_csv('./data/test_scipy.csv', dtype={'depth': 'int16'})

    plot = create_chart(task='decode', 
                        comparison_criteria='memory', 
                        nn_data=nn_df, 
                        numpy_data=numpy_df,
                        scipy_data=scipy_df)
    save_chart(plot, task='decode', comparison_criteria='memory')

    for task, criteria in itertools.product(('encode', 'decode'), ('memory', 'time')):
        plot = create_chart(task=task,
                            comparison_criteria=criteria,
                            nn_data=nn_df,
                            numpy_data=numpy_df,
                            scipy_data=scipy_df)
        save_chart(plot, task=task, comparison_criteria=criteria)


if __name__ == '__main__':
    main()
