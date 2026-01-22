import os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import config
from data_processing.data_filter import sieve_deactivated, balance_users
from data_processing.data_processor import DataProcessor
from data_processing.file_manager import load_all_users
from gnn_models.model_1.model import Model as my_model


def get_data():
    bots_users, humans_users = load_all_users(f"../{config.DATA_SOURCE['BOTS_FILE']}", f"../{config.DATA_SOURCE['HUMANS_FILE']}")

    bots_users, humans_users = sieve_deactivated(bots_users, humans_users)
    bots_users, humans_users = balance_users(bots_users, humans_users)

    processor = DataProcessor(my_model)
    data, labels, ids_list = processor.get_all_features(bots_users, humans_users)
    return data, labels, ids_list


def visualize_with_umap(data, labels, ids_list, title="UMAP Visualization",
                        n_neighbors=15, min_dist=0.1, metric='euclidean',
                        figsize=(12, 10), random_state=42, alpha=0.7,
                        save_path=None, interactive=False, show_legend=True,
                        class_names=None, density_plot=False):
    """
    Визуализирует многомерные данные в 2D пространстве с помощью UMAP.

    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        Многомерные данные для визуализации
    labels : array-like, shape (n_samples,)
        Бинарные метки классов (0 и 1)
    title : str, optional
        Заголовок графика
    n_neighbors : int, optional
        Параметр UMAP: размер локальной окрестности
    min_dist : float, optional
        Параметр UMAP: минимальное расстояние между точками
    metric : str, optional
        Метрика расстояния для UMAP
    figsize : tuple, optional
        Размер фигуры matplotlib
    random_state : int, optional
        Seed для воспроизводимости
    alpha : float, optional
        Прозрачность точек
    save_path : str, optional
        Путь для сохранения изображения
    interactive : bool, optional
        Если True, использует Plotly для интерактивной визуализации
    show_legend : bool, optional
        Показывать ли легенду
    class_names : list, optional
        Имена классов для легенды [класс_0, класс_1]
    density_plot : bool, optional
        Если True, добавляет плотность распределения

    Returns:
    --------
    umap_result : array, shape (n_samples, 2)
        2D координаты после UMAP
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        Объект фигуры
    """

    # Проверка входных данных
    data = np.array(data)
    labels = np.array(labels).flatten()

    assert len(data) == len(labels), "Длина data и labels должна совпадать"
    assert len(np.unique(labels)) == 2, "Метки должны быть бинарными (два уникальных значения)"

    # Нормализация данных (рекомендуется для UMAP)
    #scaler = StandardScaler()
    #data_normalized = scaler.fit_transform(data)

    # Создание и обучение UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_jobs=-1  # Использовать все ядра
    )

    print("Выполняется UMAP преобразование...")
    umap_result = reducer.fit_transform(data)

    # Преобразование меток в строки для лучшей визуализации
    unique_labels = np.unique(labels)
    if class_names is None:
        class_names = [f'Class {int(label)}' for label in unique_labels]
    else:
        assert len(class_names) == 2, "class_names должен содержать 2 элемента"

    # Интерактивная визуализация с Plotly
    if interactive:
        return interactive_visualization(umap_result, labels, ids_list, title,
                                         class_names, save_path)

    # Статичная визуализация с Matplotlib
    return static_visualization(umap_result, labels, title, figsize,
                                alpha, show_legend, class_names,
                                density_plot, save_path, umap_result)


def static_visualization(map_result, labels, title, figsize, alpha,
                         show_legend, class_names, density_plot,
                         save_path, map_result_return):
    """Статичная визуализация с Matplotlib"""
    # Создание фигуры
    fig, ax = plt.subplots(figsize=figsize)

    # Разделение точек по классам
    unique_labels = np.unique(labels)
    colors = ['#1f77b4', '#ff7f0e']  # Синий и оранжевый

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            map_result[mask, 0],
            map_result[mask, 1],
            c=[colors[i]],
            label=class_names[i],
            alpha=alpha,
            s=50,
            edgecolors='w',
            linewidth=0.5
        )

    # Добавление графика плотности (контур)
    if density_plot:
        from scipy import stats
        try:
            # Ядерная оценка плотности
            x = map_result[:, 0]
            y = map_result[:, 1]
            xx, yy = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([x, y])
            kernel = stats.gaussian_kde(values)
            f = np.reshape(kernel(positions).T, xx.shape)

            # Рисуем контуры
            ax.contour(xx, yy, f, colors='k', alpha=0.3, linewidths=0.5)
            ax.contourf(xx, yy, f, alpha=0.05, cmap='Blues')
        except:
            print("Не удалось построить график плотности")

    # Настройки графика
    ax.set_xlabel('UMAP Component 1', fontsize=12)
    ax.set_ylabel('UMAP Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Легенда
    if show_legend:
        ax.legend(fontsize=11, loc='best', framealpha=0.9)

    # Добавление статистики
    stats_text = f'Total samples: {len(labels)}\n'
    stats_text += f'Class {class_names[0]}: {np.sum(labels == unique_labels[0])}\n'
    stats_text += f'Class {class_names[1]}: {np.sum(labels == unique_labels[1])}'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Сохранение
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен как {save_path}")

    plt.show()

    return map_result_return, fig


def interactive_visualization(umap_result, labels, ids_list, title,
                              class_names, save_path):
    """Интерактивная визуализация с Plotly"""
    import plotly.express as px
    import pandas as pd

    # Создание DataFrame для Plotly
    df = pd.DataFrame({
        'UMAP1': umap_result[:, 0],
        'UMAP2': umap_result[:, 1],
        'Class': [class_names[0] if label == np.unique(labels)[0]
                  else class_names[1] for label in labels],
        'Label': labels,
        'UserID': ids_list
    })

    # Создание интерактивного графика
    fig = px.scatter(df, x='UMAP1', y='UMAP2', color='Class',
                     title=title,
                     opacity=0.7,
                     hover_data=['Label'],
                     custom_data=['UserID'],
                     color_discrete_sequence=['#1f77b4', '#ff7f0e'])

    # Настройки макета
    fig.update_layout(
        title_font_size=16,
        title_x=0.5,
        xaxis_title="UMAP Component 1",
        yaxis_title="UMAP Component 2",
        legend_title="Classes",
        width=900,
        height=700,
        clickmode='event+select'
    )

    #fig.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))

    # Настраиваем всплывающие подсказки
    fig.update_traces(
        marker=dict(size=10, line=dict(width=1, color='white')),
        hovertemplate=(
                "<b>User ID: %{customdata[0]}</b><br>" +
                "Class: %{customdata[1]}<br>" +
                "UMAP1: %{x:.3f}<br>" +
                "UMAP2: %{y:.3f}<br>" +
                "<extra></extra>"
        )
    )

    # Сохранение
    # if save_path and save_path.endswith('.html'):
    #     fig.write_html(save_path)
    #     print(f"Интерактивный график сохранен как {save_path}")
    # elif save_path:
    #     fig.write_image(save_path)
    #     print(f"График сохранен как {save_path}")

    # Сохраняем с JavaScript
    html = fig.to_html(include_plotlyjs='cdn')

    # Добавляем JavaScript для кликов
    js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        document.querySelector('.js-plotly-plot').on('plotly_click', function(data) {
            if (data.points[0]) {
                const userId = data.points[0].customdata[0];
                window.open(`https://vk.com/id${userId}`, '_blank');
            }
        });
    });
    </script>
    """

    html = html.replace('</body>', js + '</body>')

    with open('profiles.html', 'w', encoding='utf-8') as f:
        f.write(html)

    fig.show()

    return umap_result, fig


# Дополнительная функция для сравнения разных параметров UMAP
def compare_umap_parameters(data, labels, param_grid=None):
    """
    Сравнивает визуализацию с разными параметрами UMAP

    Parameters:
    -----------
    data : array-like
        Входные данные
    labels : array-like
        Метки классов
    param_grid : dict, optional
        Сетка параметров для сравнения
        По умолчанию: {'n_neighbors': [5, 15, 30], 'min_dist': [0.1, 0.5, 0.99]}
    """
    if param_grid is None:
        param_grid = {
            'n_neighbors': [5, 30, 50],
            'min_dist': [0.1, 0.5, 0.99]
        }

    fig, axes = plt.subplots(
        len(param_grid['n_neighbors']),
        len(param_grid['min_dist']),
        figsize=(15, 12)
    )

    for i, n_neighbors in enumerate(param_grid['n_neighbors']):
        for j, min_dist in enumerate(param_grid['min_dist']):
            ax = axes[i, j]

            # Применяем UMAP
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric='jaccard',
                n_jobs=-1
            )

            umap_result = reducer.fit_transform(data)

            # Визуализация
            unique_labels = np.unique(labels)
            colors = ['#1f77b4', '#ff7f0e']

            for k, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    umap_result[mask, 0],
                    umap_result[mask, 1],
                    c=[colors[k]],
                    alpha=0.6,
                    s=20
                )

            ax.set_title(f'n_neighbors={n_neighbors}\nmin_dist={min_dist}', fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('Сравнение параметров UMAP', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def pca_test(data, labels, ids_list):
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(data)
    interactive_visualization(x_pca, labels, ids_list, 'Интерактивная PCA визуализация',
                              ['Бот', 'Человек'], 'umap_interactive.html')

def tsne_test(data, labels, ids_list):
    tsne = TSNE(perplexity=35)
    tsne_res = tsne.fit_transform(data)
    interactive_visualization(tsne_res, labels, ids_list, 'Интерактивная PCA визуализация',
                              ['Бот', 'Человек'], 'umap_interactive.html')

def umap_test(data, labels, ids_list):
    umap_result_interactive, fig_interactive = visualize_with_umap(
        data=data,
        labels=labels,
        ids_list=ids_list,
        title="Интерактивная UMAP визуализация",
        interactive=True,
        class_names=['Бот', 'Человек'],
        metric='euclidean',
        min_dist=0.5,
        n_neighbors=50,
        save_path='umap_interactive.html'
    )


def main():
    # result, figure = example_usage()
    np.random.seed(42)
    data, labels, ids_list = get_data()
    umap_test(data, labels, ids_list)
    # compare_umap_parameters(data, labels)


if __name__ == "__main__":
    main()
