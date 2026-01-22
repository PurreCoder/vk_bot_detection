# learning_plots.py
"""
Модуль для построения графиков обучения (Loss/Accuracy по эпохам).
"""

import matplotlib.pyplot as plt
from typing import Optional, Union, List
import numpy as np


class LearningPlotter:
    """
    Класс для построения графиков обучения.

    Использование:
        plotter = LearningPlotter()
        plotter.plot(values=[0.9, 0.85, 0.8, 0.75], metric='loss')
        plotter.plot(values=[0.8, 0.85, 0.87, 0.89], metric='accuracy')
    """

    def plot(self,
             values: Union[List[float], np.ndarray],
             metric: str = 'loss',
             title: Optional[str] = None,
             color: str = 'blue',
             linewidth: int = 2,
             grid: bool = True,
             figsize: tuple = (10, 6)):
        """
        Построение графика по эпохам.

        Args:
            values: Массив значений (потерь или точности)
            metric: Тип метрики ('loss' или 'accuracy')
            title: Заголовок графика (если None, генерируется автоматически)
            color: Цвет линии
            linewidth: Толщина линии
            grid: Отображать сетку
            figsize: Размер фигуры (ширина, высота)

        Returns:
            fig, ax: Объекты matplotlib figure и axes
        """
        # Преобразуем в numpy array если нужно
        if isinstance(values, list):
            values = np.array(values)

        # Создаем массив эпох
        epochs = np.arange(0, len(values))

        # Определяем метки осей
        if metric.lower() == 'loss':
            y_label = 'Loss'
            default_title = 'Training Loss per Epoch'
        elif metric.lower() == 'accuracy':
            y_label = 'Accuracy'
            default_title = 'Training Accuracy per Epoch'
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'loss' or 'accuracy'")

        # Создаем график
        fig, ax = plt.subplots(figsize=figsize)

        # Рисуем линию
        ax.plot(epochs, values, color=color, linewidth=linewidth, linestyle='--')

        # Настройки осей
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title if title else default_title, fontsize=14, pad=20)

        # Настройки сетки
        if grid:
            ax.grid(True, alpha=0.3, linestyle='--')

        # Автоматическое масштабирование осей
        ax.set_xlim([0, len(values) - 1])

        # Для loss лучше начинать с 0 или близкого минимума
        if metric.lower() == 'loss':
            y_min = max(0, values.min() * 0.9)
            y_max = values.max() * 1.1
            ax.set_ylim([y_min, y_max])
        elif metric.lower() == 'accuracy':
            # Для точности обычно от 0 до 1
            y_min = max(0, min(values.min() * 0.98, values.min() - 0.05))
            y_max = min(1.0, max(values.max() * 1.02, values.max() + 0.05))
            ax.set_ylim([y_min, y_max])

        # Улучшаем разметку осей
        # Автоматически выбираем шаг для делений оси X
        n_epochs = len(values)
        if n_epochs <= 20:
            # Если мало эпох, показываем каждую
            xtick_step = 1
        else:
            # Если много эпох, показываем с шагом
            xtick_step = 5

        ax.set_xticks(np.arange(0, n_epochs, xtick_step))
        # Форматируем как целые числа
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))

        # Добавляем подписи значений в последней точке
        last_epoch = len(values)
        last_value = values[-1]
        if metric.lower() == 'accuracy':
            text = f'{last_value:.4f}'
        else:
            text = f'{last_value:.6f}'

        ax.annotate(text,
                    xy=(last_epoch, last_value),
                    xytext=(last_epoch + 0.5, last_value),
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

        plt.tight_layout()
        plt.show()

        return fig, ax


# Функция для быстрого использования без создания класса
def plot_learning_curve(values: Union[List[float], np.ndarray],
                        metric: str = 'loss',
                        **kwargs):
    """
    Быстрая функция для построения графика обучения.

    Args:
        values: Массив значений (потерь или точности)
        metric: Тип метрики ('loss' или 'accuracy')
        **kwargs: Дополнительные аргументы для LearningPlotter.plot()

    Returns:
        fig, ax: Объекты matplotlib figure и axes
    """
    plotter = LearningPlotter()
    return plotter.plot(values, metric, **kwargs)