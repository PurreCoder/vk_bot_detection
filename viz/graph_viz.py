import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image as mpimg


def visualize_feature_importance(feature_weights, feature_names, filename):
    plt.subplot(1, 3, 3)

    if feature_weights is None:
        print(f"Feature importance visualization failed.")
        plt.text(0.5, 0.5, 'Feature importance\nnot available',
                 ha='center', va='center', transform=plt.gca().transAxes)
        return

    plt.barh(feature_names, feature_weights)
    plt.title('Топ-10 важных признаков')
    plt.xlabel('Важность')

    # Saving to file to show ones more in the future
    fig_to_save = plt.figure()
    fig_to_save.add_subplot(1, 1, 1)
    plt.barh(feature_names, feature_weights)
    plt.title('Топ-10 важных признаков')
    plt.xlabel('Важность')
    fig_to_save.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig_to_save)


def visualize_graph_3d(graph_data):
    """3D визуализация графа"""
    try:
        import networkx as nx
        from mpl_toolkits.mplot3d import Axes3D

        ax = plt.subplot(1, 3, 2, projection='3d')

        g = nx.Graph()
        edges = graph_data.edge_index.t().cpu().numpy()

        # Ограничиваем количество узлов для производительности
        max_nodes = 20000
        if graph_data.num_nodes > max_nodes:
            # Берем случайную выборку узлов
            node_indices = np.random.choice(graph_data.num_nodes, max_nodes, replace=False)
            mask = np.isin(edges[:, 0], node_indices) & np.isin(edges[:, 1], node_indices)
            edges = edges[mask]

        g.add_edges_from(edges)

        # 3D позиции узлов
        pos = nx.spring_layout(g, dim=3)

        # Преобразуем позиции в numpy массивы
        nodes = list(g.nodes())
        x = [pos[node][0] for node in nodes]
        y = [pos[node][1] for node in nodes]
        z = [pos[node][2] for node in nodes]

        # Цвета узлов
        node_colors = []
        node_sizes = []
        for node in nodes:
            if node < len(graph_data.y):
                if graph_data.y[node] == 0:
                    node_colors.append('red')  # Боты
                    node_sizes.append(80)
                else:
                    node_colors.append('blue')  # Люди
                    node_sizes.append(50)
            #else:
            #    node_colors.append('gray')  # Неизвестные
            #    node_sizes.append(30)

        # Рисуем узлы
        ax.scatter(x, y, z, c=node_colors, s=node_sizes, alpha=0.7, edgecolors='w', linewidths=0.5)

        # Рисуем ребра
        for edge in g.edges():
            x_vals = [pos[edge[0]][0], pos[edge[1]][0]]
            y_vals = [pos[edge[0]][1], pos[edge[1]][1]]
            z_vals = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x_vals, y_vals, z_vals, 'gray', alpha=0.3, linewidth=0.5)

        ax.set_title('3D Визуализация графа\n(красные - боты, синие - люди)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Легенда
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Боты'),
            Patch(facecolor='blue', label='Люди')#,
            #Patch(facecolor='gray', label='Неизвестные')
        ]
        ax.legend(handles=legend_elements, loc='upper left')

    except ImportError as e:
        print(f"3D visualization requires mpl_toolkits: {e}")
        visualize_graph_2d(graph_data)
    except Exception as e:
        print(f"3D graph visualization failed: {e}")
        visualize_graph_2d(graph_data)


def visualize_graph_2d(graph_data, with_labels=False):
    """2D визуализация графа"""
    plt.subplot(1, 3, 2)
    try:
        import networkx as nx
        g = nx.Graph()
        for vertex in range(graph_data.num_nodes):
            g.add_node(vertex)
        edges = graph_data.edge_index.t().cpu().numpy()

        # Ограничиваем для производительности
        max_edges = 20000
        if len(edges) > max_edges:
            edges = edges[:max_edges]

        g.add_edges_from(edges)

        node_colors = []
        for i in range(min(2000, graph_data.num_nodes)):
            if i < len(graph_data.y):
                if graph_data.y[i] == 0:
                    node_colors.append('red')  # Боты
                else:
                    node_colors.append('blue')  # Люди
            else:
                node_colors.append('gray')  # Неизвестные

        pos = nx.spring_layout(g, method='force')
        nx.draw(g, pos, node_color=node_colors[:len(g.nodes)],
                node_size=50, with_labels=with_labels, alpha=0.7)
        plt.title('Граф (красные - боты, синие - люди)')

    except Exception as e:
        print(f"2D graph visualization failed: {e}")


def visualize_menu(graph_data, results, feature_weights=None, feature_names=None,
                   filename='bar_chart.png', use_3d=True, with_labels=False, show=True):
    """Основная функция визуализации с опцией 3D"""

    fig = plt.figure(figsize=(15, 5))

    # 1. Меню сравнения моделей
    plt.subplot(1, 3, 1)
    plt.bar(results.keys(), results.values())
    plt.title('Сравнение моделей')
    plt.ylabel('Точность')
    plt.xticks(rotation=45)

    # 2. Визуализация графа 2D или 3D
    if use_3d:
        visualize_graph_3d(graph_data)
    else:
        visualize_graph_2d(graph_data, with_labels)

    visualize_feature_importance(feature_weights, feature_names, filename)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

def visualize_parameters_comparison(filenames=None, show=True):
    fig = plt.figure(figsize=(18, 6))
    for i, file_name in enumerate(filenames, start=1):
        plt.subplot(1, len(filenames), i)
        plt.axis('off')
        try:
            img = mpimg.imread(file_name)
        except Exception as e:
            print(f'Error while reading image from {file_name}: {e}')
        else:
            plt.imshow(img)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)
