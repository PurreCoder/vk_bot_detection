import matplotlib.pyplot as plt


def visualize_feature_importance(feature_weights, feature_names):
    plt.subplot(1, 3, 3)

    if feature_weights is not None:
        plt.barh(feature_names, feature_weights)
        plt.title('Топ-10 важных признаков')
    else:
        print(f"Feature importance visualization failed.")
        plt.text(0.5, 0.5, 'Feature importance\nnot available',
                 ha='center', va='center', transform=plt.gca().transAxes)


def finish_visualization():
    plt.tight_layout()
    plt.savefig('bot_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_menu(graph_data, results):
    plt.figure(figsize=(15, 5))

    # 1. Меню сравнения моделей
    plt.subplot(1, 3, 1)
    plt.bar(results.keys(), results.values())
    plt.title('Сравнение моделей')
    plt.ylabel('Точность')
    plt.xticks(rotation=45)

    # 2. Визуализация графа (упрощенная)
    plt.subplot(1, 3, 2)
    try:
        import networkx as nx
        # Создаем упрощенный граф для визуализации
        g = nx.Graph()
        edges = graph_data.edge_index.t().numpy()

        # Берем только часть ребер для наглядности
        sample_edges = edges  # [:min(2144, len(edges))]
        g.add_edges_from(sample_edges)

        node_colors = []
        for i in range(graph_data.num_nodes):  # min(200, graph_data.num_nodes)):
            if graph_data.y[i] == 0:
                node_colors.append('red')  # Боты
            else:
                node_colors.append('blue')  # Люди

        pos = nx.spring_layout(g)
        nx.draw(g, pos, node_color=node_colors[:len(g.nodes)],
                node_size=50, with_labels=False, alpha=0.7)
        plt.title('Граф (красные - боты, синие - люди)')
    except Exception as e:
        print(f"Graph visualization failed: {e}")
        plt.text(0.5, 0.5, 'Graph visualization\nnot available',
                 ha='center', va='center', transform=plt.gca().transAxes)
