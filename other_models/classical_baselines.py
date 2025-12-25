import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from typing import Dict, List, Tuple, Any, Optional

import config
from data_processing.data_filter import sieve_deactivated, balance_users
from data_processing.data_processor import DataProcessor
from data_processing.file_manager import load_all_users
from gnn_models.model_1.model import Model as my_model


class ClassicalMLBaselines:
    """Класс для сравнения классических ML методов без использования графовой структуры"""

    def __init__(self, all_features: List[Dict[str, Any]], all_labels: List[int],
                 test_size: float = 0.3, random_state: int = 42):
        """
        Args:
            all_features: Список словарей с признаками пользователей
            all_labels: Список меток (0 - человек, 1 - бот)
            test_size: Доля тестовой выборки
            random_state: Seed для воспроизводимости
        """
        self.all_features = all_features
        self.all_labels = np.array(all_labels)
        self.test_size = test_size
        self.random_state = random_state

        # Для хранения результатов
        self.results = {}
        self.models = {}
        self.feature_names = []

        # Подготовка данных
        self._prepare_data()

    def _prepare_data(self):
        """Подготовка данных для классических ML методов"""
        print("Подготовка данных...")

        # Преобразуем список словарей в DataFrame
        df = pd.DataFrame(self.all_features, columns=my_model.feature_names)
        self.feature_names = list(df.columns)

        # Определяем типы признаков
        self.categorical_features = []
        self.numerical_features = []

        # for col in df.columns:
        #     if df[col].dtype == 'object' or (df[col].nunique() < 20 and df[col].dtype != 'float64'):
        #         self.categorical_features.append(col)
        #     else:
        #         self.numerical_features.append(col)

        self.categorical_features = []
        self.numerical_features = [col for col in df.columns]

        print(f"Всего признаков: {len(self.feature_names)}")
        print(f"Категориальные признаки ({len(self.categorical_features)}): {self.categorical_features}")
        print(f"Числовые признаки ({len(self.numerical_features)}): {self.numerical_features}")

        # Кодирование категориальных признаков
        # self.label_encoders = {}
        # for col in self.categorical_features:
        #     if df[col].dtype == 'object':
        #         le = LabelEncoder()
        #         df[col] = le.fit_transform(df[col].astype(str))
        #         self.label_encoders[col] = le

        # Масштабирование числовых признаков
        #if self.numerical_features:
        #    self.scaler = StandardScaler()
        #    df[self.numerical_features] = self.scaler.fit_transform(df[self.numerical_features])

        # Разделение на train/test
        self.X = df.values
        self.y = self.all_labels

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )

        print(f"Train размер: {len(self.X_train)}")
        print(f"Test размер: {len(self.X_test)}")
        print(f"Распределение классов в train: {np.bincount(self.y_train)}")
        print(f"Распределение классов в test: {np.bincount(self.y_test)}")

    def train_random_forest(self, n_estimators: int = 100, max_depth: int = 10):
        """Обучение Random Forest"""
        print("\n" + "=" * 50)
        print("Обучение Random Forest")
        print("=" * 50)

        start_time = time.time()

        # Инициализация модели
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

        # Обучение
        rf_model.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time

        # Предсказания
        y_pred = rf_model.predict(self.X_test)
        y_pred_proba = rf_model.predict_proba(self.X_test)[:, 1]

        # Вычисление метрик
        metrics = self._compute_metrics(self.y_test, y_pred, y_pred_proba, train_time)

        # Анализ важности признаков
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.models['RandomForest'] = rf_model
        self.results['RandomForest'] = {
            'metrics': metrics,
            'feature_importance': feature_importance.to_dict('records'),
            'params': {
                'n_estimators': n_estimators,
                'max_depth': max_depth
            }
        }

        print(f"Время обучения: {train_time:.2f} сек")
        self._print_metrics(metrics)

        return rf_model, metrics, feature_importance

    def train_catboost(self, iterations: int = 100, depth: int = 6, learning_rate: float = 0.1):
        """Обучение CatBoost"""
        print("\n" + "=" * 50)
        print("Обучение CatBoost")
        print("=" * 50)

        start_time = time.time()

        # Определяем индексы категориальных признаков для CatBoost
        cat_features_indices = [
            i for i, col in enumerate(self.feature_names)
            if col in self.categorical_features
        ]

        # Инициализация модели
        cat_model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_seed=self.random_state,
            verbose=False,
            auto_class_weights='Balanced',
            cat_features=cat_features_indices
        )

        # Обучение
        cat_model.fit(
            self.X_train, self.y_train,
            eval_set=(self.X_test, self.y_test),
            verbose=50  # Показывать каждые 50 итераций
        )
        train_time = time.time() - start_time

        # Предсказания
        y_pred = cat_model.predict(self.X_test)
        y_pred_proba = cat_model.predict_proba(self.X_test)[:, 1]

        # Вычисление метрик
        metrics = self._compute_metrics(self.y_test, y_pred, y_pred_proba, train_time)

        # Анализ важности признаков
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': cat_model.get_feature_importance()
        }).sort_values('importance', ascending=False)

        self.models['CatBoost'] = cat_model
        self.results['CatBoost'] = {
            'metrics': metrics,
            'feature_importance': feature_importance.to_dict('records'),
            'params': {
                'iterations': iterations,
                'depth': depth,
                'learning_rate': learning_rate
            }
        }

        print(f"Время обучения: {train_time:.2f} сек")
        self._print_metrics(metrics)

        return cat_model, metrics, feature_importance

    def _compute_metrics(self, y_true, y_pred, y_pred_proba, train_time):
        """Вычисление метрик"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'train_time': train_time,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

        # ROC-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            metrics['fpr'] = fpr.tolist()
            metrics['tpr'] = tpr.tolist()
        except:
            metrics['roc_auc'] = 0.0

        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report

        return metrics

    def _print_metrics(self, metrics):
        """Вывод метрик"""
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics.get('roc_auc', 0):.4f}")

        # Confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        print(f"\nConfusion Matrix:")
        print(f"[[TN={cm[0, 0]:3d}  FP={cm[0, 1]:3d}]")
        print(f" [FN={cm[1, 0]:3d}  TP={cm[1, 1]:3d}]]")

    def compare_models(self):
        """Сравнение всех обученных моделей"""
        if not self.results:
            print("Нет обученных моделей для сравнения")
            return

        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ МОДЕЛЕЙ")
        print("=" * 60)

        comparison_df = pd.DataFrame({
            model_name: {
                'Accuracy': results['metrics']['accuracy'],
                'Precision': results['metrics']['precision'],
                'Recall': results['metrics']['recall'],
                'F1-Score': results['metrics']['f1'],
                'ROC-AUC': results['metrics'].get('roc_auc', 0),
                'Train Time (s)': results['metrics']['train_time']
            }
            for model_name, results in self.results.items()
        }).T

        print(comparison_df.round(4))

        # Определение лучшей модели по F1-Score
        best_model = comparison_df['F1-Score'].idxmax()
        print(f"\nЛучшая модель по F1-Score: {best_model} ({comparison_df.loc[best_model, 'F1-Score']:.4f})")

        return comparison_df

    def plot_feature_importance_comparison(self, top_n: int = 15):
        """Визуализация важности признаков для всех моделей"""
        if len(self.results) < 2:
            print("Нужно обучить минимум 2 модели для сравнения")
            return

        fig, axes = plt.subplots(1, len(self.results), figsize=(6 * len(self.results), 6))

        if len(self.results) == 1:
            axes = [axes]

        for idx, (model_name, results) in enumerate(self.results.items()):
            importance_df = pd.DataFrame(results['feature_importance'])
            top_features = importance_df.head(top_n)

            ax = axes[idx]
            ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.invert_yaxis()
            ax.set_xlabel('Важность признака')
            ax.set_title(f'{model_name} - Top {top_n} признаков')

        plt.tight_layout()
        plt.show()

    def plot_roc_curves(self):
        """Визуализация ROC-кривых для всех моделей"""
        if not self.results:
            print("Нет обученных моделей")
            return

        plt.figure(figsize=(10, 8))

        for model_name, results in self.results.items():
            metrics = results['metrics']
            if 'fpr' in metrics and 'tpr' in metrics:
                fpr = metrics['fpr']
                tpr = metrics['tpr']
                roc_auc = metrics.get('roc_auc', 0)

                plt.plot(fpr, tpr, lw=2,
                         label=f'{model_name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривые моделей')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()

    def cross_validate_model(self, model_name: str, cv: int = 5):
        """Кросс-валидация модели"""
        if model_name not in self.models:
            print(f"Модель {model_name} не обучена")
            return

        model = self.models[model_name]

        # Для CatBoost нужен специальный обработчик категориальных признаков
        if model_name == 'CatBoost':
            cat_features_indices = [
                i for i, col in enumerate(self.feature_names)
                if col in self.categorical_features
            ]

            # Создаем Pool для кросс-валидации
            train_pool = Pool(self.X, self.y, cat_features=cat_features_indices)

            cv_results = cat_model.cv(
                train_pool,
                fold_count=cv,
                verbose=False
            )

            cv_scores = {
                'test-Accuracy-mean': cv_results['test-Accuracy-mean'].mean(),
                'test-Precision-mean': cv_results['test-Precision-mean'].mean(),
                'test-Recall-mean': cv_results['test-Recall-mean'].mean(),
                'test-F1-mean': cv_results['test-F1-mean'].mean(),
            }

        else:
            # Для Random Forest и других sklearn моделей
            scoring = ['accuracy', 'precision', 'recall', 'f1']
            cv_scores = {}

            for score in scoring:
                scores = cross_val_score(
                    model, self.X, self.y,
                    cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
                    scoring=score,
                    n_jobs=-1
                )
                cv_scores[f'{score}-mean'] = scores.mean()
                cv_scores[f'{score}-std'] = scores.std()

        print(f"\nКросс-валидация ({cv} folds) для {model_name}:")
        for metric, value in cv_scores.items():
            print(f"{metric}: {value:.4f}")

        return cv_scores

    def save_results(self, filename: str = 'ml_baselines_results.json'):
        """Сохранение результатов в файл"""
        import datetime

        results_to_save = {
            'timestamp': datetime.datetime.now().isoformat(),
            'data_info': {
                'total_samples': len(self.X),
                'train_size': len(self.X_train),
                'test_size': len(self.X_test),
                'features': self.feature_names,
                'categorical_features': self.categorical_features,
                'numerical_features': self.numerical_features
            },
            'test_size': self.test_size,
            'random_state': self.random_state,
            'results': self.results
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)

        print(f"\nРезультаты сохранены в {filename}")

        # Также сохраняем сравнение в CSV
        comparison_df = self.compare_models()
        if comparison_df is not None:
            comparison_df.to_csv('ml_baselines_comparison.csv')
            print("Таблица сравнения сохранена в ml_baselines_comparison.csv")


# Пример использования
if __name__ == "__main__":
    bots_users, humans_users = load_all_users('../' + config.DATA_SOURCE['BOTS_FILE'], '../' + config.DATA_SOURCE['HUMANS_FILE'])

    bots_users, humans_users = sieve_deactivated(bots_users, humans_users)
    bots_users, humans_users = balance_users(bots_users, humans_users)

    processor = DataProcessor(my_model)
    all_features, all_labels, all_ids =processor.get_all_features(bots_users, humans_users)

    # Инициализация
    ml_baselines = ClassicalMLBaselines(all_features, all_labels, test_size=0.3)

    # Обучение моделей
    rf_model, rf_metrics, rf_importance = ml_baselines.train_random_forest(
        n_estimators=100, max_depth=10
    )

    cat_model, cat_metrics, cat_importance = ml_baselines.train_catboost(
        iterations=200, depth=6, learning_rate=0.05
    )

    # Сравнение моделей
    comparison = ml_baselines.compare_models()

    # Визуализация
    ml_baselines.plot_feature_importance_comparison(top_n=10)
    ml_baselines.plot_roc_curves()

    # Кросс-валидация
    cv_scores_rf = ml_baselines.cross_validate_model('RandomForest', cv=5)
    cv_scores_cat = ml_baselines.cross_validate_model('CatBoost', cv=5)

    # Сохранение результатов
    ml_baselines.save_results()
