import numpy as np
import config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from data_processing.minmax_transformer import MinMaxTransformer


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if args[0] not in cls._instances:
             _instance = cls.__new__(cls, *args, **kwargs)
             _instance.__init__(*args, **kwargs)
             cls._instances[args[0]] = _instance
        return cls._instances[args[0]]


class FeatureScaler:
    def __init__(self, model, limits):
        self.fit_performed = False
        features = model.feature_names
        limits = limits

        if config.SCALING == config.SCALING_STRATEGY.STANDARD:
            first_transformer = ('StdScaler', StandardScaler())
        elif config.SCALING == config.SCALING_STRATEGY.MIXED:
            MyColumnTransformer = self.create_column_transformer(features, limits)
            first_transformer = ('ColumnTransformer', MyColumnTransformer)
        else:
            first_transformer = ('StdScaler', StandardScaler())
            print('Error: wrong scaling strategy. Selected StdScaler by default')

        # self.transformer = Pipeline([
        #     first_transformer,
        #     ('Normalizer L2', Normalizer(norm='l2'))])
        self.transformer = Pipeline([first_transformer])

    def create_minmax_transformer(self, features, limits, minmax_columns):
        Xmin = np.array([limits[features[col_ind]].min for col_ind in minmax_columns])
        Xmax = np.array([limits[features[col_ind]].max for col_ind in minmax_columns])
        MyMinMaxScaler = MinMaxTransformer(Xmin, Xmax)

        return MyMinMaxScaler

    def create_column_transformer(self, features, limits):
        transformers_list = []
        minmax_cur, rest_cur = [], []
        minmax_ind, rest_ind = 0, 0

        for ind, col_name in enumerate(features):
            if col_name in limits:
                minmax_cur.append(ind)
                if rest_cur:
                    transformers_list.append((f'Std{rest_ind}', StandardScaler(), rest_cur))
                    rest_cur = []
                    rest_ind += 1
            else:
                rest_cur.append(ind)
                if minmax_cur:
                    my_minmax = self.create_minmax_transformer(features, limits, minmax_cur)
                    transformers_list.append((f'MinMax{minmax_ind}', my_minmax, minmax_cur))
                    minmax_cur = []
                    minmax_ind += 1
        if rest_cur:
            transformers_list.append((f'Std{rest_ind}', StandardScaler(), rest_cur))
        if minmax_cur:
            my_minmax = self.create_minmax_transformer(features, limits, minmax_cur)
            transformers_list.append((f'MinMax{minmax_ind}', my_minmax, minmax_cur))

        MyColumnTransformer = ColumnTransformer(transformers_list)
        return MyColumnTransformer

    def fit_transform(self, data):
        if self.fit_performed:
            res = self.transformer.transform(data)
        else:
            self.fit_performed = True
            res = self.transformer.fit_transform(data)
        return res

    @classmethod
    def normalize(cls, feature_vectors):
        normalizer = Normalizer(norm='l2')
        return normalizer.fit_transform(feature_vectors)


class FeatureScalerSingleton(FeatureScaler, metaclass=SingletonMeta):
    pass
