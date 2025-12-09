import numpy as np
import config

class Model:
    profile_traits = ('site', 'status', 'about', 'activities', 'interests', 'music', 'movies',
                      'tv', 'books', 'games', 'quotes', 'mobile_phone', 'home_phone', 'relation', 'has_photo', 'bdate', 'city')

    counters = ('albums', 'videos', 'photos', 'posts', 'friends', 'followers', 'gifts', 'groups', 'subscriptions', 'pages')

    feature_names = (
        'sex', 'last_seen', 'followers_count', 'domain',
        *(('deactivated',) if not config.FLAGS['REMOVE_DEACTIVATED'] else ()),
        *profile_traits, *counters,
        'inspired_by', 'alcohol', 'life_main', 'people_main', 'smoking', 'religion',
        'occupation', 'universities', 'schools', 'languages'
    )

    features_count = len(feature_names)

    SIMILARITY_THRESHOLD = 0.95

    @staticmethod
    def extract_user_features(user):
        feature_dict = {}

        if 'sex' in Model.feature_names:
            feature_dict['sex'] = user.get('sex', 3) - 1

        # Признаки активности
        if 'last_seen' in Model.feature_names:
            last_seen = user.get('last_seen', {})
            feature_dict['last_seen'] = last_seen.get('time', 0) if last_seen else 0
        if 'followers_count' in Model.feature_names:
            feature_dict['followers_count'] = user.get('followers_count', 0)
        if 'domain' in Model.feature_names:
            feature_dict['domain'] = int(not(user.get('domain', 'id').startswith('id')))
        if 'deactivated' in Model.feature_names:
            del_reason = user.get('deactivated', '')
            feature_dict['deactivated'] = {'': 0, 'deleted': 1, 'banned': 2}.get(del_reason, 3)

        # Признаки профиля
        for trait in Model.profile_traits:
            feature_dict[trait] = 1 if user.get(trait, 0) else 0

        # Счётчики
        counters = user.get('counters', {})
        for counter in Model.counters:
            feature_dict[counter] = counters.get(counter, 0)

        # Личное
        personal = user.get('personal', {})
        if 'inspired_by' in Model.feature_names:
            feature_dict['inspired_by'] = 1 if personal.get('inspired_by') else 0
        for feature in ('alcohol', 'life_main', 'people_main', 'smoking'):
            if feature in Model.feature_names:
                feature_dict[feature] = personal.get(feature, 0)
        if 'religion' in Model.feature_names:
            feature_dict['religion'] = 1 if personal.get('religion') else 0

        # Образование и занятость
        if 'occupation' in Model.feature_names:
            occupation = user.get('occupation', {})
            feature_dict['occupation'] = 1 if occupation else 0

        if 'universities' in Model.feature_names:
            feature_dict['universities'] = len(user.get('universities', []))
        if 'schools' in Model.feature_names:
            feature_dict['schools'] = len(user.get('schools', []))
        if 'languages' in Model.feature_names:
            feature_dict['languages'] = len(personal.get('langs', []))

        return feature_dict

    @staticmethod
    def extract_feature_vector(user):
        feature_dict = Model.extract_user_features(user)
        return [feature_dict[key] for key in Model.feature_names]

    @staticmethod
    def extract_features_and_ids(users_list):
        """Извлекает признаки из данных пользователей"""
        features = []
        user_ids = []

        for user in users_list:
            features.append(Model.extract_feature_vector(user))
            user_ids.append(user.get('id'))

        return np.array(features), user_ids

    @staticmethod
    def extract_features(users_list):
        features, _ = Model.extract_features_and_ids(users_list)
        return features
