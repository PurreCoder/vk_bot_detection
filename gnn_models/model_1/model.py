import numpy as np

class Model:
    profile_traits = ('site', 'status', 'about', 'activities', 'interests', 'music', 'movies',
                      'tv', 'books', 'games', 'quotes', 'mobile_phone', 'home_phone', 'relation', 'has_photo', 'bdate')

    counters = ('albums', 'videos', 'photos', 'posts', 'friends', 'followers', 'gifts', 'groups', 'subscriptions', 'pages')

    feature_names = (
        'sex', 'last_seen', 'followers_count', 'deactivated', *profile_traits, *counters,
        'inspired_by', 'alcohol', 'life_main', 'people_main', 'smoking', 'religion',
        'occupation', 'universities', 'schools', 'languages'
    )

    features_count = len(feature_names)

    SIMILARITY_THRESHOLD = 0.9

    @staticmethod
    def extract_user_features(user):
        feature_vector = []

        if 'sex' in Model.feature_names:
            feature_vector.append(user.get('sex', 3) - 1)

        # Признаки активности
        if 'last_seen' in Model.feature_names:
            last_seen = user.get('last_seen', {})
            feature_vector.append(last_seen.get('time', 0) if last_seen else 0)
        if 'followers_count' in Model.feature_names:
            feature_vector.append(user.get('followers_count', 0))
        if 'deactivated' in Model.feature_names:
            del_reason = user.get('deactivated', '')
            feature_vector.append({'': 0, 'deleted': 1, 'banned': 2}.get(del_reason, 3))

        # Признаки профиля
        for trait in Model.profile_traits:
            feature_vector.append(1 if user.get(trait, 0) else 0)

        # Счётчики
        counters = user.get('counters', {})
        for counter in Model.counters:
            feature_vector.append(counters.get(counter, 0))

        # Личное
        personal = user.get('personal', {})
        if 'inspired_by' in Model.feature_names:
            feature_vector.append(1 if personal.get('inspired_by') else 0)
        for feature in ('alcohol', 'life_main', 'people_main', 'smoking'):
            if feature in Model.feature_names:
                feature_vector.append(personal.get(feature, 0))
        if 'religion' in Model.feature_names:
            feature_vector.append(1 if personal.get('religion') else 0)

        # Образование и занятость
        if 'occupation' in Model.feature_names:
            occupation = user.get('occupation', {})
            feature_vector.append(1 if occupation else 0)

        if 'universities' in Model.feature_names:
            feature_vector.append(len(user.get('universities', [])))
        if 'schools' in Model.feature_names:
            feature_vector.append(len(user.get('schools', [])))
        if 'languages' in Model.feature_names:
            feature_vector.append(len(personal.get('langs', [])))

        return feature_vector

    @staticmethod
    def extract_features_and_ids(users_list):
        """Извлекает признаки из данных пользователей"""
        features = []
        user_ids = []

        for user in users_list:
            feature_vector = Model.extract_user_features(user)
            features.append(feature_vector)
            user_ids.append(user.get('id'))

        return np.array(features), user_ids

    @staticmethod
    def extract_features(users_list):
        features, _ = Model.extract_features_and_ids(users_list)
        return features
