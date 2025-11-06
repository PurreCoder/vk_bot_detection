import numpy as np

class Model:
    profile_traits = ('blacklisted', 'site', 'status', 'about', 'activities', 'interests', 'music', 'movies',
                      'tv', 'books', 'games', 'quotes', 'mobile_phone', 'home_phone', 'relation', 'has_photo', 'bdate')

    counters = ('albums', 'videos', 'photos', 'notes', 'friends', 'gifts', 'groups', 'subscriptions')

    feature_names = (
        'sex', 'last_seen', 'followers_count', *profile_traits, *counters,
        'alcohol', 'inspired_by', 'life_main', 'people_main', 'smoking',
        'has_education', 'occupation', 'universities_count', 'schools_count', 'languages'
    )

    features_count = len(feature_names)

    SIMILARITY_THRESHOLD = 0.9

    @staticmethod
    def extract_features_and_ids(users_list):
        """Извлекает признаки из данных пользователей"""
        features = []
        user_ids = []

        for user in users_list:
            feature_vector = [user.get('sex', 3) - 1]

            # Признаки активности
            last_seen = user.get('last_seen', {})
            feature_vector.append(last_seen.get('time', 0) if last_seen else 0)
            feature_vector.append(user.get('followers_count', 0))

            # Признаки профиля
            for trait in Model.profile_traits:
                feature_vector.append(1 if user.get(trait, 0) else 0)

            # Счётчики
            counters = user.get('counters', {})
            for counter in Model.counters:
                feature_vector.append(counters.get(counter, 0))

            # Личное
            personal = user.get('personal', {})
            feature_vector.append(personal.get('alcohol', 0))
            feature_vector.append(1 if personal.get('inspired_by') else 0)
            feature_vector.append(user.get('life_main', 0))
            feature_vector.append(user.get('people_main', 0))
            feature_vector.append(user.get('smoking', 0))

            # Образование и занятость
            education = user.get('education', {})
            feature_vector.append(1 if education else 0)

            occupation = user.get('occupation', {})
            feature_vector.append(1 if occupation else 0)

            feature_vector.append(len(user.get('universities', [])))
            feature_vector.append(len(user.get('schools', [])))
            feature_vector.append(len(user.get('langs', [])))

            features.append(feature_vector)
            user_ids.append(user.get('id'))

        return np.array(features), user_ids

    @staticmethod
    def extract_features(users_list):
        features, _ = Model.extract_features_and_ids(users_list)
        return features
