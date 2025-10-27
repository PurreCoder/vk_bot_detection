import numpy as np

class Model:
    profile_traits = ('site', 'status', 'about', 'activities', 'interests', 'interests',
                           'music', 'movies', 'tv', 'books', 'games', 'quotes')
    feature_names = (
        'sex', 'blacklisted', *profile_traits, 'last_seen', 'followers_count',
        'has_education', 'universities_count', 'schools_count'
    )

    @staticmethod
    def extract_features(users_list, label):
        """Извлекает признаки из данных пользователей"""
        features = []
        user_ids = []

        for user in users_list:
            feature_vector = [user.get('sex', 0), 1 if user.get('blacklisted', 0) else 0]

            # Признаки активности
            last_seen = user.get('last_seen', {})
            feature_vector.append(last_seen.get('time', 0) if last_seen else 0)
            feature_vector.append(user.get('followers_count', 0))

            # Признаки профиля
            for trait in Model.profile_traits:
                feature_vector.append(1 if user.get(trait, 0) else 0)

            # Образование
            education = user.get('education', {})
            feature_vector.append(1 if education else 0)
            feature_vector.append(len(user.get('universities', [])))
            feature_vector.append(len(user.get('schools', [])))

            features.append(feature_vector)
            user_ids.append(user.get('id'))

        return np.array(features), [label] * len(features), user_ids