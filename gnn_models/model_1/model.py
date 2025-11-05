import numpy as np

class Model:
    profile_traits = ('blacklisted', 'site', 'status', 'about', 'activities', 'interests', 'music', 'movies',
                      'tv', 'books', 'games', 'quotes', 'mobile_phone', 'home_phone', 'relation')
    feature_names = (
        'sex', 'last_seen', 'followers_count', *profile_traits,
        'alcohol', 'inspired_by', 'life_main', 'people_main', 'smoking',
        'has_education', 'occupation', 'universities_count', 'schools_count', 'languages'
    )

    SIMILARITY_THRESHOLD = 0.75

    @staticmethod
    def extract_features(users_list, label):
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

        return np.array(features), [label] * len(features), user_ids