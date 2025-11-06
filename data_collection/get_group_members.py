import requests
import config


def get_group_members_simple(access_token, group_id, count=1000):
    """
    Простая функция для получения участников группы
    """
    url = "https://api.vk.com/method/groups.getMembers"

    params = {
        'access_token': access_token,
        'group_id': group_id,
        'count': min(count, 1000),
        'fields': 'first_name,last_name,domain',
        'v': '5.131'
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if 'response' in data:
            members = data['response']['items']
            print(f"Найдено участников: {len(members)}")
            return members
        else:
            error = data.get('error', {})
            print(f"Ошибка: {error.get('error_msg')}")
            return []

    except Exception as e: # noqa
        print(f"Ошибка: {e}")
        return []


if __name__ == "__main__":
    ACCESS_TOKEN = config.ACCESS_TOKEN
    GROUP_ID = "77402688" # сообщество (не беседа)

    members = get_group_members_simple(ACCESS_TOKEN, GROUP_ID, count=275)
    print(','.join([str(member['id']) for member in members]))