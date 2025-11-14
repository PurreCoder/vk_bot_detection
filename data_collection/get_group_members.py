import requests
import config


def get_group_members(access_token, group_id, count=1000, batch_size=1000):
    """
    Простая функция для получения участников группы
    """
    url = "https://api.vk.com/method/groups.getMembers"

    params = {
        'access_token': access_token,
        'group_id': group_id,
        'offset': 0,
        'count': batch_size,
        'fields': 'first_name,last_name,domain',
        'v': '5.131'
    }

    offset = 0
    all_members = []

    while True:
        try:
            params['offset'] = offset
            params['count'] = min(batch_size, count - offset)
            response = requests.get(url, params=params)
            data = response.json()

            if 'response' not in data:
                error = data.get('error', {})
                print(f"Ошибка: {error.get('error_msg')}")
                return []

            new_members = data['response']['items']

            if offset == 0:
                total_count = data['response']['count']
                print(f"Всего участников в группе: {total_count}")
                total_count = min(total_count, count)
                print(f"Будет выкачано: {total_count}")

            all_members.extend(new_members)
            print(f"Получено {len(new_members)} участников ({len(all_members)} / {total_count})")

            # Проверяем, достигли ли конца
            if len(new_members) == 0 or offset + len(new_members) >= total_count:
                break

            offset += batch_size

        except Exception as e: # noqa
            print(f"Ошибка: {e}")
            return []

    return all_members


if __name__ == "__main__":
    ACCESS_TOKEN = config.ACCESS_TOKEN
    GROUP_ID = "201766823" # сообщество (не беседа)

    members_list = get_group_members(ACCESS_TOKEN, GROUP_ID, count=1005)

    with open("group_members.txt", "w") as f:
        f.writelines(str(member['id']) + ',' for member in members_list)

    print(','.join([str(member['id']) for member in members_list]))