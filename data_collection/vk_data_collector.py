import requests
import json
import time
import config


class VKDataCollector:
    def __init__(self, access_token):
        """
        Инициализация сборщика данных

        Args:
            access_token (str): Access token для VK API
        """
        self.access_token = access_token
        self.base_url = "https://api.vk.com/method/"
        self.version = "5.131"

    def download_ids_file(self, url, output_file='../bots_ids.json'):
        """
        Скачивает JSON файл с ID пользователей

        Args:
            url (str): URL для скачивания JSON файла
            output_file (str): Имя файла для сохранения

        Returns:
            dict: Данные из JSON файла или None в случае ошибки
        """
        try:
            print(f"Скачиваем файл с ID пользователей с {url}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"Файл успешно скачан и сохранен как {output_file}")
            print(f"Найдено пользователей: {data.get('count', 0)}")
            print(f"Timestamp: {data.get('timestamp', 'N/A')}")

            return data

        except requests.exceptions.RequestException as e:
            print(f"Ошибка при скачивании файла: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Ошибка при парсинге JSON: {e}")
            return None
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")
            return None

    def load_ids_from_file(self, filename='../bots_ids.json'):
        """
        Загружает ID пользователей из JSON файла

        Args:
            filename (str): Имя файла с ID пользователей

        Returns:
            list: Список ID пользователей или None в случае ошибки
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            items = data.get('items', [])
            count = data.get('count', 0)
            timestamp = data.get('timestamp', 'N/A')

            print(f"Загружено ID пользователей: {len(items)}")
            print(f"Общее количество: {count}")
            print(f"Timestamp: {timestamp}")

            return items

        except FileNotFoundError:
            print(f"Файл {filename} не найден")
            return None
        except json.JSONDecodeError as e:
            print(f"Ошибка при чтении JSON файла: {e}")
            return None
        except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")
            return None

    def get_user_info(self, params_string, user_id):
        """
        Получает информацию о пользователе через VK API

        Args:
            user_id (str/int): ID пользователя ВК

        Returns:
            dict: Информация о пользователе или None в случае ошибки
        """

        params = {'access_token': self.access_token, 'v': self.version, 'user_ids': str(user_id),
                  'fields': params_string}
        try:
            response = requests.get(f"{self.base_url}users.get", params=params, timeout=10)
            data = response.json()

            if 'response' in data and len(data['response']) > 0:
                user_data = data['response'][0]
                user_data['profile_url'] = f"https://vk.com/{user_data.get('domain', f'id{user_id}')}"
                user_data['collected_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
                user_data['original_id'] = str(user_id)
                return user_data
            else:
                error_msg = data.get('error', {})
                print(f"Ошибка API для ID {user_id}: {error_msg.get('error_msg', 'Unknown error')}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Ошибка сети для ID {user_id}: {e}")
            return None
        except Exception as e:
            print(f"Неожиданная ошибка для ID {user_id}: {e}")
            return None

    def save_users_to_json(self, data, filename='../vk_users_data.json'):
        """
        Сохраняет данные пользователей в JSON файл

        Args:
            data (list): Список с данными пользователей
            filename (str): Имя файла для сохранения
        """
        try:
            output_data = {
                'total_collected': len(data),
                'collection_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'users': data
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"Данные {len(data)} пользователей сохранены в файл: {filename}")
        except Exception as e:
            print(f"Ошибка при сохранении файла: {e}")

    def collect_users_data(self, user_ids, n_users=100, params_file='../gnn_models/model_1/params.csv', output_file='../vk_users_data.json', delay=0.5):
        """
        Основная функция для сбора данных

        Args:
            user_ids (list): Список ID пользователей
            n_users (int): Количество пользователей для сбора (первые N)
            params_file(str): Файл, откуда читается список выгружаемых для пользователя параметров
            output_file (str): Имя выходного файла
            delay (float): Задержка между запросами (в секундах)

        Returns:
            list: Собранные данные
        """

        params_string = ''
        with open(params_file, 'r', encoding='utf-8') as f:
            params_string = f.read().replace('\n', '')

        if params_string == '':
            print('Список собираемых параметров пользователя пуст')
            return []

        users_to_process = user_ids[:n_users]
        users_data = []

        print(f"Начинаем сбор данных для первых {n_users} пользователей из {len(user_ids)}...")

        for i, user_id in enumerate(users_to_process, 1):
            print(f"Обрабатываем {i}/{len(users_to_process)}: ID {user_id}")

            user_data = self.get_user_info(params_string, user_id)
            if user_data:
                users_data.append(user_data)
                print(f"✓ Получены данные: {user_data.get('first_name', '')} {user_data.get('last_name', '')}")
            else:
                print(f"✗ Не удалось получить данные для ID {user_id}")

            # Задержка для избежания блокировки
            time.sleep(delay)

        # Сохраняем данные
        self.save_users_to_json(users_data, output_file)

        print(f"\nСбор данных завершен! Успешно обработано: {len(users_data)}/{len(users_to_process)}")
        return users_data


def main():
    ACCESS_TOKEN = config.ACCESS_TOKEN
    N_USERS_TO_COLLECT = 100  # Количество пользователей для сбора
    DELAY_BETWEEN_REQUESTS = 0.8  # Задержка между запросами в секундах

    if ACCESS_TOKEN == "":
        print("Пожалуйста, укажите ваш access token для VK API!")
        print("Воспользуйтесь скриптом vk_token_helper.py")
        return

    collector = VKDataCollector(ACCESS_TOKEN)
    bots_ids = collector.load_ids_from_file('../data/for_model_1/bots_ids.json')

    if not bots_ids:
        print("Не удалось получить список ID пользователей. Завершение работы.")
        return

    collector.collect_users_data(
        user_ids=bots_ids,
        n_users=N_USERS_TO_COLLECT,
        params_file='../gnn_models/model_1/params.csv',
        output_file='../data/for_model_1/bots_data.json',
        delay=DELAY_BETWEEN_REQUESTS
    )

    users_ids = collector.load_ids_from_file('../data/for_model_1/humans_ids.json')

    if not users_ids:
        print("Не удалось получить список ID пользователей. Завершение работы.")
        return

    collector.collect_users_data(
        user_ids=users_ids,
        n_users=N_USERS_TO_COLLECT,
        params_file='../gnn_models/model_1/params.csv',
        output_file='../data/for_model_1/humans_data.json',
        delay=DELAY_BETWEEN_REQUESTS
    )


if __name__ == "__main__":
    main()