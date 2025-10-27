import requests
import json
import time


def download_bots_json(url, output_file='../bots_ids.json'):
    """
    Скачивает JSON файл с ID пользователей

    Args:
        url (str): URL для скачивания
        output_file (str): Имя файла для сохранения
    """
    try:
        print(f"Скачиваем файл с {url}...")

        # Добавляем заголовки чтобы избежать блокировки
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Проверяем структуру данных
        if 'items' in data and 'count' in data:
            # Сохраняем скачанный файл
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"✓ Файл успешно скачан и сохранен как {output_file}")
            print(f"✓ Найдено пользователей: {data.get('count', 0)}")
            print(f"✓ Timestamp: {data.get('timestamp', 'N/A')}")
            print(f"✓ Примеры ID: {data['items'][:5]}...")

            return True
        else:
            print("✗ Неверный формат JSON файла: отсутствуют обязательные поля 'items' или 'count'")
            return False

    except requests.exceptions.RequestException as e:
        print(f"✗ Ошибка при скачивании файла: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Ошибка при парсинге JSON: {e}")
        return False
    except Exception as e:
        print(f"✗ Неожиданная ошибка: {e}")
        return False


def main():
    """
    Основная функция для скачивания JSON файла
    """
    BOTS_JSON_URL = "https://api.botnadzor.org/bots"
    OUTPUT_FILE = "../bots_ids.json"

    success = download_bots_json(BOTS_JSON_URL, OUTPUT_FILE)

    if success:
        print(f"\nФайл {OUTPUT_FILE} готов для использования основным скриптом.")
    else:
        print("\nНе удалось скачать файл. Проверьте URL и подключение к интернету.")


if __name__ == "__main__":
    main()