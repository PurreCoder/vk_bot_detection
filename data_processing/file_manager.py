import json
import os


def load_json_data(users_file):
    try:
        with open(users_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (IOError, OSError) as e:
        print(e)
    return data

def load_users(users_file):
    data = load_json_data(users_file)
    users = data.get('users', [])
    return users

def load_all_users(bots_file, humans_file):
    """Загружает и объединяет данные ботов и людей"""
    print("Загрузка данных...")

    bots_users = load_users(bots_file)
    humans_users = load_users(humans_file)

    print(f"Загружено ботов: {len(bots_users)}")
    print(f"Загружено людей: {len(humans_users)}")

    return bots_users, humans_users

def save_json_data(data: dict, file_name):
    try:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except (IOError, OSError) as e:
        print(e)

def serialize_object(obj, file_name):
    import pickle
    try:
        with open(file_name, 'wb') as obj_file:
            pickle.dump(obj, obj_file)
    except (IOError, OSError) as e:
        print(e)

def deserialize_object(file_name):
    try:
        import pickle
        with open(file_name, 'rb') as f:
            obj = pickle.load(f)
    except (IOError, OSError) as e:
        obj = None
    return obj

def save_array(data_array, file_name):
    try:
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(str(data_array))
    except (IOError, OSError) as e:
        print(e)

def prepare_clean_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def ensure_file_deleted(file_name):
    try:
        os.remove(file_name)
    except FileNotFoundError:
        pass
