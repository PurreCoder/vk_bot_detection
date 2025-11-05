import ast
import webbrowser

BOTS_LIST_SIZE = 0

def get_element_from_file(file_name, index):
    global BOTS_LIST_SIZE

    with open(file_name, 'r') as f:
        ids_str = f.read()
    ids = ast.literal_eval(ids_str)

    user_id = -1
    if index < len(ids):
        user_id = ids[index]
    else:
        BOTS_LIST_SIZE = len(ids)
    return user_id

def main():
    while True:
        index = int(input())
        if index <= 0:
            break

        user_id = get_element_from_file('../saves/used_bots_ids.txt', index)
        if user_id == -1:
            user_id = get_element_from_file('../saves/used_humans_ids.txt', index - BOTS_LIST_SIZE)
        if user_id == -1:
            print('Sorry, but no can do :(')
            continue

        user_url = f'https://vk.com/id{user_id}'
        webbrowser.open(user_url)

if __name__ == '__main__':
    main()
