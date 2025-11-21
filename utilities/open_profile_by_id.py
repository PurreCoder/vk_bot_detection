import ast
import webbrowser
import config

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
        index_list = list(map(int, input().split()))
        for index in index_list:
            if index <= 0:
                break

            user_id = get_element_from_file('../' + config.GRAPH_DATA_LOGS['BOTS_IDS_FILE'], index)
            if user_id == -1:
                user_id = get_element_from_file('../' + config.GRAPH_DATA_LOGS['HUMANS_IDS_FILE'], index - BOTS_LIST_SIZE)
            if user_id == -1:
                print('Sorry, but no can do :(')
                continue

            user_url = f'https://vk.com/id{user_id}'
            webbrowser.open(user_url)

if __name__ == '__main__':
    main()
