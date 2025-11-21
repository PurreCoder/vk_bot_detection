import config


def sieve_deactivated(bots_users, humans_users):
    if config.FLAGS['FILTER_DEACTIVATED']:
        is_not_deactivated = lambda _user: _user.get('deactivated', '') == ''
        filter_deactivated = lambda _users: list(filter(is_not_deactivated, _users))

        bots_users = filter_deactivated(bots_users)
        print(f'\nВыделено {len(bots_users)} ботов в соответствии с фильтром')
        humans_users = filter_deactivated(humans_users)
        print(f'Выделено {len(humans_users)} людей в соответствии с фильтром\n')

    return bots_users, humans_users

def balance_users(bots_users, humans_users):
    if config.BOTS_TO_USERS > 1 or len(bots_users) != len(humans_users):
        common_size = min(len(bots_users), len(humans_users))
        humans_limit = int(common_size / config.BOTS_TO_USERS)

        if len(humans_users) > humans_limit:
            humans_users = humans_users[:humans_limit]

        if len(bots_users) > common_size:
            bots_users = bots_users[:common_size]

        print(f'Общая выборка будет содержать {len(humans_users)} людей и {len(bots_users)} ботов\n')

    return bots_users, humans_users
