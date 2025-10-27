import webbrowser
import requests


def get_token_interactive():
    """
    Интерактивное получение токена через OAuth
    """
    print("=" * 60)
    print("ПОЛУЧЕНИЕ ACCESS TOKEN VK")
    print("=" * 60)

    client_id = "51527129" # id из ВК
    scope = "friends,photos,status,video,stories,pages,notes,wall,groups,offline"

    auth_url = (
        f"https://oauth.vk.com/authorize?"
        f"client_id={client_id}&"
        f"display=page&"
        f"redirect_uri=https://oauth.vk.com/blank.html&"
        f"scope={scope}&"
        f"response_type=token&"
        f"v=5.131"
    )

    print("1. Открываю браузер для авторизации...")
    print("2. Войдите в свой аккаунт VK")
    print("3. Разрешите доступ приложению")
    print("4. Скопируйте access_token из адресной строки")
    print("=" * 60)

    webbrowser.open(auth_url)

    token = input("Введите access_token: ").strip()

    if token:
        if test_token(token):
            print("✓ Токен прошел проверку!")
            return token
        else:
            print("✗ Токен не прошел проверку!")
            return None
    return None

def test_token(token_to_test):
    """Проверяет валидность токена"""

    url = "https://api.vk.com/method/users.get"
    params = {
        'access_token': token_to_test,
        'v': '5.131'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if 'response' in data:
            return True
    except Exception: # noqa
        pass
    return False


if __name__ == '__main__':
    get_token_interactive()
