## Описание
***
### Проект на основе лабораторной работы "Предсказание характеристик профиля с помощью графовых нейронных сетей (GNN)"

**Цель работы**: Использовать графовые нейронные сети для выявления ботов в 
социальной сети ВКонтакте. Исследовать, как структура связей между пользователями (друзья, подписки, 
взаимодействия) влияет на вероятность того, что профиль является ботом.

В данном проекте представлены скрипты для сбора данных пользователей из социальной сети ВКонтакте, обучения и тестирования моделей GNN и представления полученных результатов.

## Структура проекта
***
```
VK_GNN_Research
│   config.py
│   main.py
│   README.md
│   requirements.txt
│
├───data
│   ├───for_model_1
│   │       bots_data.json
│   │       bots_ids.json
│   │       humans_data.json
│   │       humans_ids.json
│   │
│   └───for_model_2
├───data_collection
│       download_bots_json.py
│       get_group_members.py
│       vk_data_collector.py
│       vk_token_helper.py
│
└───gnn_models
    │   bot_detector_gnn.py
    │   bot_detector_trainer.py
    │   bot_gnn.py
    │   model_tester.py
    │
    ├───model_1
    │       model.py
    │       params.csv
    │
    └───model_2
```

## Установка и запуск
***
1. `git clone https://github.com/PurreCoder/vk_bot_detection`
2.  `pip install -r requirements.txt`
    - Если некоторые модули не установились, доустановить их командой `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html`
    - Продолжить и завершить установку
3. Получить access token с помощью скрипта vk_token_helper.py
4. Записать полученный access token в config.py
5. Обеспечить наличие JSON с id ботов `for_model_1/bots_ids.json`
    - В JSON файле список id соответствует ключу "items"
    - Можно воспользоваться скриптом `download_bots_json.py`
    - В случае, если сайт дает список ботов только по Captcha, скачать JSON вручную по URL из файла скрипта `download_bots_json.py`
6. Обеспечить наличие JSON с id подлинных пользователей `for_model_1/humans_ids.json`
    - Для этого предлагается найти закрытое **сообщество** ВК, в котором вы знаете участников, и скопировать id сообщества
    - Вставить id в переменную `GROUP_ID` скрипта `get_group_members.py`
    - Запустить скрипт `get_group_members.py`
7. Запустить скрипт `vk_data_collector.py`
    - Скрипт через VK API подтянет данные пользователей по их id и запишет в JSON файлы в папку for_model_1
    - Создадутся два отдельных файла: `bots_data.json` и `humans_data.json`
8. Запустить main.py для обучения и тестирования модели с отображением результатов

## Авторы
***
<table>
	<tr>
		<td align="center" valign="top">
			<a href="https://github.com/PurreCoder">
				<img src="https://avatars.githubusercontent.com/PurreCoder" width="80" height="80" alt=""/>
				<br>
				<sub><b>Немченко Денис</b></sub>
			</a>
			<br>
			<sub><b>Team lead</b></sub>
		</td>
		<td align="center" valign="top">
			<a href="https://github.com/AleksCombo">
				<img src="https://avatars.githubusercontent.com/AleksCombo" width="80" height="80" alt=""/>
				<br>
				<sub><b>Шкроб Александр</b></sub>
			</a>
			<br>
			<sub></sub>
		</td>
		<td align="center" valign="top">
			<a href="https://github.com/SoaringHedgehog">
				<img src="https://avatars.githubusercontent.com/SoaringHedgehog" width="80" height="80" alt=""/>
				<br>
				<sub><b>Ларионов Евгений</b></sub>
			</a>
			<br>
			<sub></sub>
		</td>
	</tr>
</table>

