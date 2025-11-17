## Описание
***
### Проект на основе лабораторной работы "Предсказание характеристик профиля с помощью графовых нейронных сетей (GNN)"

**Цель работы**: Использовать графовые нейронные сети для выявления ботов в 
социальной сети ВКонтакте. Исследовать, как структура связей между пользователями влияет на вероятность того, что профиль является ботом.

В данном проекте представлены скрипты для сбора данных пользователей из социальной сети ВКонтакте, обучения и тестирования моделей GNN, а также представления и анализа полученных результатов.

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
│   ├───for_inference
│   │       users_data.json
│   │
│   ├───for_model_1
│   │       bots_data.json
│   │       bots_ids.json
│   │       humans_data.json
│   │       humans_ids.json
│   │
│   └───for_model_2
├───data_collection
│   │   download_bots_json.py
│   │   get_group_members.py
│   │   vk_data_collector.py
│
├───data_processing
│   │   data_processor.py
│   │   feature_processor.py
│   │   file_manager.py
│
├───gnn_models
│   │   gnn.py
│   │   metrics_calc.py
│   │
│   ├───model_1
│   │   │   model.py
│   │   │   params.csv
│   │
│   ├───model_2
│   ├───model_inductive
│   │   │   graphsage_predictor.py
│   │   │   graphsage_predictor_tester.py
│   │   │   graphsage_predictor_trainer.py
│   │
│   ├───model_transductive
│   │   │   model_tester.py
│   │   │   model_trainer.py
│
├───saves
│   │   best_bot_detector_gnn.pth
│   │   GAT.png
│   │   GAT_mistakes.json
│   │   GCN.png
│   │   GCN_mistakes.json
│   │   inductive_gnn.pth
│   │   inductive_training.png
│   │   SAGE.png
│   │   SAGE_mistakes.json
│   │   scaler.pkl
│   │   shap_feature_importance.png
│   │   shap_summary.png
│   │   used_bots_ids.txt
│   │   used_humans_ids.txt
│   │
│   ├───dependence_plots
│   │
│   └───explanation_plots
│
├───shap_analysis
│   │   deep_values_computer.py
│   │   explanation_constructor.py
│   │   explanation_pipeline.py
│   │   gradient_values_computer.py
│   │   kernel_values_computer.py
│   │   values_computer.py
│
├───utilities
│       open_profile_by_id.py
│       vk_token_helper.py
│
├───viz
│   │   graph_viz.py
│   │   roc_viz.py
│   │   shap_viz.py
```

## Инструкция по работе с проектом
***
1. `git clone https://github.com/PurreCoder/vk_bot_detection`
2.  `pip install -r requirements.txt`
    - Если отсутствуют необходимые для работы проекта модули, попробовать доустановить их командой `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html`
    - Продолжить и завершить установку
3. Получить access token с помощью скрипта `utilities/vk_token_helper.py`, запустив его <u>из директории utilities</u>
4. Записать полученный access token в `config.py`
5. Обеспечить наличие JSON с id ботов `data/for_model_1/bots_ids.json`
    - В JSON файле список id соответствует ключу "items"
    - Можно воспользоваться скриптом `data_collection/download_bots_json.py`
    - В случае, если сайт дает список ботов только по Captcha, скачать JSON вручную по URL из файла скрипта `download_bots_json.py`
6. Обеспечить наличие JSON с id подлинных пользователей `data/for_model_1/humans_ids.json`
    - Для этого предлагается найти закрытое **сообщество** ВК, в котором вы знаете участников, и скопировать id сообщества
    - Вставить id в переменную `GROUP_ID` скрипта `data_collection/get_group_members.py`
    - Запустить скрипт `data_collection/get_group_members.py` <u>из директории data_collection</u>
7. Запустить скрипт `data_collection/vk_data_collector.py` <u>из директории data_collection</u>
    - Скрипт через VK API подтянет данные пользователей по их id и запишет в JSON файлы в папку for_model_1
    - Создадутся два отдельных файла: `bots_data.json` и `humans_data.json`
8. В корне проекта создать папку `saves`
9. Запустить `main.py` для обучения и тестирования модели с отображением результатов. В папке `saves` будут сохранены:
   - Для всех моделей диаграммы самых значимых признаков, выбранных с помощью метода, написанного авторами проекта
   - Summary plot (beeswarm plot) shap-значений для одной из моделей (по умолчанию `SAGE`)
   - Топ-15 самых значимых признаков с точки зрения shap-анализа для выбранной модели
   - Графики взаимных зависимостей признаков на основе анализа shap-значений (в подпапке `dependence_plots`) 
   - JSON-файлы с ошибками (ложными предсказаниями) GCN, GAT, GraphSAGE на тестовой выборке. Для каждого id будет указан прогноз, который сделала модель
   - Графики, объясняющие, какие признаки повлияли на то, что выбранная модель (по умолчанию `GAT`) сделала ложное предсказание (в подпапке `explanation_plots`)
10. Для выбора трансдуктивных моделей (обрабатываются в `model_tester.py`) вызвать `ModelTester()` в `main.py`. Для анализа:
    - Можно установить `use_3d=True` в вызове метода `visualize_menu` в `model_tester.py`
    - Также для исследования графов можно:
       - В `model_tester.py` в вызове метода `visualize_menu` установить `with_labels=True` для отображения меток для узлов графа
       - Параллельно `main.py` запустить скрипт `utilities/open_profile_by_id.py`
       - Выбрать интересующие метки и ввести их в одну строку через пробел в терминал скрипта `open_profile_by_id.py`
    - Для выбора SHAP-анализа интересующей модели (`GCN`, `GAT`, `GraphSAGE`), указать её в качестве аргумента `self.add_shap_analysis(...)` в теле метода `process_models` в `model_tester.py`
    - В вызове метода `add_shap_analysis` также можно выбрать способ подсчёта значений (`gradient`, `deeplift`, `kernel`) 
    - Имеет смысл настраивать параметры для метода `kernel` в вызове `add_shap_analysis`:
        - `background_size` отвечает за объем данных, на которых Explainer "щупает" нашу модель (можно уменьшать)
        - `test_size` отвечает за объем тестовых данных, которые Explainer будет "объяснять"
        - `n_samples` для KernelExplainer: кол-во отклоненных оцениваний для каждого предсказания. Не ставить меньше удвоенного кол-ва признаков плюс один!
        - Значения по умолчанию рассчитаны на очень долгий и точный расчет
11. Для выбора индуктивной модели:
    - Создать папку `data/for_inference`
    - В `main.py` оставить лишь вызов `GraphSAGETrainer()`, запустить `main.py`. Произойдет обучение индуктивной модели
    - В `main.py` оставить лишь вызов `check_predictions()`
    - Зафиксировать интересующие id пользователей ВК в списке `id_list` в `check_predictions()`
    - Запустить `main.py`. Будут показаны предсказания нейронной сети
12. Проект поддерживает как cpu, так и cuda в качестве устройства для вычислений

### Полезный контекст
В рамках использования данного проекта потребуется получение id пользователей или сообществ.

Безусловно, для этого можно воспользоваться специальным сервисом.
Более доступный способ:
* Открыть медиа-контент интересующего пользователя/сообщества
* Скопировать из адресной строки число, заключенное между "photo" и "_"

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

