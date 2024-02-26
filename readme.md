# Zoneminder

## Первоначальная настройка контейнера

### Запуск контейнера
docker compose up -d

### Просмотр логов
docker compose logs -f
Ctrl + C - закрыть

### После старта контейнера (установка и обновление необходимых пакетов)
sh ./script.sh

### Изменение длительности нахождения объекта в кадре
Изменить переменную duration в my_event_start.sh, находящемуся по адресу (на выбор):
Хост: detector/_data
Контейнер: /var/lib/zmeventnotification/bin/my_detection


## Настройка zoneminder

### Сервис доступен по адресу
http://0.0.0.0/zm/
http://0.0.0.0:80/zm/

### Изменение rtsp адреса камеры
Console -> Monitor -> Monitor (по центру) -> Source -> Source Path	

### Изменение полигона зоны срабатывания
Console -> Monitor (Zones справа) -> ЛКМ по полигону

### Изменение таймера на срабатывание
Console -> Monitor -> Monitor -> MISC -> Motion Frame Skip

### Отправка сообщений на email (!Не работает!)
Options -> Email
Filters -> Use filter -> Monitor2ToMail
Работает так:
Определяются условия при которых срабатывает фильтр
Если фильтр находит соответствующий event, то отправляет на почту
(Чекбокс "Email details of all matches", справа вводятся данные)