# Data dictionary — файлы данных

Набор на основе UNSW-NB15 (Moustafa et al., 2015) для учебной задачи. Данные разделены на два файла:

- `data/UNSW_NB15_training-set.csv` — обучающая выборка (`split = train`, 175341 строка)
- `data/UNSW_NB15_testing-set.csv` — тестовая выборка (`split = test`, 82332 строки)

| Колонка | Тип | Описание |
|---|---|---|
| `id` | int | Идентификатор записи. **Не признак.** |
| `dur` | float | Длительность соединения |
| `proto` | str | Протокол (`tcp`, `udp`, `unas`, …) |
| `service` | str | Сервис (`http`, `dns`, `ftp`, `-`, …) |
| `state` | str | Состояние соединения (`FIN`, `CON`, `INT`, …) |
| `spkts` / `dpkts` | int | Пакеты от источника / к источнику |
| `sbytes` / `dbytes` | int | Байты от источника / к источнику |
| `rate` | float | Скорость |
| `sttl` / `dttl` | int | TTL источника / назначения |
| `sload` / `dload` | float | Нагрузка источника / назначения |
| `sloss` / `dloss` | int | Потери |
| `sinpkt` / `dinpkt` | float | Интервалы между пакетами |
| `sjit` / `djit` | float | Джиттер |
| `swin` / `dwin` | int | TCP window |
| `stcpb` / `dtcpb` | int | TCP sequence |
| `tcprtt` | float | TCP RTT |
| `synack` / `ackdat` | float | Задержки handshake |
| `smean` / `dmean` | int | Средний размер пакета |
| `trans_depth` | int | Глубина транзакции |
| `response_body_len` | int | Длина тела ответа |
| `ct_*` | int | Контекстные счётчики соединений |
| `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`, `is_sm_ips_ports` | int | Служебные признаки |
| `attack_cat` | str | Тип атаки (`Normal`, `DoS`, `Exploits`, …) |
| `label` | int | **Истина:** `1` — атака, `0` — норма. Используется только в `evaluate`. |
| `split` | str | `train` или `test` (соответствует файлу) |
| `payload_preview` | str | Короткий фрагмент payload. |

## Особенности

- Часть числовых колонок содержит пропуски.
- Некоторые колонки могут иметь нулевую дисперсию на train — политика обработки задаётся конфигом.
- `label` и `attack_cat` не должны попадать в признаки ни при обучении, ни при выборе порога.
