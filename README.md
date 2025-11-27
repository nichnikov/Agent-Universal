# Universal Autonomous Agent (MVP)

Мультиагентная система на базе LangGraph с Supervisor и Legal Expert узлами.

## Архитектура

- **Supervisor Node**: Управляющий узел для маршрутизации запросов
- **Legal Expert Node**: Специализированный агент для юридических консультаций
- **LangGraph**: Оркестрация потока между агентами
- **Langfuse**: Управление промтами и трассировка

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd Agent-Universal
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Настройте переменные окружения:
```bash
cp env.example .env
# Отредактируйте .env файл, добавив ваши API ключи
```

## Конфигурация

### API Ключи

Необходим хотя бы один из следующих ключей:
- `OPENAI_API_KEY` - для использования GPT-4o
- `ANTHROPIC_API_KEY` - для использования Claude-3.5-Sonnet

### Langfuse (опционально)

Для управления промтами и трассировки:
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_HOST` (по умолчанию: https://cloud.langfuse.com)

Если Langfuse не настроен, система будет использовать встроенные fallback промты.

## Запуск

```bash
python main.py
```

## Тестовые сценарии

Система автоматически выполнит следующие тесты:

1. **"Как правильно оформить продажу офисной мебели юрлицом?"** 
   - Ожидаемый поток: Supervisor → Legal Expert → Tool Call → Supervisor → FINISH

2. **"Привет, кто ты?"** 
   - Ожидаемый поток: Supervisor → FINISH

3. **"Какие налоги нужно платить при продаже имущества?"** 
   - Ожидаемый поток: Supervisor → Legal Expert → Tool Call → Supervisor → FINISH

## Структура проекта

```
/project_root
├── env.example          # Шаблон переменных окружения
├── main.py              # Точка входа для тестирования
├── requirements.txt     # Зависимости Python
├── README.md           # Документация
└── src/
    ├── __init__.py
    ├── state.py         # AgentState - глобальное состояние
    ├── graph.py         # Сборка LangGraph
    ├── utils.py         # Утилиты для Langfuse
    ├── nodes/
    │   ├── __init__.py
    │   ├── supervisor.py    # Supervisor узел
    │   └── legal_expert.py  # Legal Expert узел
    └── tools/
        ├── __init__.py
        └── legal_tools.py   # Инструменты для поиска в законодательстве
```

## Возможности Legal Expert

- Поиск в российском законодательстве (ГК РФ, УК РФ, КоАП РФ)
- Консультации по налогообложению
- Помощь с договорами и сделками
- Разъяснение правовых процедур

## Масштабирование

Система готова к добавлению новых агентов:
1. Создайте новый узел в `src/nodes/`
2. Добавьте специализированные инструменты в `src/tools/`
3. Обновите Supervisor для маршрутизации к новому агенту
4. Добавьте узел и ребра в `src/graph.py`

## Трассировка

При настроенном Langfuse все взаимодействия будут отслеживаться в дашборде:
- Вызовы Supervisor
- Решения о маршрутизации
- Работа Legal Expert
- Вызовы инструментов
- Финальные ответы
