# Генератор синтетических показаний IMU по траектории камеры

Python 3.11, numpy, scipy, matplotlib. Запуск из папки `code/`:

    python checks.py        # проверка производных и непрерывности в узлах
    python convergence.py   # эксперимент со сходимостью -> ../out/convergence.json (~20 с)
    python figures.py       # рисунки для Хабра -> ../out/*.png
    python convergence.py --refined   # то же для модели В после уточнения
                                      # ориентации (задача 1) -> ../out/convergence_refined.json
    python figures_article.py         # рисунки статьи -> ../figures/*.pdf
    python allan.py         # проверка шума по вариации Аллана -> ../out/allan.json (~15 с)

Модели: A — отрезки + SLERP, B — кубический сплайн + SQUAD, V — B-сплайны в R^3 и на SO(3).
`snippet.py` — короткая версия ядра генератора (та, что приведена в статье на Хабре).
