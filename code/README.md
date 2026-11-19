# Генератор синтетических показаний IMU по траектории камеры

Python 3.11, numpy, scipy, matplotlib. Запуск из папки `code/`:

    python checks.py        # проверка производных и непрерывности в узлах
    python convergence.py   # эксперимент со сходимостью -> ../out/convergence.json (~20 с)
    python figures.py       # рисунки -> ../out/*.png

Модели: A — отрезки + SLERP, B — кубический сплайн + SQUAD, V — B-сплайны в R^3 и на SO(3).
`snippet.py` — короткая версия ядра генератора (та, что приведена в статье на Хабре).
