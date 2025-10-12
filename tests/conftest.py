import os, sys
# Добавляем корень проекта в PYTHONPATH, чтобы импортировался пакет "src"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
