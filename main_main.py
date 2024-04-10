import subprocess
import time

def run_python_script(script_path):
    # Запускаем скрипт и дожидаемся его завершения
    process = subprocess.run(['python', script_path], check=True)
    print(f"Скрипт {script_path} завершился с кодом возврата {process.returncode}")

# Пример использования

run_python_script("C:/Users/rosti/Desktop/Весь код новый/main1.py")
