from datetime import date, datetime
import subprocess
import time
import os
# from apscheduler.schedulers.background import BackgroundScheduler


def main():
    criterion_list = ['mae', 'huber']
    for criterion in criterion_list:
        pic_key = f'criterion_{criterion}'
        model_id = pic_key
        run(pic_key, model_id, criterion)


def run(pic_key, model_id, criterion):
    i = rf"python E:\Project\LTSF-Linear\run_longExp.py --pic_save_key {pic_key} --model_id {model_id} --criterion {criterion}"
    print(f"running ....\t>>{i}")
    if subprocess.call(i) == 0:
        time.sleep(15)
    else:
        return f"Error is happened >>{i}\n"


if __name__ == "__main__":
    main()
