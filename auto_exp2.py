from datetime import date, datetime
import subprocess
import time
import os
# from apscheduler.schedulers.background import BackgroundScheduler


def main():
    d_model_list = [8, 24, 56, 128, 256, 512, 1024, 1048]
    for d_model in d_model_list:
        model_id = f'd_model_{d_model}'
        run(d_model, model_id)


def run(d_model, model_id):
    i = rf"python E:\Project\LTSF-Linear\run_longExp.py --label_len 32 --model_id {model_id} --model Transformer --d_model {d_model} --batch_size 48"
    print(f"running ....\t>>{i}")
    if subprocess.call(i) == 0:
        time.sleep(15)
    else:
        return f"Error is happened >>{i}\n"


if __name__ == "__main__":
    main()
