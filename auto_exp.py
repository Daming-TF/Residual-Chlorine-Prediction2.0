from datetime import date, datetime
import subprocess
import time
import os
# from apscheduler.schedulers.background import BackgroundScheduler


def main():
    model = ['DLinear', 'OurCNN', 'Informer']   # [5, 10, 20, 50, 100, 200, 300]
    data_path = {
        'RC': 'RC.xlsx',
        'ETTm1': 'ETTm1.csv'
    }
    seq_len = {
        'DLinear': 336,
        'Informer': 96,
        'OurCNN': [336, 96]
    }
    pred_len = [48, 96, 192, 336, 720]

    for m in model:
        for p_len in pred_len:
            if m == 'OurCNN':
                s_list = seq_len[m]
                data = 'RC'
                path = data_path[data]
                for s_len in s_list:
                    run(m, p_len, s_len, data, path)

            elif m == 'Informer':
                s_len = seq_len[m]
                data = 'ETTm1'
                path = data_path[data]
                run(m, p_len, s_len, data, path)

            elif m == 'DLinear':
                s_len = seq_len[m]
                data_list = ['RC', 'ETTm1']
                for data in data_list:
                    path = data_path[data]
                    run(m, p_len, s_len, data, path)

            # os.system(f'e:')
            # os.system(
            #     f"python E:\Project\ResidualChlorinePrediction\scripts\main.py --window_size {ws} --label_length {ll}")
            # print(f"sleep! - {datetime.now()}")
            # time.sleep(650)


def run(m, p_len, s_len, data, path):
    i = rf"python E:\Project\LTSF-Linear\run_longExp.py --model {m} --data {data} --data_path {path} --seq_len {s_len} --pred_len {p_len}"
    print(f"running ....\t>>{i}")
    if subprocess.call(i) == 0:
        time.sleep(15)
    else:
        return f"Error is happened >>{i}\n"


if __name__ == "__main__":
    main()
