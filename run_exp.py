import subprocess
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import datetime
import os
import multiprocessing

# 設定 log 檔案名稱
log_file = 'train_info/execution_log.txt'
# 確保日誌目錄存在
os.makedirs(os.path.dirname(log_file), exist_ok=True)
# 建立全域的鎖
log_lock = threading.Lock()

# 新增 GPU 相關設定
GPUS = [0, 1]  # 使用的 GPU 編號
MAX_EXPERIMENTS_PER_GPU = 4  # 每個 GPU 最多執行的實驗數

class GPUManager:
    def __init__(self, gpu_ids, max_experiments_per_gpu):
        self.gpu_ids = gpu_ids
        self.max_experiments_per_gpu = max_experiments_per_gpu
        self.gpu_counts = {gpu_id: 0 for gpu_id in gpu_ids}
        self.lock = threading.Lock()

    def acquire_gpu(self):
        with self.lock:
            available_gpus = [gpu_id for gpu_id, count in self.gpu_counts.items() 
                            if count < self.max_experiments_per_gpu]
            if not available_gpus:
                return None
            
            selected_gpu = min(available_gpus, 
                             key=lambda gpu_id: self.gpu_counts[gpu_id])
            self.gpu_counts[selected_gpu] += 1
            print(f"分配 GPU {selected_gpu}，當前使用量: {self.gpu_counts[selected_gpu]}/{self.max_experiments_per_gpu}")
            return selected_gpu

    def release_gpu(self, gpu_id):
        with self.lock:
            if gpu_id in self.gpu_counts:
                self.gpu_counts[gpu_id] = max(0, self.gpu_counts[gpu_id] - 1)
                print(f"釋放 GPU {gpu_id}，當前使用量: {self.gpu_counts[gpu_id]}/{self.max_experiments_per_gpu}")

def generate_param_combinations(param_mapping):
    """生成所有可能的參數組合"""
    params = list(param_mapping.keys())
    combinations = []
    for r in range(len(params)+1):
        for combo in itertools.combinations(params, r):
            combinations.append(combo)
    return combinations

def build_commands(transformations, configs, param_mapping):
    """根據轉換方式、配置檔案和參數組合生成所有命令"""
    param_combinations = generate_param_combinations(param_mapping)
    commands = []
    for transformation in transformations:
        for config in configs[transformation]:
            config_path = f"configs/cifar100/{transformation}/{config}.yaml"
            base_command = ["pipenv", "run", "python3", "tools/train.py", "--cfg", config_path, 
                            'EXPERIMENT.PROJECT', 'ablation', 
                            #'OURKD.LS', 'True', 
                            #'OURKD.MT', 'True', 
                            #'OURKD.STD', 'True', 
                            #'OURKD.MTLS', 'True', 
                            #'OURKD.MSE', 'True',
                            #'OURKD.ALPHA', '1.0',
                            #'OURKD.BETA', '8.0',
                            #'OURKD.LS_WEIGHT', '1.0',
                            #'OURKD.MSE_WEIGHT', '1.0',
                            #'OURKD.MAE_WEIGHT', '1.0',
                            #'OURKD.RV_WEIGHT', '1.0',
                            #'OURKD.CT_WEIGHT', '1.0',
                            'SOLVER.TRAINER', 'aug_dot']
            for param_combo in param_combinations:
                params = []
                for param in param_combo:
                    params.extend([param_mapping[param][0], param_mapping[param][1]])
                cmd = base_command + params
                commands.append(cmd)
    return commands

def read_successful_commands(log_file):
    """從 log 檔案中讀取已成功執行的命令"""
    successful_commands = set()
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' | ')
                if len(parts) >= 4:
                    command = parts[1]
                    status = parts[2]
                    if status == 'Success':
                        successful_commands.add(command)
    return successful_commands

def run_command(cmd, gpu_manager):
    """執行單一命令並返回結果"""
    cmd_str = ' '.join(cmd)
    
    # 檢查命令是否已成功執行過
    with log_lock:
        successful_commands = read_successful_commands(log_file)
    if cmd_str in successful_commands:
        print(f"命令已成功執行過，跳過: {cmd_str}")
        return (cmd_str, True, "Skipped")

    # 獲取 GPU
    gpu_id = gpu_manager.acquire_gpu()
    if gpu_id is None:
        print(f"無可用的 GPU，跳過命令: {cmd_str}")
        return (cmd_str, False, "No GPU available")

    # 設定環境變數指定 GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"開始在 GPU {gpu_id} 上執行: {cmd_str}")
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, text=True, env=env)
        end_time = time.time()
        duration = end_time - start_time
        print(f"在 GPU {gpu_id} 上完成: {cmd_str}")
        
        # 記錄成功執行的命令到 log 檔案
        with log_lock:
            with open(log_file, 'a') as f:
                log_entry = f"{datetime.datetime.now()} | {cmd_str} | Success | {duration:.2f}s | GPU {gpu_id}\n"
                f.write(log_entry)
        return (cmd_str, True, result.stdout)
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"在 GPU {gpu_id} 上失敗: {cmd_str}")
        
        # 記錄失敗的命令到 log 檔案
        with log_lock:
            with open(log_file, 'a') as f:
                log_entry = f"{datetime.datetime.now()} | {cmd_str} | Failure | {duration:.2f}s | GPU {gpu_id}\n"
                f.write(log_entry)
        return (cmd_str, False, e.stderr)
    finally:
        # 釋放 GPU 資源
        gpu_manager.release_gpu(gpu_id)

def main():
    # 定義轉換方式
    transformations = ['our_kd']

    # 定義每個轉換方式下的配置檔案
    configs = {
        'our_kd': [
            'res32x4_res8x4',
            #'res32x4_shuv1',
            #'res32x4_shuv2',
            'res50_mv2',
            #'res56_res20',
            #'res110_res32',
            #'vgg13_mv2',
            'vgg13_vgg8',
            #'wrn40_2_shuv1',
            #'wrn40_2_wrn_16_2',
            #'wrn40_2_wrn_40_1'
        ],
        'kd': [
            'res32x4_res8x4',
            'res32x4_shuv1',
            'res32x4_shuv2',
            'res50_mv2',
            'res56_res20',
            'res110_res32',
            'vgg13_mv2',
            'vgg13_vgg8',
            'wrn40_2_shuv1',
            'wrn40_2_wrn_16_2',
            'wrn40_2_wrn_40_1'
        ],
        'dkd': [
            'res32x4_res8x4',
            'res32x4_shuv1',
            'res32x4_shuv2',
            'res50_mv2',
            'res56_res20',
            'res110_res32',
            'vgg13_mv2',
            'vgg13_vgg8',
            'wrn40_2_shuv1',
            'wrn40_2_wrn_16_2',
            'wrn40_2_wrn_40_1'
        ],
        'dtkd': [
            'res32x4_res8x4',
            'res32x4_shuv1',
            'res32x4_shuv2',
            'res50_mv2',
            'res56_res20',
            'res110_res32',
            'vgg13_mv2',
            'vgg13_vgg8',
            'wrn40_2_shuv1',
            'wrn40_2_wrn_16_2',
            'wrn40_2_wrn_40_1'
        ]
    }

    # 定義參數映射
    param_mapping = {
        #'aug_dot': ('SOLVER.TRAINER', 'aug_dot'),
        'LS': ('OURKD.LS', 'True'),
        #'ER': ('OURKD.ER', 'True'),
        'MTLS': ('OURKD.MTLS', 'True'), 
        'MT': ('OURKD.MT', 'True'), 
        'STD': ('OURKD.STD', 'True'),
        'MSE': ('OURKD.MSE', 'True'),
        'MAE': ('OURKD.MAE', 'True'),
        'RV': ('OURKD.RV', 'True'),
        'CT': ('OURKD.CT', 'True'),
        'CR': ('OURKD.CR', 'True'),
        #'DT': ('OURKD.DT', 'True'),
    }

    # 生成所有命令
    commands = build_commands(transformations, configs, param_mapping)
    print(f"共有 {len(commands)} 個參數組合待執行")
    
    # 列出所有生成的命令
    for cmd in commands:
        print('Generated command:', ' '.join(cmd))

    # 創建 GPU 管理器
    gpu_manager = GPUManager(GPUS, MAX_EXPERIMENTS_PER_GPU)

    # 設定執行池，總執行數為所有 GPU 可執行的實驗總和
    max_workers = len(GPUS) * MAX_EXPERIMENTS_PER_GPU

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有命令到執行池
        future_to_cmd = {executor.submit(run_command, cmd, gpu_manager): cmd 
                        for cmd in commands}

        for future in as_completed(future_to_cmd):
            cmd_str, success, output = future.result()
            if success and output != "Skipped":
                print(f"成功: {cmd_str}\n輸出:\n{output}\n")
            elif success and output == "Skipped":
                print(f"跳過: {cmd_str}")
            else:
                print(f"錯誤: {cmd_str}\n錯誤訊息:\n{output}\n")

if __name__ == "__main__":
    main()