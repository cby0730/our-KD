import subprocess
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

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
            base_command = ["pipenv", "run", "python3", "tools/train.py", "--cfg", config_path]
            for param_combo in param_combinations:
                # 將參數組合轉換為配置參數
                params = []
                for param in param_combo:
                    config_param = f"{param_mapping[param][0]} {param_mapping[param][1]}"
                    params.extend([param_mapping[param][0], param_mapping[param][1]])
                # 將配置參數作為命令的一部分
                cmd = base_command + params
                commands.append(cmd)
    return commands

def run_command(cmd):
    """執行單一命令並返回結果"""
    cmd_str = ' '.join(cmd)
    print(f"開始執行: {cmd_str}")
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"完成: {cmd_str}")
        return (cmd_str, True, result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"失敗: {cmd_str}")
        return (cmd_str, False, e.stderr)

def main():
    # 定義轉換方式
    transformations = ['our_kd']
    
    # 定義每個轉換方式下的配置檔案（不包含路徑與副檔名）
    configs = {
        'our_kd': [
            'res32x4_res8x4',
            'res32x4_shuv2',
            'res50_mv2',
            'vgg13_mv2',
            'vgg13_vgg8',
            'wrn40_2_wrn_16_2',
            'wrn40_2_wrn_40_1'
        ],
        'kd': [
            'res32x4_res8x4',
            'res32x4_shuv2',
            'res50_mv2',
            'vgg13_mv2',
            'vgg13_vgg8',
            'wrn40_2_wrn_16_2',
            'wrn40_2_wrn_40_1'
        ],
        'dkd': [
            'res32x4_res8x4',
            'res32x4_shuv2',
            'res50_mv2',
            'vgg13_mv2',
            'vgg13_vgg8',
            'wrn40_2_wrn_16_2',
            'wrn40_2_wrn_40_1'
        ]
    }
    
    # 定義參數映射：鍵為參數名稱，值為 (配置參數名稱, 配置參數值)
    param_mapping = {
        'ER': ('OURKD.ER', 'True'),
        'DOT': ('SOLVER.TRAINER', 'dot'),
        'STD': ('OURKD.STD', 'True'),
        'STD2': ('OURKD.STD2', 'True'),
        # 'MT': ('OURKD.MT', 'True')
    }
    
    # 生成所有命令
    commands = build_commands(transformations, configs, param_mapping)
    
    # 設定同時執行的最大實驗數量
    max_workers = 8
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有命令到執行池
        future_to_cmd = {executor.submit(run_command, cmd): cmd for cmd in commands}
        
        for future in as_completed(future_to_cmd):
            cmd_str, success, output = future.result()
            if success:
                print(f"成功: {cmd_str}\n輸出:\n{output}\n")
            else:
                print(f"錯誤: {cmd_str}\n錯誤訊息:\n{output}\n")

if __name__ == "__main__":
    main()
