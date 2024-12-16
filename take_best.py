import os
import re
import pandas as pd

def extract_initial_part(s):
    """
    提取字串中第一個大寫字母之前的部分，並去除末尾的逗號。
    如果字串中沒有大寫字母，則返回原字串。
    """
    match = re.search(r'[A-Z]', s)
    if match:
        return s[:match.start()].rstrip(',')
    return s

def get_best_acc_and_last_test_acc(worklog_path):
    """
    從 worklog.txt 中提取 best_acc 的值。
    如果找不到，返回 ('Unfinish', last_test_acc)。
    如果找不到 test_acc，也返回 ('Unfinish', 'N/A')。
    """
    if not os.path.isfile(worklog_path):
        return ('Unfinish', 'N/A')
    
    best_acc = 'Unfinish'
    last_test_acc = 'N/A'
    try:
        with open(worklog_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith('best_acc'):
                    parts = line.split()
                    if len(parts) >= 2:
                        best_acc = parts[1]
            # 重新打開文件以提取最後一個 test_acc
        with open(worklog_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith('test_acc:'):
                    parts = line.split(':')
                    if len(parts) >= 2:
                        last_test_acc = parts[1].strip()
        return (best_acc, last_test_acc if best_acc == 'Unfinish' else '')
    except Exception as e:
        print(f"讀取 {worklog_path} 時出現錯誤: {e}")
        return ('Unfinish', 'N/A')

def determine_group(name):
    """
    根據名稱中的最後兩個部分來確定組別。
    例如，'dot,our_kd,res32x4,res8x4' 的組別為 'res32x4,res8x4'
    """
    parts = name.split(',')
    if len(parts) >= 2:
        return ','.join(parts[-2:])
    elif len(parts) == 1:
        return parts[0]
    else:
        return 'Unknown'

def process_directories(base_dir, output_csv):
    """
    處理 base_dir 下的所有子目錄，提取所需信息並寫入 CSV 檔案。
    """
    if not os.path.isdir(base_dir):
        print(f"指定的目錄不存在: {base_dir}")
        return
    
    # 列出所有子目錄
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        print(f"在 {base_dir} 下未找到任何子目錄。")
        return
    
    data = []
    
    for subdir in subdirs:
        extracted_name = extract_initial_part(subdir)
        group = determine_group(extracted_name)
        worklog_path = os.path.join(base_dir, subdir, 'worklog.txt')
        best_acc, last_test_acc = get_best_acc_and_last_test_acc(worklog_path)
        data.append({
            'Directory': extracted_name,
            'Group': group,
            'Best_Acc': best_acc,
            'Last_Test_Acc': last_test_acc
        })
        print(f"處理完成: {extracted_name}, Group: {group}, Best_Acc: {best_acc}, Last_Test_Acc: {last_test_acc}")
    
    # 將資料轉換為 DataFrame
    df = pd.DataFrame(data)
    
    # 排序：先按 Group，然後按 Directory
    df.sort_values(by=['Group', 'Directory'], inplace=True)
    
    # 將 DataFrame 寫入 CSV
    try:
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')  # 使用 'utf-8-sig' 以便在 Excel 中正常顯示
        print(f"所有結果已寫入 {output_csv}")
    except Exception as e:
        print(f"寫入 CSV 檔案時出現錯誤: {e}")

if __name__ == "__main__":
    # 指定 base directory 和輸出 CSV 檔案名稱
    base_directory = '/root/Work/mdistiller/output/test/'
    output_csv_file = 'train_info/best_acc_results.csv'
    
    process_directories(base_directory, output_csv_file)
