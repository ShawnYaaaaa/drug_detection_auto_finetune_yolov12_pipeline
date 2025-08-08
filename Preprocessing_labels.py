import os
import glob
import re
from converted_backlash import converted_backlash
import shutil

def modify_first_number_in_txt_files(folder_path, class_num):
    if not os.path.isdir(folder_path):
        print(f"資料夾路徑不存在: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split(' ')
                if len(parts) > 0 and parts[0].isdigit():
                    parts[0] = str(class_num)
                    new_line = ' '.join(parts)
                    new_lines.append(new_line + '\n')
                else:
                    new_lines.append(line)

            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(new_lines)

    print(f"所有txt檔案的第一個數字已被修改為{class_num}。")

def check_labels_error(folders):
    error_count = 0
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"[警告] 資料夾不存在：{folder}")
            continue
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) < 7 or (len(parts) - 5) % 2 != 0:
                            print(f"{filename} Line {i}: 標註格式錯誤 ({len(parts)} 欄)")
                            error_count += 1
    print(f"共{error_count}筆格式錯誤")

def fix_yolo_seg_line(line):
    parts = line.strip().split()
    if len(parts) < 7:
        return None  # 太短不可修
    class_id, bbox = parts[:1], parts[1:5]
    mask_pts = parts[5:]
    if (len(mask_pts) % 2) != 0:
        mask_pts = mask_pts[:-1]  # 移除最後一點
    return ' '.join(class_id + bbox + mask_pts)

def fix_folder_segmentation_format(folder_path):
    if not os.path.isdir(folder_path):
        print(f"資料夾不存在: {folder_path}")
        return
    for fname in os.listdir(folder_path):
        if fname.endswith('.txt'):
            filepath = os.path.join(folder_path, fname)
            with open(filepath, encoding='utf-8') as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                fixed = fix_yolo_seg_line(line)
                if fixed:
                    new_lines.append(fixed + '\n')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
    print(f"資料夾 {folder_path} 中所有標註格式已修正")

def clean_yolo_labels(folder_path, backup=True):
    if not os.path.exists(folder_path):
        print(f"錯誤: 資料夾 '{folder_path}' 不存在")
        return None

    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    if not txt_files:
        print(f"在 '{folder_path}' 中沒有找到txt文件")
        return None

    results = {
        'total_files': len(txt_files),
        'processed_files': 0,
        'error_files': 0,
        'removed_chars': 0
    }

    for txt_file in txt_files:
        try:
            with open(txt_file, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')

            if backup:
                backup_file = txt_file + '.backup'
                with open(backup_file, 'wb') as f:
                    f.write(content.encode('utf-8'))

            cleaned_content = re.sub(r'[^0-9.\s]', '', content)
            cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
            final_content = cleaned_content.strip().rstrip()

            removed_count = len(content) - len(final_content)

            with open(txt_file, 'wb') as f:
                f.write(final_content.encode('utf-8'))

            if removed_count > 0:
                results['removed_chars'] += removed_count
                print(f"✓ 已清理: {os.path.basename(txt_file)} (移除了 {removed_count} 個非數字字符)")
            else:
                print(f"- 無需清理: {os.path.basename(txt_file)}")

            results['processed_files'] += 1

        except Exception as e:
            print(f"✗ 處理文件 {os.path.basename(txt_file)} 時發生錯誤: {str(e)}")
            results['error_files'] += 1

    print(f"\n處理完成!")
    print(f"總文件數: {results['total_files']}")
    print(f"成功處理: {results['processed_files']}")
    print(f"錯誤文件: {results['error_files']}")
    print(f"清理的非數字字符數: {results['removed_chars']}")

    return results

def clean_single_file(file_path, backup=True):
    if not os.path.exists(file_path):
        print(f"錯誤: 文件 '{file_path}' 不存在")
        return False

    if not file_path.endswith('.txt'):
        print(f"錯誤: '{file_path}' 不是txt文件")
        return False

    try:
        with open(file_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')

        if backup:
            backup_file = file_path + '.backup'
            with open(backup_file, 'wb') as f:
                f.write(content.encode('utf-8'))

        cleaned_content = re.sub(r'[^0-9.\s]', '', content)
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        final_content = cleaned_content.strip().rstrip()

        with open(file_path, 'wb') as f:
            f.write(final_content.encode('utf-8'))

        removed_count = len(content) - len(final_content)
        print(f"✓ 已清理文件: {os.path.basename(file_path)} (移除了 {removed_count} 個非數字字符)")
        return True

    except Exception as e:
        print(f"✗ 處理文件時發生錯誤: {str(e)}")
        return False

def verify_clean_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            content = f.read()

        text_content = content.decode('utf-8', errors='ignore')
        non_numeric_chars = re.findall(r'[^0-9.\s]', text_content)

        result = {
            'file': os.path.basename(file_path),
            'is_clean': len(non_numeric_chars) == 0,
            'file_size': len(content),
            'non_numeric_chars': non_numeric_chars,
            'ends_with_newline': content.endswith(b'\n') or content.endswith(b'\r\n')
        }

        if result['is_clean'] and not result['ends_with_newline']:
            print(f"✓ {result['file']}: 完全乾淨")
        else:
            print(f"✗ {result['file']}: 發現問題")
            if non_numeric_chars:
                print(f"  - 非數字字符: {set(non_numeric_chars)}")
            if result['ends_with_newline']:
                print(f"  - 文件末尾有換行符")

        return result

    except Exception as e:
        print(f"✗ 驗證文件 {file_path} 時發生錯誤: {str(e)}")
        return None


def main_process_labels(folder_path, class_num=None, do_fix_seg=False, do_clean=True, do_check=True, backup=True):
    """
    整合處理 labels 的流程函數

    Parameters:
        folder_path (str): labels 資料夾路徑
        class_num (int or None): 若有，修改所有標籤第一個class編號為 class_num
        do_fix_seg (bool): 是否執行 fix_yolo_seg_line 修正
        do_clean (bool): 是否執行 clean_yolo_labels 清理
        do_check (bool): 是否執行錯誤檢查
        backup (bool): 是否備份原始檔案
    """
    folder_path = converted_backlash(folder_path)
    if class_num is not None:
        modify_first_number_in_txt_files(folder_path, class_num)

    if do_fix_seg:
        fix_folder_segmentation_format(folder_path)

    if do_clean:
        clean_yolo_labels(folder_path, backup=backup)

    if do_check:
        check_labels_error([folder_path])
    
