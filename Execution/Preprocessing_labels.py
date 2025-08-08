import os
import glob
import re
from converted_backlash import converted_backlash
import shutil


def modify_first_number_in_txt_files(folder_path, class_num):
    if not os.path.isdir(folder_path):
        print(f"Folder path does not exist: {folder_path}")
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


    print(f"The first number of all txt files has been changed to {class_num}.")


def check_labels_error(folders):
    error_count = 0
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"[Warning] Folder does not exist: {folder}")
            continue
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) < 7 or (len(parts) - 5) % 2 != 0:
                            print(f"{filename} Line {i}: Annotation format error ({len(parts)} columns)")
                            error_count += 1
    print(f"Total {error_count} format errors found")


def fix_yolo_seg_line(line):
    parts = line.strip().split()
    if len(parts) < 7:
        return None  # Too short to fix
    class_id, bbox = parts[:1], parts[1:5]
    mask_pts = parts[5:]
    if (len(mask_pts) % 2) != 0:
        mask_pts = mask_pts[:-1]  # Remove last point
    return ' '.join(class_id + bbox + mask_pts)


def fix_folder_segmentation_format(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Folder does not exist: {folder_path}")
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
    print(f"All annotation formats in folder {folder_path} have been fixed")


def clean_yolo_labels(folder_path, backup=True):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return None


    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    if not txt_files:
        print(f"No txt files found in '{folder_path}'")
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
                print(f"✓ Cleaned: {os.path.basename(txt_file)} (removed {removed_count} non-numeric characters)")
            else:
                print(f"- No cleaning needed: {os.path.basename(txt_file)}")


            results['processed_files'] += 1


        except Exception as e:
            print(f"✗ Error processing file {os.path.basename(txt_file)}: {str(e)}")
            results['error_files'] += 1


    print(f"\nProcessing completed!")
    print(f"Total files: {results['total_files']}")
    print(f"Successfully processed: {results['processed_files']}")
    print(f"Files with errors: {results['error_files']}")
    print(f"Removed non-numeric characters count: {results['removed_chars']}")


    return results


def clean_single_file(file_path, backup=True):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist")
        return False


    if not file_path.endswith('.txt'):
        print(f"Error: '{file_path}' is not a txt file")
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
        print(f"✓ Cleaned file: {os.path.basename(file_path)} (removed {removed_count} non-numeric characters)")
        return True


    except Exception as e:
        print(f"✗ Error occurred while processing file: {str(e)}")
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
            print(f"✓ {result['file']}: Completely clean")
        else:
            print(f"✗ {result['file']}: Issues found")
            if non_numeric_chars:
                print(f"  - Non-numeric characters: {set(non_numeric_chars)}")
            if result['ends_with_newline']:
                print(f"  - File ends with newline character")


        return result


    except Exception as e:
        print(f"✗ Error verifying file {file_path}: {str(e)}")
        return None



def main_process_labels(folder_path, class_num=None, do_fix_seg=False, do_clean=True, do_check=True, backup=True):
    """
    Integrated label processing pipeline function


    Parameters:
        folder_path (str): Path to the labels folder
        class_num (int or None): If given, modifies the first class index in all labels to class_num
        do_fix_seg (bool): Whether to perform fix_yolo_seg_line correction
        do_clean (bool): Whether to perform clean_yolo_labels cleaning
        do_check (bool): Whether to perform error checking
        backup (bool): Whether to backup original files
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
