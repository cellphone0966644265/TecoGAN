'''
several running examples, run with
python3 runGan.py 1 /path/to/your/LR_frames # the last number is the run case number

runcase == 1    inference a trained model using frames from <input_lr_directory>
runcase == 2    calculate the metrics, and save the numbers in csv
runcase == 3    training TecoGAN
runcase == 4    training FRVSR
runcase == ...  coming... data preparation and so on...
'''
import os, subprocess, sys, datetime, signal, shutil

if len(sys.argv) < 2:
    print("Usage: python3 runGan.py <runcase_number> [additional_arguments...]")
    print("For runcase 1 (inference): python3 runGan.py 1 <input_lr_directory>")
    sys.exit(1)

runcase = int(sys.argv[1])
print ("Executing run case %d" % runcase)

def preexec(): # Don't forward signals.
    os.setpgrp()

def mycall(cmd, block=False):
    if not block:
        return subprocess.Popen(cmd)
    else:
        # For Python 3, preexec_fn is generally safe with os.setpgrp
        return subprocess.Popen(cmd, preexec_fn = preexec)

def folder_check(path):
    try_num = 1
    oripath = path[:-1] if path.endswith('/') else path
    while os.path.exists(path):
        print("Delete existing folder " + path + "?(Y/N)")
        # Auto-reply 'Y' in non-interactive environments, or handle appropriately
        # For Colab, direct input() might be an issue if not run interactively in a specific way.
        # Consider making this non-interactive or configurable.
        # For now, let's assume it can be run or we skip this for simplicity in auto-run.
        # decision = input()
        print("Folder exists. Appending suffix or deleting (manual step for now).")
        # Simplified: if it exists, use it or fail. Or auto-rename.
        # shutil.rmtree(path, ignore_errors=True) # Example: auto-delete
        # break
        path = oripath + "_%d/"%try_num # Example: auto-rename
        try_num += 1
        print(f"Path changed to: {path}")
        if try_num > 5: # Safety break for auto-rename
            print("Too many rename attempts, exiting folder_check.")
            return None # Indicate failure or use original path and risk overwrite
    return path

if( runcase == 0 ): # download inference data, trained models
    # (Nội dung gốc của runcase == 0 giữ nguyên ở đây nếu bạn cần)
    print("Runcase 0: Download data - Not implemented in this update, keep original if needed.")

elif( runcase == 1 ): # inference a trained model
    if len(sys.argv) < 3:
        print("Usage for runcase 1: python3 runGan.py 1 <input_lr_directory>")
        print("Error: Missing <input_lr_directory> argument for runcase 1.")
        sys.exit(1)

    input_lr_dir = sys.argv[2] # Đường dẫn đến thư mục chứa frame LR, ví dụ: "./LR/segment_1"

    if not os.path.isdir(input_lr_dir):
        print(f"Error: Input LR directory '{input_lr_dir}' not found or is not a directory.")
        sys.exit(1)

    dirstr = './results/' # Nơi lưu kết quả upscale
    # Tên thư mục con trong 'results' sẽ dựa trên tên thư mục input LR
    input_folder_name = os.path.basename(os.path.normpath(input_lr_dir))

    if (not os.path.exists(dirstr)):
        try:
            os.makedirs(dirstr, exist_ok=True)
            print(f"Created results directory: {dirstr}")
        except OSError as e:
            print(f"Error creating results directory {dirstr}: {e}")
            sys.exit(1)

    print(f"Input LR directory: {input_lr_dir}")
    print(f"Output will be saved in a subfolder named '{input_folder_name}' inside '{dirstr}'")

    # Model checkpoint path - đảm bảo file này tồn tại
    tecogan_checkpoint = './model/TecoGAN'
    if not (os.path.exists(tecogan_checkpoint + '.data-00000-of-00001') or os.path.exists(tecogan_checkpoint + '.index')): # Check for typical TF checkpoint files
        print(f"LỖI: Không tìm thấy TecoGAN model checkpoint tại '{tecogan_checkpoint}'.")
        print("Hãy đảm bảo bạn đã tải model và đặt đúng đường dẫn.")
        # Bạn có thể thêm code tải model ở đây nếu muốn (tương tự runcase == 0)
        sys.exit(1)

    cmd1 = ["python3", "main.py", # Giả sử main.py cùng thư mục với runGan.py hoặc trong PATH
            "--cudaID", "0",            # Chỉ dùng 1 GPU
            "--output_dir",  dirstr,    # Thư mục gốc chứa kết quả
            "--summary_dir", os.path.join(dirstr, 'log/', input_folder_name), # Thư mục log, tạo subfolder
            "--mode","inference",
            "--input_dir_LR", input_lr_dir,   # Thư mục chứa frame LR (đã được truyền vào)
            "--output_pre", input_folder_name, # Tên thư mục con trong output_dir để lưu ảnh upscale
            "--num_resblock", "16",
            "--checkpoint", tecogan_checkpoint,
            "--output_ext", "png"
           ]
    print(f"Chuẩn bị thực thi main.py với lệnh: {' '.join(cmd1)}")
    try:
        process = mycall(cmd1) # Không block để có thể xem output nếu cần
        stdout, stderr = process.communicate() # Chờ tiến trình hoàn thành
        if process.returncode != 0:
            print(f"Lỗi khi chạy main.py. Return code: {process.returncode}")
            if stdout: print(f"Stdout:\n{stdout.decode(errors='ignore')}")
            if stderr: print(f"Stderr:\n{stderr.decode(errors='ignore')}")
        else:
            print(f"main.py đã chạy xong cho thư mục: {input_lr_dir}")
    except Exception as e:
        print(f"Ngoại lệ khi gọi main.py: {e}")


elif( runcase == 2 ): # calculate all metrics
    # (Nội dung gốc của runcase == 2 giữ nguyên ở đây nếu bạn cần)
    print("Runcase 2: Calculate metrics - Not implemented in this update, keep original if needed.")

elif( runcase == 3 ): # Train TecoGAN
    # (Nội dung gốc của runcase == 3 giữ nguyên ở đây nếu bạn cần)
    print("Runcase 3: Train TecoGAN - Not implemented in this update, keep original if needed.")

elif( runcase == 4 ): # Train FRVSR
    # (Nội dung gốc của runcase == 4 giữ nguyên ở đây nếu bạn cần)
    print("Runcase 4: Train FRVSR - Not implemented in this update, keep original if needed.")

else:
    print(f"Runcase {runcase} không hợp lệ hoặc chưa được triển khai trong bản cập nhật này.")

print(f"Hoàn thành runcase {runcase}.")
