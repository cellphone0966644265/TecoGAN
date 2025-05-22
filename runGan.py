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
        return subprocess.Popen(cmd, preexec_fn = preexec)

def folder_check(path): # Hàm này có thể cần điều chỉnh cho môi trường không tương tác
    try_num = 1
    oripath = path[:-1] if path.endswith('/') else path
    new_path = path # Khởi tạo new_path
    while os.path.exists(new_path):
        print(f"Warning: Folder {new_path} exists.")
        # Trong môi trường tự động, không nên hỏi input.
        # Lựa chọn: ghi đè, tạo tên mới, hoặc báo lỗi.
        # Ở đây, chúng ta sẽ thử tạo tên mới.
        new_path = oripath + "_%d/"%try_num
        try_num += 1
        print(f"Attempting new path: {new_path}")
        if try_num > 10: # Giới hạn số lần thử để tránh vòng lặp vô hạn
            print(f"Error: Too many existing folders like {oripath}_N/. Please clean up.")
            return None # Trả về None nếu không thể tạo thư mục mới
    try:
        os.makedirs(new_path, exist_ok=True) # Tạo thư mục nếu nó (hoặc phiên bản đã đổi tên) chưa tồn tại
    except OSError as e:
        print(f"Error creating directory {new_path}: {e}")
        return None
    return new_path


if( runcase == 0 ): # download inference data, trained models
    # download the trained model
    if(not os.path.exists("./model/")): os.mkdir("./model/")
    cmd1 = "wget https://ge.in.tum.de/download/data/TecoGAN/model.zip -O model/model.zip;"
    cmd1 += "unzip -o model/model.zip -d model; rm model/model.zip" # -o for overwrite
    print("Downloading and unzipping TecoGAN model...")
    subprocess.call(cmd1, shell=True)
    
    # download some test data (optional, có thể bỏ nếu không dùng cho runcase 1 tùy chỉnh)
    if(not os.path.exists("./LR/")): os.mkdir("./LR/")
    cmd2 = "wget https://ge.in.tum.de/download/data/TecoGAN/vid3_LR.zip -O LR/vid3.zip;"
    cmd2 += "unzip -o LR/vid3.zip -d LR; rm LR/vid3.zip"
    print("Downloading and unzipping sample LR data (vid3)...")
    subprocess.call(cmd2, shell=True)
    
    cmd2 = "wget https://ge.in.tum.de/download/data/TecoGAN/tos_LR.zip -O LR/tos.zip;"
    cmd2 += "unzip -o LR/tos.zip -d LR; rm LR/tos.zip"
    print("Downloading and unzipping sample LR data (tos)...")
    subprocess.call(cmd2, shell=True)
    
    # download the ground-truth data (optional)
    if(not os.path.exists("./HR/")): os.mkdir("./HR/")
    cmd3 = "wget https://ge.in.tum.de/download/data/TecoGAN/vid4_HR.zip -O HR/vid4.zip;"
    cmd3 += "unzip -o HR/vid4.zip -d HR; rm HR/vid4.zip"
    print("Downloading and unzipping sample HR data (vid4)...")
    subprocess.call(cmd3, shell=True)
    
    cmd3 = "wget https://ge.in.tum.de/download/data/TecoGAN/tos_HR.zip -O HR/tos.zip;"
    cmd3 += "unzip -o HR/tos.zip -d HR; rm HR/tos.zip"
    print("Downloading and unzipping sample HR data (tos)...")
    subprocess.call(cmd3, shell=True)
    print("Runcase 0: Download complete.")

elif( runcase == 1 ): # inference a trained model
    if len(sys.argv) < 3:
        print("Usage for runcase 1: python3 runGan.py 1 <input_lr_directory>")
        print("Error: Missing <input_lr_directory> argument for runcase 1.")
        sys.exit(1)

    input_lr_dir = sys.argv[2]

    if not os.path.isdir(input_lr_dir):
        print(f"Error: Input LR directory '{input_lr_dir}' not found or is not a directory.")
        sys.exit(1)

    dirstr = './results/' # Nơi lưu kết quả upscale (ví dụ: /content/TecoGAN/results/)
    input_folder_name = os.path.basename(os.path.normpath(input_lr_dir)) # Tên thư mục con trong 'results'

    # Đảm bảo thư mục results gốc tồn tại
    if not os.path.exists(dirstr):
        try:
            os.makedirs(dirstr, exist_ok=True)
            print(f"Created results directory: {dirstr}")
        except OSError as e:
            print(f"Error creating results directory {dirstr}: {e}")
            sys.exit(1)
            
    # Tạo thư mục summary_dir cụ thể cho lần chạy này
    summary_output_dir = os.path.join(dirstr, 'log/', input_folder_name)
    try:
        os.makedirs(summary_output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating summary directory {summary_output_dir}: {e}")
        # Có thể không cần thoát nếu chỉ là lỗi tạo thư mục log

    print(f"Input LR directory: {input_lr_dir}")
    print(f"Output will be saved in a subfolder named '{input_folder_name}' inside '{dirstr}'")

    tecogan_checkpoint = './model/TecoGAN' # Đường dẫn tương đối tới model đã tải bởi runcase 0
    if not (os.path.exists(tecogan_checkpoint + '.data-00000-of-00001') or os.path.exists(tecogan_checkpoint + '.index') or os.path.exists(tecogan_checkpoint)):
        print(f"LỖI: Không tìm thấy TecoGAN model checkpoint tại '{tecogan_checkpoint}'.")
        print("Chạy 'python3 runGan.py 0' để tải model nếu chưa có.")
        sys.exit(1)

    # main.py nằm cùng thư mục với runGan.py
    main_script_path = os.path.join(os.path.dirname(__file__), "main.py")

    cmd1 = ["python3", main_script_path,
            "--cudaID", "0",
            "--output_dir",  dirstr,
            "--summary_dir", summary_output_dir,
            "--mode","inference",
            "--input_dir_LR", input_lr_dir,
            "--output_pre", input_folder_name,
            "--num_resblock", "16",
            "--checkpoint", tecogan_checkpoint,
            "--output_ext", "png"
           ]
    print(f"Chuẩn bị thực thi main.py với lệnh: {' '.join(cmd1)}")
    try:
        # Chạy main.py
        process = subprocess.Popen(cmd1)
        stdout, stderr = process.communicate() # Chờ tiến trình hoàn thành
        if process.returncode != 0:
            print(f"Lỗi khi chạy main.py. Return code: {process.returncode}")
            if stdout: print(f"Stdout:\n{stdout.decode(errors='ignore') if isinstance(stdout, bytes) else stdout}")
            if stderr: print(f"Stderr:\n{stderr.decode(errors='ignore') if isinstance(stderr, bytes) else stderr}")
        else:
            print(f"main.py đã chạy xong cho thư mục: {input_lr_dir}")
    except Exception as e:
        print(f"Ngoại lệ khi gọi main.py: {e}")


elif( runcase == 2 ):
    print("Runcase 2: Calculate metrics - Not fully implemented in this update, keep original if needed.")
    # (Giữ lại nội dung gốc của runcase 2 từ file của bạn nếu bạn cần dùng)
    testpre = ["calendar"] # just put more scenes to evaluate all of them
    dirstr = './results/'  # the outputs
    tarstr = './HR/'       # the GT
    tar_list = [(tarstr+_) for _ in testpre]
    out_list = [(dirstr+_) for _ in testpre]
    main_script_path = os.path.join(os.path.dirname(__file__), "metrics.py") # Giả sử metrics.py cũng ở cùng thư mục
    cmd1 = ["python3", main_script_path,
        "--output", os.path.join(dirstr,"metric_log/"),
        "--results", ",".join(out_list),
        "--targets", ",".join(tar_list),
    ]
    print(f"Executing metrics.py with command: {' '.join(cmd1)}")
    mycall(cmd1).communicate()
    
elif( runcase == 3 ):
    print("Runcase 3: Train TecoGAN - Not fully implemented in this update, keep original if needed.")
    # (Giữ lại toàn bộ nội dung gốc của runcase 3 từ file của bạn nếu bạn cần dùng, bao gồm tải VGG, FRVSR model,...)
    
elif( runcase == 4 ):
    print("Runcase 4: Train FRVSR - Not fully implemented in this update, keep original if needed.")
    # (Giữ lại toàn bộ nội dung gốc của runcase 4 từ file của bạn nếu bạn cần dùng)

else:
    print(f"Runcase {runcase} không hợp lệ hoặc chưa được triển khai đầy đủ trong bản cập nhật này.")

print(f"Hoàn thành runcase {runcase}.")

