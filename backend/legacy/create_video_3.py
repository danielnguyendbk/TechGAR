import cv2
import os

def create_video_from_images(image_dir, output_video_path, fps=1):
    # Lấy danh sách ảnh và sắp xếp theo số (1.jpg, 2.jpg, ..., 500.jpg)
    images = []
    for i in range(1, 501):
        filename = f"{i}.jpg"
        filepath = os.path.join(image_dir, filename)
        if os.path.exists(filepath):
            images.append(filepath)
            
    if not images:
        print("[Error] Khong tim thay anh nao trong thu muc!")
        return

    print(f"[Info] Dang xu ly {len(images)} anh de tao video...")

    # Đọc ảnh đầu tiên để lấy kích thước
    first_img = cv2.imread(images[0])
    height, width, layers = first_img.shape
    size = (width, height)

    # Khởi tạo VideoWriter (sử dụng codec mp4v cho đuôi .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    count = 0
    for filepath in images:
        img = cv2.imread(filepath)
        if img is None:
            print(f"[Warning] Khong doc duoc anh: {filepath}")
            continue
        
        # Nếu kích thước ảnh khác biệt, resize về kích thước ảnh đầu tiên để tránh lỗi ghi file
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, size)
            
        out.write(img)
        count += 1

    out.release()
    print(f"[Success] Da tao video thanh cong tai: {output_video_path}")
    print(f"[Stats] Tong so khung hinh da ghep: {count}/{len(images)}")

if __name__ == "__main__":
    IMAGE_DIR = r"d:\NCKH\TechGAR\output2"
    OUTPUT_PATH = r"d:\NCKH\TechGAR\output3_video.mp4"
    create_video_from_images(IMAGE_DIR, OUTPUT_PATH, fps=1)
