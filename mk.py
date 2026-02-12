import os
import pickle

# 你的实际图片根目录（多层子目录所在路径）
IMAGE_ROOT = "/root/tomcat/TOMCAT/download_data/data/ut-zappos/images"
# 映射文件保存路径
MAP_PATH = "/root/tomcat/TOMCAT/ut_zappos_path_map.pkl"

# 生成「图片文件名→实际完整路径」的映射
path_map = {}
for root, _, files in os.walk(IMAGE_ROOT):
    for file in files:
        if file.endswith(".jpg"):
            path_map[file] = os.path.join(root, file)

# 保存映射
with open(MAP_PATH, "wb") as f:
    pickle.dump(path_map, f)

print(f"映射生成完成！共匹配 {len(path_map)} 张图片，文件保存在 {MAP_PATH}")