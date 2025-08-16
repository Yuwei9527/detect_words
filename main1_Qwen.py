# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:15:46 2025

@author: aiuser
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np

# 必要安裝套件
# pip install pdf2image
# pip install python-dateutil
# conda install -c conda-forge poppler
# pip install qwen-vl-utils
# torchaudio-2.6.0 torchvision-0.21.0
# pip install opencv-python
# pip install pytesseract
# pip install accelerate
# pip install flash-attn
# pip install opencc

from pdf2image import convert_from_path

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from tqdm import tqdm
import shutil
from pathlib import Path
import sys
import cv2
from PIL import Image
import opencc
import unicodedata
import pandas as pd
import json
from docx import Document
import datetime
# pip install Spire.Doc # doc -> img 有浮水印
# pip install plum-dispatch==1.7.4
# from docx2pdf import convert # doc不支援

import math
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import re
import string

#%%
import pytesseract # OCR 檢測圖片是否經過翻轉
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # 指定執行檔位置
config = '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'

#%%
# 初始化簡體轉成繁體的工具
converter = opencc.OpenCC('s2t.json')

#%%
# 
class save_console_as_file(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

#%%
# config

# prompt = '''你現在看到的影像是工廠內的文書資料 畫面中的標題是什麼? 請你單獨把結果輸出''' # 版本一

# prompt = '''
# 你是一個使用繁體中文和英文說話的專家
# 在你眼前的影像是工廠內的文書資料，文字的語言是繁體中文或是英文，不可能出現簡體中文
# 請使用結果是...，來描述文件中的標題名稱
# 否則使用結果是：沒有標題''' # 版本二

# prompt = '''
# 你是一位擅長繁體中文與英文的文件分析專家。
# 你面前的影像中包含工廠內部的文書資料，其中文字僅可能為繁體中文或英文，不會出現簡體中文。
# 請的任務是使用「結果是：...」（並將標題內容填入點點處），來逐字識別圖片中的標題文字，不要猜測，也不要修改字詞或是顛倒順序，也不要遺漏任何字，同時確保輸出的文字不是簡體中文。
# 請排除掉出現在標題右手邊的頁碼，例如1/2，這個代表總共有兩頁，這是第一頁。
# 請排除掉任何出現在標題右手邊的版本號，例如Ver3.92或是Version3.92，這個代表第3.92個版本
# '''

# 找標題的提示詞
prompt_get_標題 = '''
請逐頁檢查這份文件，根據下列規則找出每一頁的標題，並按順序列出，不需要其他說明：

1. 標題通常位於每頁最上方，或在頁面中央的框框裡。
2. 若有框框，請優先考慮框框內的文字是否為標題。
3. 標題的內容如果有包含「XXX單」或「XXX表」字樣，請優先選取這一行作為標題。
4. 如果該頁沒有「XXX單」或「XXX表」字樣，請輸出你判斷最接近標題的內容（例如最上方或框框內的文字）。
5. 標題不應包含公司名稱或工程行名稱。
6. 只輸出一行你判斷為標題的文字，不需其他解釋。
'''

# 找日期的提示詞 <- in table
# 工程承攬切結書偵測出「施工期限」
# 施工作業安全告知單(廠商適用)偵測出「預定工期、合約工期、安全告知日期」
# 施工作業安全告知單(施工人員適用)偵測出「預定工期、合約工期」
# 工作安全分析JSA記錄偵測出「訓練日期」

# prompt_get_time = '''
# 請逐頁檢查這份文件，根據下列規則找出每一頁的日期，並按順序列出，不需要其他說明：

# 1. 日期通常位於頁面的框框內。
# 2. 請先告訴你找到那一個的時間 例如：「施工期限、預定工期、合約工期、安全告知日期和訓練日期」再告訴我對應的日期是什麼時候。
# 3. 請優先偵測出施工期限、預定工期、合約工期、安全告知日期或是訓練日期。
# 4. 如果畫面中同時出現多個優先偵測的目標，那就按順序全部顯示出來。
# 5. 日期的表達方式是西元或是民國年。
# 6. 不應該包含製表日期、公佈日期、修訂日期、文件制定日期或是第幾次修訂。
# 7. 只輸出一行你判斷為日期的文字，不需其他解釋。

# 以下的範例是文件中實際的狀況
# 施工期限：
# (民國年的版本) 自民國XX年XX月XX日起至XX年XX月XX日止共XXX日曆天
# (西元年的版本) 自XXXX年XX月XX日起至XXXX年XX月XX日止共XXX日曆天

# 預定期限：
# (民國年的版本) XX年XX月XX日~XX年XX月XX日
# (西元年的版本) XXXX年XX月XX日~XXXX年XX月XX日

# 合約工期：
# (民國年的版本) XX年XX月XX日~XX年XX月XX日
# (西元年的版本) XXXX年XX月XX日~XXXX年XX月XX日

# 安全告知日期：
# (民國年的版本) XX年XX月XX日
# (西元年的版本) XXXX年XX月XX日

# 訓練時間：
# (民國年的版本) 民國XX年XX月XX日
# (西元年的版本) XXXX年XX月XX日
# '''

prompt_get_施工期限 = '''
你是一个只能根据图片观察来判断的助理。你眼前看到的是一张图片，请找出每张影像中的「施工期限」及其对应的「日期时间」，不要添加或刪除任何字元，也不要推測缺失的部分。

以下的範例是文件中實際的狀況
1. 施工期限：自民国109年08月01日起至110年07月31日止共730日历天
2. 施工期限：自2020年08月01日起至2021年07月31日止共730日历天

请参考这个文字输出文字：
施工期限：自YYYY年MM月DD日起至YYYY年MM月DD日止共AAA日历天
施工期限：找不到

请遵守以上规则。
'''

prompt_get_安全告知日期 = '''
你是一个只能根据图片观察来判断的助理。你眼前看到的是一张图片，请找出每张影像中的「安全告知日期」及其对应的「日期时间」，不要添加或刪除任何字元，也不要推測缺失的部分。

以下的範例是文件中實際的狀況
1. 安全告知日期：2025年08月01日

请参考这个文字输出文字：
安全告知日期：YYYY年MM月DD日
安全告知日期：找不到

请遵守以上规则。
'''

prompt_get_訓練日期 = '''
你是一个只能根据图片观察来判断的助理。你眼前看到的是一张图片，请找出每张影像中的「训练日期」及其对应的「日期时间」，不要添加或刪除任何字元，也不要推測缺失的部分。

以下的範例是文件中實際的狀況
1. 訓練日期：2025年08月01日
2. 訓練日期：109年01月03日

请参考这个文字输出文字：
訓練日期：YYYY年MM月DD日
訓練日期：找不到

请遵守以上规则。
'''

# 辨識結果旋轉與否
trigger_rotated = True

# model_type = 'Qwen/Qwen2-VL-7B-Instruct'
# model_type = 'Qwen/Qwen2.5-VL-3B-Instruct' # 12.8GB vram
model_type = 'Qwen/Qwen2.5-VL-7B-Instruct' # 21.2GB vram 【V】
# model_type = 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ'
# model_type = 'OpenGVLab/InternVL3-8B'
# model_type = 'OpenGVLab/InternVL3-14B-AWQ'

torch_dtype= torch.bfloat16 # auto, torch.bfloat16

min_pixels = 256*28*28
max_pixels = 1024*28*28

vlm_correction_dict_dir = './錯字寶典.json'
time_gt_dir =  'C:/Users/aiuser/Desktop/lai/detect_pdf_words/完工驗收資料(Sample)_0619_實際日期時間.xlsx'
save_dir = 'C:/Users/aiuser/Desktop/lai/detect_pdf_words/detection/prompt_v6_test_rotated_time_testing_0815/'
root = 'C:/Users/aiuser/Desktop/lai/detect_pdf_words/完工驗收資料(Sample)_0619/'

# 廠商要看的標題
last_answer_title_list = [
    '施工記錄表',
    '施工前後及過程照片(監工)',
    '保養帶料進出廠清單',
    '材料檢驗表(帶料)',
    '工程承攬切結書',
    '施工作業安全告知單(廠商適用)',
    '施工作業安全告知單(施工人員適用)',
    '工作安全分析JSA記錄',
    '子案工號申請單(保養定檢專案)',
    '施工品質檢查單',
    '施工品質差異表',
    '保養維修異常扣款清單',
    '保養維修彙總完工明細表',
    '保底金額核算表',
    '開工協調會議記錄',
    '其他(如檢測報告)'
]

# 要另外偵測的工作
work_list = ['日期擷取']

# 標準答案日期時間

last_answer_time_df = pd.read_excel(time_gt_dir, sheet_name=None)
last_answer_time_df = last_answer_time_df[list(last_answer_time_df)[0]]
last_answer_time_df_group = last_answer_time_df.groupby('資料夾').groups
last_answer_time_df_group_name = list(last_answer_time_df_group)

#%%
# VLM辨識寶典
last_vlm_correction_dict = {
  "查覈": "查核",
  "加修": "叫修",
  "承諾": "承攬",
  "案總": "彙總",
  "通用": "適用",
  "選擇": "遴選"
}

if os.path.isfile(vlm_correction_dict_dir) == True:
    # load
    with open('./錯字寶典.json', encoding="utf-8") as f:
        previous_vlm_correction_dict = json.load(f)
    if (last_vlm_correction_dict == previous_vlm_correction_dict) == True:
        pass
    else:
        # save
        with open('./錯字寶典.json', 'w', encoding='utf-8') as f:
            json.dump(last_vlm_correction_dict, f, indent=2, ensure_ascii=False)
else:
    # save
    with open('./錯字寶典.json', 'w', encoding='utf-8') as f:
        json.dump(last_vlm_correction_dict, f, indent=2, ensure_ascii=False)

#%%
# 把print()內容存起來
os.makedirs(save_dir, exist_ok=True)
logfile = open(save_dir + '/console.txt', 'a', encoding="utf-8")
sys.stdout = save_console_as_file(sys.stdout, logfile)

print(prompt_get_標題 + '\n')
print(prompt_get_施工期限 + '\n')
print(prompt_get_安全告知日期 + '\n')
print(prompt_get_訓練日期 + '\n')

# 本次使用的VLM模型
for item in Path(save_dir).rglob('*.md'):
    os.remove(str(item))
with open(save_dir + '/' + model_type.split('/')[-1] + '.md', 'w', encoding="utf-8"):
    pass

#%%
# default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
if model_type == 'Qwen/Qwen2-VL-7B-Instruct':
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_type,
        torch_dtype= torch_dtype,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_type, min_pixels=min_pixels, max_pixels=max_pixels)
elif model_type == 'Qwen/Qwen2.5-VL-3B-Instruct':
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_type,
        torch_dtype= torch_dtype,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_type, min_pixels=min_pixels, max_pixels=max_pixels)
elif model_type == 'Qwen/Qwen2.5-VL-7B-Instruct':
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_type,
        torch_dtype= torch_dtype,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_type, min_pixels=min_pixels, max_pixels=max_pixels)
elif model_type == 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ':
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct",
    torch_dtype=torch_dtype,
    # attn_implementation="flash_attention_2",
    device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_type, min_pixels=min_pixels, max_pixels=max_pixels)
    
elif model_type == 'OpenGVLab/InternVL3-8B':
    model = AutoModel.from_pretrained(
        model_type,
        torch_dtype=torch_dtype,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True, use_fast=False)
elif model_type == 'OpenGVLab/InternVL3-14B-AWQ':
    model = AutoModel.from_pretrained(
        model_type,
        torch_dtype=torch_dtype,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True, use_fast=False)
else:
    raise

model.eval()

#%%
all_files_dir = []
for item in Path(root).rglob('*'): # 撈出路徑內所有檔案和資料夾
    if item.is_file():
        full_dir = str(item)
        full_dir_ext = full_dir.split('.')[-1]
        if full_dir.split(str(Path(root)))[1].split('\\')[1][-1] != 'X': # and full_dir.split(str(Path(root)))[1].split('\\')[1] == 'C04工作安全分析JSA記錄 +日期擷取':
            if full_dir_ext != 'db':
                all_files_dir.append(full_dir)
    elif item.is_dir():
        pass

#%%
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def auto_rotate_iamge(image=None,
                      nun_page=0,
                      save_dir=None,
                      folder=None,
                      filename=None,
                      filename_ext=None
    ):
    image_save_dir = save_dir + '/' + folder + '/out_' + filename.split(filename_ext)[0][:-1] + '_' + str(nun_page) + '.jpg'
    try:
        osd = pytesseract.image_to_osd(image)
        rotated_angle = 360 - int(osd.split('Rotate:')[1].split('\n')[0])
        if rotated_angle == 360:
            rotated_angle = 0
        if trigger_rotated == True:
            image = image.rotate(rotated_angle, expand=True)
        else:
            pass
    except:
        rotated_angle = 'NAN'
    return image, image_save_dir

def rm_white_area_coordinate(image_gray):
    _, otsu_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_image = cv2.morphologyEx(otsu_binary, cv2.MORPH_OPEN, kernel)

    # (array([   0,    0,    0, ..., 2207, 2207, 2207], dtype=int64),
    #  array([3482, 3483, 3484, ..., 1417, 1418, 1419], dtype=int64))
    temp = np.where(cleaned_image != 0)
    x_min, y_min = min(temp[1]), min(temp[0])
    x_max, y_max = max(temp[1]), max(temp[0])

    # _, _, stats, _ = cv2.connectedComponentsWithStats(dilate_image, connectivity=8)
    # df = pd.DataFrame(stats)
    # df_index = df[4].sort_values(ascending=False)
    # area_index = list(df.loc[list(df_index.index)[1]][:-1])

    return y_min, y_max, x_min, x_max

def vlm_text_detection(
        model_type=None,
        image_dir=None,
        prompt_get_target=None,
    ):
    if model_type.split('/')[0] == 'Qwen':
        if prompt_get_target != None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": str(image_dir),
                        },
                        {"type": "text", "text": prompt_get_target}, # 版本二
                    ],
                }
            ]
        else:
            raise ValueError('messages不能是空的')
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        temp_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = ' '.join(temp_output_text)
    elif model_type.split('/')[0] == 'OpenGVLab':
        if prompt_get_target != None:
            messages = prompt_get_target
        else:
            raise ValueError('messages不能是空的')
        pixel_values = load_image(image_dir, max_num=1).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=512, do_sample=True, eos_token_id=151645, pad_token_id=151645)
        
        output_text = model.chat(tokenizer, pixel_values, '<image>\n' + messages, generation_config)
    return output_text
    
def text_processing(text):
    # 簡體中文 -> 繁體中文
    output_text_convert = converter.convert(text)
    # 全形 -> 半形
    output_text_convert = unicodedata.normalize("NFKC", output_text_convert)
    # 去除換行
    output_text_convert = output_text_convert.replace('\n', '').replace('\r', '')
    # 去除空格
    output_text_convert = output_text_convert.replace(' ', '')
    return output_text_convert


#%%
class CONVERT_TIME_FORMAT():
    def __init__(self, detect_time):
        self.detect_time = detect_time
        
    # 民國年 -> 西元年
    def year(self):
        path = []
        detect_time = self.detect_time.replace('民國', '')
        try:
            detect_time_rm_年_list = detect_time.split('年')
            for i in range(len(detect_time_rm_年_list)):
                自_index = detect_time_rm_年_list[i].find('自')
                至_index = detect_time_rm_年_list[i].find('至')
                if 自_index != -1:
                    年 = detect_time_rm_年_list[i][自_index+1:]
                    if int(年) <= 1911:
                        path.append(detect_time_rm_年_list[i].replace(
                            年, str(int(年) + 1911))
                        )
                    else:
                        path.append(detect_time_rm_年_list[i])
                elif 至_index != -1:
                    年 = detect_time_rm_年_list[i][至_index+1:]
                    if int(年) <= 1911:
                        path.append(detect_time_rm_年_list[i].replace(
                            年, str(int(年) + 1911))
                        )
                    else:
                        path.append(detect_time_rm_年_list[i])
                else:
                    path.append(detect_time_rm_年_list[i])
            process_time = '年'.join(path)
            return process_time
        except:
            return detect_time
    
    def month_day(self):
        path = []
        temp = []
        detect_time = self.detect_time.replace('民國', '')
        try:
            detect_time_rm_月_list = detect_time.split('月')
            for i in range(len(detect_time_rm_月_list)):
                年_index = detect_time_rm_月_list[i].find('年')
                日_index = detect_time_rm_月_list[i].find('日')
                if 年_index != -1:
                    月 = detect_time_rm_月_list[i][年_index+1:]
                    try:
                        int(月)
                        if len(月) == 1:
                            path = detect_time_rm_月_list[i].replace(
                                '年' + 月, '年0' + str(月))
                        else:
                            path = detect_time_rm_月_list[i]
                    except:
                        path = detect_time_rm_月_list[i]
                else:
                    path = detect_time_rm_月_list[i]
                if 日_index != -1:
                    日 = detect_time_rm_月_list[i][:日_index]
                    if path != []:
                        try:
                            int(日)
                            if len(日) == 1:
                                path = (path.replace(
                                    日 + '日', '0' + 日 + '日'))
                        except:
                            pass
                    else:
                        pass
                temp.append(path)
            process_time = '月'.join(temp)
            return process_time
        except:
            return detect_time
        
#%%
final_full_dir = []
final_full_dir_page = []
final_full_dir_ext = []
final_matched_title = []
final_matched_time = []
final_detect_title = []
final_detect_time = []

# total_file = 0
# target_file = 0

# 顯示在圖片中的字體
font = ImageFont.truetype('./TaipeiSansTCBeta-Regular.ttf', 100) # font
font2 = ImageFont.truetype('./TaipeiSansTCBeta-Regular.ttf', 80) # font

# 在圖片周圍增加空白區域
right = 0
left = 0
top = 500
bottom = 0

# 排除中文和英文的符號
punct = string.punctuation # 英文標點
ch_punct = "，。、？！：；" # 常見中文標點
all_punct = punct + ch_punct

# 影像處理手法
clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8,8)) # clipLimit==10.0 OK

for full_dir in tqdm(all_files_dir):
    filename = full_dir.split('\\')[-1]
    filename_ext = filename.split('.')[-1].lower()
    folder = full_dir.split(filename)[0].split(str(Path(root)))[1].replace('\\', '/') # 檔案完整路徑 去掉 root filename
    os.makedirs(save_dir + '/' + folder, exist_ok=True)
    folder_temp = folder.split('/')[1]
    
    if filename_ext == 'pdf':
        vlm_runable = True
        images = convert_from_path(root + '/' + folder + '/' + filename, dpi=300)
    elif filename_ext == 'jpg' or filename_ext == 'jpeg'  or filename_ext == 'png':
        vlm_runable = True
        image = Image.open(root + '/' + folder + '/' + filename)
    elif filename_ext == 'docx':
        vlm_runable = False
        document = Document(root + '/' + folder + '/' + filename)
        doc = document.paragraphs
        for nun_page in range(0, len(doc)):
            text = doc[nun_page].text
            # 標題
            detect_title = text_processing(text=text)
            for item in last_vlm_correction_dict: # 用寶典矯正已知錯字
                if item in detect_title:
                    detect_title = detect_title.replace(item, last_vlm_correction_dict[item])
            temp2 = []
            for item in last_answer_title_list: # 找出output_text_convert中存在answer_title_list的連續子字串
                if item in detect_title:
                    temp2.append(item)
                    break
            # 日期時間
            detect_time = ''
            
            if len(temp2) != 0:
                final_full_dir.append(full_dir)
                final_full_dir_page.append(nun_page)
                final_full_dir_ext.append(filename_ext)
                final_matched_title.append('OK')
                final_detect_title.append(detect_title)
            else:
                final_full_dir.append(full_dir)
                final_full_dir_page.append(nun_page)
                final_full_dir_ext.append(filename_ext)
                final_matched_title.append('NG')
                final_detect_title.append(detect_title)
            
            final_matched_time.append('NAN')
            final_detect_time.append(detect_time)
    else:
        vlm_runable = False
        try:
            os.makedirs(save_dir + '/none_pdf_or_img_docx/', exist_ok=True)
            shutil.copy(root + '/' + folder + '/' + filename,
                        save_dir + '/none_pdf_or_img_docx/' + filename)
        except:
            pass
        final_full_dir.append(full_dir)
        final_full_dir_page.append(0)
        final_full_dir_ext.append(filename_ext)
        final_matched_title.append('NAN')
        final_matched_time.append('NAN')
        final_detect_title.append('')
        final_detect_time.append('')
    
    if vlm_runable == True:
        for nun_page in range(len(images)):
            image = images[nun_page] # 不論pdf的頁數只取第一頁來分析
            rotate_iamge, image_dir = auto_rotate_iamge(
                image=image,
                nun_page=nun_page,
                save_dir=save_dir,
                folder=folder,
                filename=filename,
                filename_ext=filename_ext
            )
            # 客製化ROI區域
            temp_image_folder = '/'.join(image_dir.split('/')[:-1]) + '/temp/'
            os.makedirs(temp_image_folder, exist_ok=True)
            
            roi_tilte_img_dir = temp_image_folder + filename.split('.')[0] + '_' + str(nun_page) + '_t.jpg'          # 輸入給VLM辨識「標題」的圖片路徑
            rotate_iamge.save(roi_tilte_img_dir)                                             # 儲存 待偵測的標題圖片
            rotate_iamge_gray = cv2.cvtColor(np.asarray(rotate_iamge), cv2.COLOR_RGB2GRAY)
            rotate_iamge_gray_enh = clahe.apply(rotate_iamge_gray)                           # 影像對比度強化
            y_min, y_max, x_min, x_max = rm_white_area_coordinate(rotate_iamge_gray_enh)
            rotate_iamge_gray_enh = Image.fromarray(rotate_iamge_gray_enh)
            
            rotate_iamge_gray_enh = rotate_iamge_gray_enh.crop((x_min-50, y_min-200, x_max+50, y_max+200)) # 找出關鍵區域
            
            width, height = rotate_iamge_gray_enh.size
            
            if folder_temp == 'C01工程承攬切結書 +日期擷取':
                crop_area = (0, height//7, width//1.5, height//3)
                croped = rotate_iamge_gray_enh.crop(crop_area)                       # 局部的關鍵區域
                roi_cp_img_dir = temp_image_folder + filename.split('.')[0] + '_' + str(nun_page) + '_cp.jpg'    # 輸入給VLM辨識「施工期限」的圖片路徑
                croped.save(roi_cp_img_dir, 'JPEG')                                        # 儲存 待偵測的「施工期限」圖片
            elif folder_temp == 'C02施工作業安全告知單(廠商適用) +日期擷取':
                crop_area = (width//1.3, height//2, width, height//1.2)
                croped = rotate_iamge_gray_enh.crop(crop_area)                       # 局部的關鍵區域
                roi_snd_img_dir = temp_image_folder + filename.split('.')[0] + '_' + str(nun_page) + '_snd.jpg'  # 輸入給VLM辨識「安全告知日期」的圖片路徑
                croped.save(roi_snd_img_dir, 'JPEG')                                       # 儲存 待偵測的「安全告知」圖片
            elif folder_temp == 'C04工作安全分析JSA記錄 +日期擷取':
                crop_area = (0, height//6, width, height//2)
                croped = rotate_iamge_gray_enh.crop(crop_area)                       # 局部的關鍵區域
                roi_td_img_dir = temp_image_folder + filename.split('.')[0] + '_' + str(nun_page) + '_td.jpg'    # 輸入給VLM辨識「訓練日期」的圖片路徑
                croped.save(roi_td_img_dir, 'JPEG')                                        # 儲存 待偵測的「訓練日期」圖片
            
            # 辨識標題
            detect_title = vlm_text_detection(
                model_type=model_type,
                image_dir=roi_tilte_img_dir,
                prompt_get_target=prompt_get_標題,
            )
            detect_title = text_processing(text=detect_title) # 整理成簡體改成繁體、全形改成半形、換行改成空格
            for item in last_vlm_correction_dict: # 用寶典矯正已知錯字
                if item in detect_title:
                    detect_title = detect_title.replace(item, last_vlm_correction_dict[item])
            
            # 辨識日期時間
            if folder_temp == 'C01工程承攬切結書 +日期擷取':
                detect_date = vlm_text_detection(
                    model_type=model_type,
                    image_dir=roi_cp_img_dir,
                    prompt_get_target=prompt_get_施工期限,
                )
                detect_date = text_processing(text=detect_date) # 整理成簡體改成繁體、全形改成半形、換行改成空格
                detect_date = re.sub(f"[{re.escape(all_punct)}]", "", detect_date)
                detect_date = detect_date.replace('施工期限', '')
            elif folder_temp == 'C02施工作業安全告知單(廠商適用) +日期擷取':
                detect_date = vlm_text_detection(
                    model_type=model_type,
                    image_dir=roi_snd_img_dir,
                    prompt_get_target=prompt_get_安全告知日期,
                )
                detect_date = text_processing(text=detect_date) # 整理成簡體改成繁體、全形改成半形、換行改成空格
                detect_date = re.sub(f"[{re.escape(all_punct)}]", "", detect_date)
                detect_date = detect_date.replace('安全告知日期', '')
            elif folder_temp == 'C04工作安全分析JSA記錄 +日期擷取':
                detect_date = vlm_text_detection(
                    model_type=model_type,
                    image_dir=roi_td_img_dir,
                    prompt_get_target=prompt_get_訓練日期,
                )
                detect_date = text_processing(text=detect_date) # 整理成簡體改成繁體、全形改成半形、換行改成空格
                detect_date = re.sub(f"[{re.escape(all_punct)}]", "", detect_date)
                detect_date = detect_date.replace('訓練日期', '')
            else:
                manual_data = '經過人工判讀畫面中沒有要檢測的日期時間'
            
            width, height = rotate_iamge.size
            
            new_width = width + right + left
            new_height = height + top + bottom
            
            result = Image.new(rotate_iamge.mode, (new_width, new_height), (255, 255, 255))
            result.paste(rotate_iamge, (left, top))
            
            # 對標題的答案
            temp2 = []
            for item in last_answer_title_list: # 找出output_text_convert中存在answer_title_list的連續子字串
                if item in detect_title:
                    temp2.append(item)
                    break
            try:
                temp3 = []
                detect_date = CONVERT_TIME_FORMAT(detect_date).year() # 統一日期格式
                detect_date = CONVERT_TIME_FORMAT(detect_date).month_day() # 統一日期格式
                detect_date_rm_symbol = detect_date.replace('。', '').replace('，', '').replace(':', '').replace('?', '') # 刪除標點符號
                del detect_date # 刪除用不到的變數
                
                # 對日期時間的答案
                if folder_temp in last_answer_time_df_group_name:
                    df_time = last_answer_time_df.loc[last_answer_time_df_group[folder_temp]]
                    df_time_index = list(df_time.index)[int(np.where(df_time['檔名'] == filename)[0][0])]
                    item = df_time['日期時間(實際)'][df_time_index]
                    if type(item) == datetime.datetime:
                        items = [item.strftime('%Y年%m月%d日')]
                    elif type(item) == str:
                        items = item.split(' ')
                        try:
                            temp4 = []
                            for item in items:
                                temp4.append(item.strptime('%Y年%m月%d日'))
                            items = temp4
                        except:
                            pass
                    for j in range(len(items)):
                        if items[j] == detect_date_rm_symbol:
                            temp3.append(detect_date_rm_symbol)
            except:
                pass
            
            draw = ImageDraw.Draw(result)
            
            # 標題
            if len(temp2) != 0:
                final_matched_title.append('OK')
                final_detect_title.append(detect_title)
                draw.text(xy=(50, 50), text=str(temp2[0]),
                          fill=(46, 139, 87), font=font) # Add text
            else:
                final_matched_title.append('NG')
                final_detect_title.append(detect_title)
                draw.text(xy=(50, 50), text=str(detect_title),
                          fill=(255, 0, 0), font=font) # Add text
            
            # 日期時間
            if len(temp3) != 0:
                final_matched_time.append('OK')
                final_detect_time.append(' '.join(temp3))
                draw.text(xy=(50, 200), text=str(' '.join(temp3)),
                          fill=(46, 139, 87), font=font2) # Add text
            else:
                try:
                    manual_data
                    final_matched_time.append('-')
                    final_detect_time.append(manual_data)
                    del manual_data
                except:
                    final_matched_time.append('NG')
                    final_detect_time.append(detect_date_rm_symbol)
                    draw.text(xy=(50, 200), text=detect_date_rm_symbol,
                              fill=(255, 0, 0), font=font2) # Add text
            
            result.save(image_dir, 'JPEG')
            
            final_full_dir.append(full_dir)
            final_full_dir_page.append(nun_page)
            final_full_dir_ext.append(filename_ext)
    else:
        pass

#%%
df = pd.DataFrame(
    {
    "檔案路徑": final_full_dir,
    "頁碼": final_full_dir_page,
    "附檔名": final_full_dir_ext,
    "標題命中與否": final_matched_title,
    "標題偵測結果": final_detect_title,
    "日期命中與否": final_matched_time,
    "日期偵測結果": final_detect_time
    }
)
df.to_csv(save_dir + '總管理處_保養管理組_vlm_pred.csv', encoding='big5', index=False, errors='ignore')
