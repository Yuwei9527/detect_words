# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:15:46 2025

@author: aiuser
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

prompt = '''
請逐頁檢查這份文件，根據下列規則找出每一頁的標題，並按順序列出，不需要其他說明：

1. 標題通常位於每頁最上方，或在頁面中央的框框裡。
2. 若有框框，請優先考慮框框內的文字是否為標題。
3. 標題的內容如果有包含「XXX單」或「XXX表」字樣，請優先選取這一行作為標題。
4. 如果該頁沒有「XXX單」或「XXX表」字樣，請輸出你判斷最接近標題的內容（例如最上方或框框內的文字）。
5. 標題不應包含公司名稱或工程行名稱。
6. 只輸出一行你判斷為標題的文字，不需其他解釋。
'''

# 辨識結果旋轉與否
trigger_rotated = True

# model_type = 'Qwen/Qwen2-VL-7B-Instruct'
# model_type = 'Qwen/Qwen2.5-VL-3B-Instruct' # 12.8GB vram
model_type = 'Qwen/Qwen2.5-VL-7B-Instruct' # 21.2GB vram

torch_dtype= torch.bfloat16 # auto, torch.bfloat16

min_pixels = 256*28*28
max_pixels = 1024*28*28

vlm_correction_dict_dir = './錯字寶典.json'
save_dir = 'C:/Users/aiuser/Desktop/lai/detect_pdf_words/detection/prompt_v6_test_rotated/'
root = 'C:/Users/aiuser/Desktop/lai/detect_pdf_words/完工驗收資料(Sample)_0619/'

# 廠商要看的標題
answer_list = [
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
# 預測結果
os.makedirs(save_dir, exist_ok=True)
logfile = open(save_dir + '/console.txt', 'a', encoding="utf-8")
sys.stdout = save_console_as_file(sys.stdout, logfile)

print(prompt, '\n')

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
elif model_type == 'Qwen/Qwen2.5-VL-3B-Instruct':
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_type,
        torch_dtype= torch_dtype,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )
elif model_type == 'Qwen/Qwen2.5-VL-7B-Instruct':
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_type,
        torch_dtype= torch_dtype,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )
else:
    raise

model.eval()

# default processer
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
processor = AutoProcessor.from_pretrained(model_type, min_pixels=min_pixels, max_pixels=max_pixels)


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
result_dict = {}

total_file = 0
target_file = 0

font = ImageFont.truetype('./TaipeiSansTCBeta-Regular.ttf', 100) # font

for full_dir in tqdm(all_files_dir):   
    trigger_合法的附檔名與否 = False
    trigger_使用VLM辨識與否 = False
    
    filename = full_dir.split('\\')[-1]
    filename_ext = filename.split('.')[-1].lower()
    
    folder = full_dir.split(filename)[0].split(str(Path(root)))[1].replace('\\', '/') # 檔案完整路徑 去掉 root filename    
    os.makedirs(save_dir + '/' + folder, exist_ok=True)
    
    if filename_ext == 'pdf': # 只處理pdf
        image = convert_from_path(root + '/' + folder + '/' + filename, dpi=300)
        image = image[0] # 不論pdf的頁數只取第一頁來分析
        trigger_合法的附檔名與否 = True
        trigger_使用VLM辨識與否 = True
    elif filename_ext == 'jpg' or filename_ext == 'jpeg'  or filename_ext == 'png':
        image = Image.open(root + '/' + folder + '/' + filename)
        trigger_合法的附檔名與否 = True
        trigger_使用VLM辨識與否 = True
    elif filename_ext == 'docx':
        document = Document(root + '/' + folder + '/' + filename)
        output_text = document.paragraphs[0].text # 使用第一行的文字當成標題
        trigger_合法的附檔名與否 = True
        trigger_使用VLM辨識與否 = False
    else:
        pass
    
    if trigger_使用VLM辨識與否 == True:
        # img_np  = np.array(image.convert('L'))
        # _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # kernel = np.ones((3,3), np.uint8) # 建立 3x3 的 kernel
        # opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) # 先侵蝕（Erosion），再膨脹（Dilation）=> 開運算
        # clean_img = Image.fromarray(opened)
    
        try:
            osd = pytesseract.image_to_osd(image)
            rotated_angle = 360 - int(osd.split('Rotate:')[1].split('\n')[0])
            if rotated_angle == 360:
                rotated_angle = 0
            if trigger_rotated == True:
                image = image.rotate(rotated_angle, expand=True)
                image.save(save_dir + '/' + folder + '/out_' + filename.split('pdf')[0][:-1] + '.jpg', 'JPEG')
            else:
                image_rotated = image.rotate(rotated_angle, expand=True)
                image_rotated.save(save_dir + '/' + folder + '/out_' + filename.split('pdf')[0][:-1] + '.jpg', 'JPEG')
        except:
            rotated_angle = 'NAN'
            image.save(save_dir + '/' + folder + '/out_' + filename.split('pdf')[0][:-1] + '.jpg', 'JPEG')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(save_dir + '/' + folder + '/out_' + filename.split('pdf')[0][:-1] + '.jpg'),
                    },
                    # {"type": "text", "text": "你現在看到的影像是工廠內的文書資料 畫面中的標題是什麼? 請你單獨把結果輸出"}, # 版本一 # 偶爾關鍵答案前會有一段廢話
                    {"type": "text", "text": prompt}, # 版本二
                ],
            }
        ]
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = output_text[0]
    
    if trigger_合法的附檔名與否 == True:
        # 簡體中文 -> 繁體中文 
        output_text_convert = converter.convert(output_text)  # 漢字
        
        # 全形 -> 半形
        output_text_convert = unicodedata.normalize("NFKC", output_text_convert)
        
        # 去除換行
        output_text_convert = output_text_convert.replace('\n', ' ').replace('\r', ' ')
        
        # 用寶典矯正已知錯字
        for item in last_vlm_correction_dict:
            if item in output_text_convert:
                output_text_convert = output_text_convert.replace(item, last_vlm_correction_dict[item])
        
        # 找出output_text_convert中存在answer_list的連續子字串
        temp = []
        for item in answer_list:
            if item in output_text_convert:
                temp.append(item)
                break
        
        if len(temp) != 0:
            matched = temp[0]
            
            result_dict['檔案路徑'] = full_dir
            result_dict['標題命中與否'] = '√'
            result_dict['標題偵測結果'] = matched
            
            # print(full_dir, '>>>', '√', '<<<', matched)
            
            draw = ImageDraw.Draw(image)
            draw.text(xy=(50, 50), text=str(matched), fill=(46, 139, 87), font=font) # Add text
            # draw.text(xy=(50, 200), text=str(rotated_angle), fill=(0, 0, 0), font=font) # Add text
            image.save(save_dir + '/' + folder + '/out_' + filename.split('pdf')[0][:-1] + '.jpg', 'JPEG')
            
            target_file = target_file + 1
        else:
            result_dict['檔案路徑'] = full_dir
            result_dict['標題命中與否'] = '×'
            result_dict['標題偵測結果'] = output_text_convert
            
            # print(full_dir, '>>>', '×', '<<<', output_text_convert)
            
            draw = ImageDraw.Draw(image)
            draw.text(xy=(50, 50), text=str(output_text_convert), fill=(255, 0, 0), font=font) # Add text
            # draw.text(xy=(50, 200), text=str(rotated_angle), fill=(0, 0, 0), font=font) # Add text
            image.save(save_dir + '/' + folder + '/out_' + filename.split('pdf')[0][:-1] + '.jpg', 'JPEG')
        
        
        
        del image, inputs, image_inputs, video_inputs, generated_ids, \
            generated_ids_trimmed, output_text, output_text_convert, temp
        torch.cuda.empty_cache()
    else:
        result_dict['檔案路徑'] = full_dir
        result_dict['標題命中與否'] = 'NAN'
        result_dict['標題偵測結果'] = ''
        
        # print(full_dir, '>>>',  'NAN', '<<<')
        
        try:
            os.makedirs(save_dir + '/none_pdf/', exist_ok=True)
            shutil.copy(root + '/' + folder + '/' + filename,
                        save_dir + '/none_pdf/' + filename)
        except:
            pass
    total_file = total_file + 1

#%%
df = pd.DataFrame(data=result_dict)
df.to_csv(save_dir + '總管理處_保養管理組_vlm_pred.csv', encoding='utf-8', index=False)

#%%
# 計算正確率
print('正確率:', target_file/total_file)