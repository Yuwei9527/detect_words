# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 08:12:02 2025

@author: aiuser
"""

#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

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
# pip install xlsxwriter

from pdf2image import convert_from_path

import torch
from ollama import chat
from ollama import ChatResponse
from tqdm import tqdm
import re

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration # TorchAoConfig
from transformers import Qwen3VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
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
from datetime import datetime as dt
# pip install Spire.Doc # doc -> img 有浮水印
# pip install plum-dispatch==1.7.4
# from docx2pdf import convert # doc不支援

# import math
import torchvision.transforms as T
# from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import re
import string
from concurrent.futures import ThreadPoolExecutor, TimeoutError


#%%
import pytesseract # OCR 檢測圖片是否經過翻轉
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # 指定執行檔位置
config = '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'

#%%
# 初始化簡體轉成繁體的工具
converter = opencc.OpenCC('s2t.json')

#%%
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
def auto_rotate_iamge(image=None,
                      nun_page=0,
                      save_dir=None,
                      folder=None,
                      filename=None,
                      filename_ext=None):
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
        options=None,
        stpe_word_list=None,
    ):
    output_text = ''
    if prompt_get_target != None:
        try:
            if options == None:
                response: ChatResponse = chat(
                    model=model_type,
                    # options={              # 👈 VLM重要參數
                        # "temperature": 0.1,
                        # "top_p": 1,
                        # "top_k": 10,
                        # "stpe": stpe_word_list
                        # "mirostat": 0,
                        # "num_predict": 512, # -1
                        # "repeat_penalty": 1.5,
                        # "repeat_last_n": 5,
                    # },
                   	messages=[
                    {
                        'role': 'user',
                        'content': prompt_get_target,
                        'images': [image_dir]
                    }
                   ],
                   stream=True
                )
            elif type(options) == type({}):
                response: ChatResponse = chat(
                    model=model_type,
                    options=options,
                   	messages=[
                    {
                        'role': 'user',
                        'content': prompt_get_target,
                        'images': [image_dir]
                    }
                   ],
                   stream=True
                )
            for chunk in response: # 即時判斷是否有滿足stpe_word_list
                text = chunk['message']['content']
                # print(text, end='', flush=True)
                output_text += text
                
                if type(stpe_word_list) == None or type(stpe_word_list) != list:
                    pass
                else:
                    if output_text in stpe_word_list:
                        break
        except KeyboardInterrupt:
            pass
    else:
        raise ValueError('messages不能是空的')
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

# 避免字串出現疊字
def remove_duplicate_chars(text):
    # (.)\1+ 代表：任意字元後面跟著相同的字元 1 次以上
    return re.sub(r'(.)\1+', r'\1', text)

class CONVERT_DATETIME_FORMAT():
    def __init__(self, detect_date):
        self.detect_date = detect_date
        
    # 民國年 -> 西元年
    def year(self):
        path = []
        detect_date = self.detect_date.replace('民國', '')
        try:
            detect_datetime_rm_年_list = detect_date.split('年')
            for i in range(len(detect_datetime_rm_年_list)):
                自_index = detect_datetime_rm_年_list[i].find('自')
                至_index = detect_datetime_rm_年_list[i].find('至')
                if 自_index != -1:
                    年 = detect_datetime_rm_年_list[i][自_index+1:]
                    if int(年) <= 1911:
                        path.append(detect_datetime_rm_年_list[i].replace(
                            年, str(int(年) + 1911))
                        )
                    else:
                        path.append(detect_datetime_rm_年_list[i])
                elif 至_index != -1:
                    年 = detect_datetime_rm_年_list[i][至_index+1:]
                    if int(年) <= 1911:
                        path.append(detect_datetime_rm_年_list[i].replace(
                            年, str(int(年) + 1911))
                        )
                    else:
                        path.append(detect_datetime_rm_年_list[i])
                else:
                    path.append(detect_datetime_rm_年_list[i])
            process_date = '年'.join(path)
            return process_date
        except:
            return detect_date
    
    def month_day(self):
        path = []
        temp = []
        detect_date = self.detect_date.replace('民國', '')
        try:
            detect_datetime_rm_月_list = detect_date.split('月')
            for i in range(len(detect_datetime_rm_月_list)):
                年_index = detect_datetime_rm_月_list[i].find('年')
                日_index = detect_datetime_rm_月_list[i].find('日')
                if 年_index != -1:
                    月 = detect_datetime_rm_月_list[i][年_index+1:]
                    try:
                        int(月)
                        if len(月) == 1:
                            path = detect_datetime_rm_月_list[i].replace(
                                '年' + 月, '年0' + str(月))
                        else:
                            path = detect_datetime_rm_月_list[i]
                    except:
                        path = detect_datetime_rm_月_list[i]
                else:
                    path = detect_datetime_rm_月_list[i]
                if 日_index != -1:
                    日 = detect_datetime_rm_月_list[i][:日_index]
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
            process_date = '月'.join(temp)
            return process_date
        except:
            return detect_date


#%% 提示詞

prompt_get_標題 = '''
請逐頁檢查這份文件，根據下列規則找出每一頁的標題，不需要其他說明：

1. 標題通常位於每頁上邊緣的中心點、下邊緣的中心點、左邊緣的中心點、右邊緣的中心點，或在頁面中的框框裡。
2. 若有框框，請優先考慮框框內的文字是否為標題。
3. 標題的內容如果有包含「XXX單」或「XXX表」字樣，請優先選取這一行作為標題。
4. 如果該頁沒有「XXX單」或「XXX表」字樣，請輸出你判斷最接近標題的內容（例如最上方或框框內的文字）。
5. 標題不應包含公司名稱或工程行名稱。
6. 只輸出一行你判斷為標題的文字，不需其他解釋。
'''

# 待修改
prompt_get_施工期限 = '''
請逐頁檢查這份文件，根據下列規則找出每一頁的施工期限，不需要其他說明：

1. 施工期限的內容由固定的中文模板結構搭配人工手寫阿拉伯數字組成，輸出格式必須是「自YYYY年MM月DD日起至yyyy年mm月dd日止共XXX日曆天」。
2. 其中「自」、「年」、「月」、「日」、「起」、「至」、「止」、「共」、「日曆天」這些中文字必定固定。
3. 圖片上的施工期限發生在PLACEHOLDER_TIME之前。
4. 7個手寫欄位YYYY、MM、DD、yyyy、mm、dd、XXX皆為阿拉伯數字（0–9），由人工書寫，因此字形可能不規則但字義明確。
5. 請在內部逐步檢視影像中的筆畫形狀、方向、粗細、斷點與連接方式後，再做出最終判斷。
6. 你必須進行完整的逐步視覺推理，但禁止在輸出中透露任何推理過程、分析、描述或中間想法。
7. 最終輸出時，只輸出「施工期限」句子本身，不得補充任何解釋、推論、重寫或其他附加內容。
'''

prompt_get_安全告知日期 = '''
請逐頁檢查這份文件，根據下列規則找出每一頁的安全告知日期，不需要其他說明：

1. 安全告知日期的內容由固定的中文模板結構搭配人工手寫阿拉伯數字組成，輸出格式必須是「YYYY年MM月DD日」。
2. 其中「年」、「月」、「日」這些中文字必定固定。
3. 圖片上的安全告知日期發生在PLACEHOLDER_TIME之前。
4. 3個手寫欄位YYYY、MM、DD皆為阿拉伯數字（0–9），由人工書寫，因此字形可能不規則但字義明確。
5. 請在內部逐步檢視影像中的筆畫形狀、方向、粗細、斷點與連接方式後，再做出最終判斷。
6. 你必須進行完整的逐步視覺推理，但禁止在輸出中透露任何推理過程、分析、描述或中間想法。
7. 最終輸出時，只輸出「安全告知日期」句子本身，不得補充任何解釋、推論、重寫或其他附加內容。
'''

prompt_get_訓練日期 = '''
請逐頁檢查這份文件，根據下列規則找出每一頁的訓練日期，不需要其他說明：

1. 訓練日期的內容由固定的中文模板結構搭配人工手寫阿拉伯數字組成，輸出格式必須是「YYYY年MM月DD日」。
2. 其中「年」、「月」、「日」這些中文字必定固定。
3. 圖片上的訓練日期發生在PLACEHOLDER_TIME之前。
4. 3個手寫欄位YYYY、MM、DD皆為阿拉伯數字（0–9），由人工書寫，因此字形可能不規則但字義明確。
5. 請在內部逐步檢視影像中的筆畫形狀、方向、粗細、斷點與連接方式後，再做出最終判斷。
6. 你必須進行完整的逐步視覺推理，但禁止在輸出中透露任何推理過程、分析、描述或中間想法。
7. 最終輸出時，只輸出「訓練日期」句子本身，不得補充任何解釋、推論、重寫或其他附加內容。
'''

#%%
# 辨識結果旋轉與否
trigger_rotated = True

# model_type = 'qwen3-vl:8b-instruct-q4_K_M'
model_type = 'qwen3-vl:8b-instruct-q8_0'
# model_type = 'qwen3-vl:8b-instruct-bf16' # V100不支援這個精度
# model_type = 'qwen3-vl:30b'
# model_type = 'qwen2.5vl:7b-fp16'
# model_type = 'gemma3:12b'
# model_type = 'minicpm-v:8b-2.6-fp16'


vlm_correction_dict_dir = './錯字寶典.json'
time_gt_dir =  'C:/Users/aiuser/Desktop/lai/detect_pdf_words/完工驗收資料(Sample)_0619_實際日期時間.xlsx'

timeout_seconds = 120 # timeout門檻

# 廠商要看的標題
# TODO 要一字不漏
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
last_answer_datetime_df = pd.read_excel(time_gt_dir, sheet_name=None)
last_answer_datetime_df = last_answer_datetime_df[list(last_answer_datetime_df)[0]]
last_answer_time_df_group = last_answer_datetime_df.groupby('資料夾').groups
last_answer_time_df_group_name = list(last_answer_time_df_group)

# VLM辨識寶典
last_vlm_correction_dict = {
  "查覈": "查核",
  "加修": "叫修",
  "承諾": "承攬",
  "案總": "彙總",
  "通用": "適用",
  "選擇": "遴選",
  "日歷天": "日曆天"
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

root = 'C:/Users/aiuser/Desktop/lai/detect_pdf_words/完工驗收資料(Sample)_0619/'
for folder_name in os.listdir(root):
    save_dir = 'C:/Users/aiuser/Desktop/lai/detect_pdf_words/detection/20251125_' + model_type.replace(':', '_').replace('/', '_') + '/' + folder_name
    os.makedirs(save_dir, exist_ok=True)
    dict_prompt = dict()
    dict_prompt['prompt_get_標題'] = prompt_get_標題
    dict_prompt['prompt_get_施工期限'] = prompt_get_施工期限
    dict_prompt['prompt_get_安全告知日期'] = prompt_get_安全告知日期
    dict_prompt['prompt_get_訓練日期'] = prompt_get_訓練日期
    with open(save_dir + '/console.json', "w", encoding="utf-8") as f: # 存出prompt
        json.dump(dict_prompt, f, ensure_ascii=False, indent=4)
    
    # 本次使用的VLM模型
    for item in Path(save_dir).rglob('*.md'):
        os.remove(str(item))
    with open(save_dir + '/' + model_type.replace(':', '_').replace('/', '_') + '.md', 'w', encoding="utf-8"):
        pass

    #%%
    all_files_dir = []
    for item in Path(root).rglob('*'): # 撈出路徑內所有檔案和資料夾
        if item.is_file():
            full_dir = str(item)
            full_dir_ext = full_dir.split('.')[-1]
            # if full_dir.split(str(Path(root)))[1].split('\\')[1][-1] != 'X':
            if full_dir.split(str(Path(root)))[1].split('\\')[1][-1] != 'X' and full_dir.split(str(Path(root)))[1].split('\\')[1] == folder_name:
            # if full_dir.split(str(Path(root)))[1].split('\\')[1][-1] != 'X' and full_dir.split(str(Path(root)))[1].split('\\')[1] == 'B01施工記錄表':
            # if full_dir.split(str(Path(root)))[1].split('\\')[1][-1] != 'X' and full_dir.split(str(Path(root)))[1].split('\\')[1] == 'B02施工前後及過程照片(監工)':
            # if full_dir.split(str(Path(root)))[1].split('\\')[1][-1] != 'X' and full_dir.split(str(Path(root)))[1].split('\\')[1] == 'C01工程承攬切結書 +日期擷取':
            # if full_dir.split(str(Path(root)))[1].split('\\')[1][-1] != 'X' and full_dir.split(str(Path(root)))[1].split('\\')[1] == 'C02施工作業安全告知單(廠商適用) +日期擷取':
                if full_dir_ext != 'db':
                    all_files_dir.append(full_dir)
        elif item.is_dir():
            pass
    
    #%%
    final_full_folder = []
    final_full_dir = []
    final_full_dir_page = []
    final_full_dir_ext = []
    final_full_process_run_time = []
    final_full_title_run_time = []
    final_full_datetime_run_time = []
    final_matched_title = []
    final_matched_time = []
    final_detect_title = []
    final_detect_time = []
    
    # total_file = 0
    # target_file = 0
    
    # 顯示在圖片中的字體
    font = ImageFont.truetype('./TaipeiSansTCBeta-Regular.ttf', 120) # font
    font2 = ImageFont.truetype('./TaipeiSansTCBeta-Regular.ttf', 90) # font
    
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
    
    for i in tqdm(range(0, len(all_files_dir))):
    # for full_dir in tqdm(all_files_dir[703:]):
    # for full_dir in tqdm(['C:\\Users\\aiuser\\Desktop\\lai\\detect_pdf_words\\完工驗收資料(Sample)_0619\\C04工作安全分析JSA記錄 +日期擷取\\19-2 AAAA1Z08 JSA.pdf']): # 除錯用
        full_dir = all_files_dir[i]
        filename = full_dir.split('\\')[-1]
        filename_ext = filename.split('.')[-1].lower()
        folder = full_dir.split(filename)[0].split(str(Path(root)))[1].replace('\\', '/') # 檔案完整路徑 去掉 root filename
        os.makedirs(save_dir + '/' + folder, exist_ok=True)
        folder_temp = folder.split('/')[1]
        
        # 檔案的時間戳記
        ctime = os.path.getctime(full_dir)                 # 檔案建立時間
        ctime_string = dt.fromtimestamp(int(ctime))
        mtime = os.path.getmtime(full_dir)                 # 檔案修改時間
        mtime_string = dt.fromtimestamp(int(mtime))
        
        if ctime_string < mtime_string:
            last_file_time = ctime_string.strftime("%Y年%m月%d日")
        elif ctime_string >= mtime_string:
            last_file_time = mtime_string.strftime("%Y年%m月%d日")
        
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
                detect_title = remove_duplicate_chars(detect_title) # 糾正疊字
                # temp2 = []
                # for item in last_answer_title_list: # 找出output_text_convert中存在answer_title_list的連續子字串
                #     if item in detect_title:
                #         temp2.append(item)
                #         break
                # 日期時間
                detect_date = ''
                
                if detect_title in last_answer_title_list:
                    final_matched_title.append('OK')
                    final_detect_title.append(detect_title)
                else:
                    
                    final_matched_title.append('NG')
                    final_detect_title.append(detect_title)
                final_full_folder.append(folder)
                final_full_dir.append(full_dir)
                final_full_dir_page.append(nun_page)
                final_full_dir_ext.append(filename_ext)
                final_full_process_run_time.append('NAN')
                final_matched_time.append('NAN')
                final_full_title_run_time.append('NAN')
                final_detect_time.append(detect_date)
                final_full_datetime_run_time.append('NAN')
        else:
            vlm_runable = False
            try:
                os.makedirs(save_dir + '/none_pdf_or_img_docx/', exist_ok=True)
                shutil.copy(root + '/' + folder + '/' + filename,
                            save_dir + '/none_pdf_or_img_docx/' + filename)
            except:
                pass
            final_full_folder.append(folder)
            final_full_dir.append(full_dir)
            final_full_dir_page.append(0)
            final_full_dir_ext.append(filename_ext)
            final_full_process_run_time.append('NAN')
            final_matched_title.append('NAN')
            final_matched_time.append('NAN')
            final_full_title_run_time.append('NAN')
            final_detect_title.append('')
            final_detect_time.append('')
            final_full_datetime_run_time.append('NAN')
        
        if vlm_runable == True:
            for nun_page in range(len(images)):
                time_start = dt.now()
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
                # rotate_iamge_gray_enh = clahe.apply(rotate_iamge_gray)                           # 影像對比度強化
                
                y_min, y_max, x_min, x_max = rm_white_area_coordinate(rotate_iamge_gray)
                rotate_iamge_gray = Image.fromarray(rotate_iamge_gray)
                
                rotate_iamge_gray = rotate_iamge_gray.crop((x_min-50, y_min-200, x_max+50, y_max+200)) # 找出關鍵區域
                
                width, height = rotate_iamge_gray.size
                
                if folder_temp == 'C01工程承攬切結書 +日期擷取':
                    crop_area = (0, height//7, width//1.5, height//3)
                    croped = rotate_iamge_gray.crop(crop_area)                       # 局部的關鍵區域
                    roi_cp_img_dir = temp_image_folder + filename.split('.')[0] + '_' + str(nun_page) + '_cp.jpg'    # 輸入給VLM辨識「施工期限」的圖片路徑
                    croped.save(roi_cp_img_dir, 'JPEG')                                        # 儲存 待偵測的「施工期限」圖片
                elif folder_temp == 'C02施工作業安全告知單(廠商適用) +日期擷取':
                    crop_area = (width//1.3, height//2, width, height//1.2)
                    croped = rotate_iamge_gray.crop(crop_area)                       # 局部的關鍵區域
                    roi_snd_img_dir = temp_image_folder + filename.split('.')[0] + '_' + str(nun_page) + '_snd.jpg'  # 輸入給VLM辨識「安全告知日期」的圖片路徑
                    croped.save(roi_snd_img_dir, 'JPEG')                                       # 儲存 待偵測的「安全告知」圖片
                elif folder_temp == 'C04工作安全分析JSA記錄 +日期擷取':
                    crop_area = (0, height//6, width, height//2)
                    croped = rotate_iamge_gray.crop(crop_area)                       # 局部的關鍵區域
                    roi_td_img_dir = temp_image_folder + filename.split('.')[0] + '_' + str(nun_page) + '_td.jpg'    # 輸入給VLM辨識「訓練日期」的圖片路徑
                    croped.save(roi_td_img_dir, 'JPEG')                                        # 儲存 待偵測的「訓練日期」圖片
                time_end = dt.now()
                final_full_process_run_time.append(round((time_end-time_start).total_seconds(), 2))
                del time_end, time_start
                
                # 辨識標題
                time_start = dt.now()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(vlm_text_detection, model_type, roi_tilte_img_dir, prompt_get_標題)
                    try:
                        detect_title = future.result(timeout=timeout_seconds)
                        detect_title = text_processing(text=detect_title) # 整理成簡體改成繁體、全形改成半形、換行改成空格
                        # detect_title = re.sub(f"[{re.escape(all_punct)}]", "", detect_title)
                        detect_title = detect_title.replace('施工期限', '')
                        # print('\n', detect_title)
                    except TimeoutError:
                        detect_title = 'TIME_OUT'
                    except Exception as e:
                        pass
                detect_title = text_processing(text=detect_title) # 整理成簡體改成繁體、全形改成半形、換行改成空格
                for item in last_vlm_correction_dict: # 用寶典矯正已知錯字
                    if item in detect_title:
                        detect_title = detect_title.replace(item, last_vlm_correction_dict[item])
                detect_title = remove_duplicate_chars(detect_title) # 糾正疊字
                time_end = dt.now()
                final_full_title_run_time.append(str(round((time_end-time_start).total_seconds(), 2)))
                del time_end, time_start
                
                # 辨識日期時間
                if folder_temp == 'C01工程承攬切結書 +日期擷取':
                    time_start = dt.now()
                    stpe_word_list = None # 針對標題客製化的停止詞
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(vlm_text_detection,
                                                 model_type=model_type,
                                                 image_dir=roi_cp_img_dir,
                                                 prompt_get_target=prompt_get_施工期限.replace('PLACEHOLDER_TIME', last_file_time),
                                                 options=None,
                                                 stpe_word_list=['日曆天', '日历天'])
                        try:
                            detect_date = future.result(timeout=timeout_seconds)
                            detect_date = text_processing(text=detect_date) # 整理成簡體改成繁體、全形改成半形、換行改成空格
                            # detect_date = re.sub(f"[{re.escape(all_punct)}]", "", detect_date)
                            detect_date = detect_date.split('日曆天')[0] + '日曆天'
                            detect_date = detect_date.replace('施工期限', '')
                            # print('\n', detect_date)
                        except TimeoutError:
                            detect_date = 'TIME_OUT'
                        except Exception as e:
                            pass
                    time_end = dt.now()
                    final_full_datetime_run_time.append(str(round((time_end-time_start).total_seconds(), 2)))
                    del time_end, time_start
                elif folder_temp == 'C02施工作業安全告知單(廠商適用) +日期擷取':
                    time_start = dt.now()
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(vlm_text_detection,
                                                 model_type=model_type,
                                                 image_dir=roi_snd_img_dir,
                                                 prompt_get_target=prompt_get_安全告知日期.replace('PLACEHOLDER_TIME', last_file_time),
                                                 stpe_word_list=['日'])
                        try:
                            detect_date = future.result(timeout=timeout_seconds)
                            detect_date = text_processing(text=detect_date) # 整理成簡體改成繁體、全形改成半形、換行改成空格
                            # detect_date = re.sub(f"[{re.escape(all_punct)}]", "", detect_date)
                            detect_date = detect_date.replace('安全告知日期', '')
                            # print('\n', detect_date)
                        except TimeoutError:
                            detect_date = 'TIME_OUT'
                        except Exception as e:
                            pass
                    time_end = dt.now()
                    final_full_datetime_run_time.append(str(round((time_end-time_start).total_seconds(), 2)))
                    del time_end, time_start
                elif folder_temp == 'C04工作安全分析JSA記錄 +日期擷取':
                    time_start = dt.now()
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(vlm_text_detection,
                                                 model_type=model_type,
                                                 image_dir=roi_snd_img_dir,
                                                 prompt_get_target=prompt_get_訓練日期.replace('PLACEHOLDER_TIME', last_file_time),
                                                 options=None,
                                                 stpe_word_list=['日'])
                        try:
                            detect_date = future.result(timeout=timeout_seconds)
                            detect_date = text_processing(text=detect_date) # 整理成簡體改成繁體、全形改成半形、換行改成空格
                            # detect_date = re.sub(f"[{re.escape(all_punct)}]", "", detect_date)
                            detect_date = detect_date.replace('訓練日期', '')
                            # print('\n', detect_date)
                        except TimeoutError:
                            detect_date = 'TIME_OUT'
                        except Exception as e:
                            pass
                    time_end = dt.now()
                    final_full_datetime_run_time.append(str(round((time_end-time_start).total_seconds(), 2)))
                    del time_end, time_start
                else:
                    manual_data = '經過人工判讀畫面中沒有要檢測的日期時間'
                    final_full_datetime_run_time.append('NAN')
                
                width, height = rotate_iamge.size
                
                new_width = width + right + left
                new_height = height + top + bottom
                
                result = Image.new(rotate_iamge.mode, (new_width, new_height), (255, 255, 255))
                result.paste(rotate_iamge, (left, top))
                
                # 對標題的答案
                # temp2 = []
                # for item in last_answer_title_list: # 找出output_text_convert中存在answer_title_list的連續子字串
                #     if item in detect_title:
                #         temp2.append(item)
                #         break
                try:
                    temp3 = []
                    detect_date = CONVERT_DATETIME_FORMAT(detect_date).year() # 統一日期格式
                    detect_date = CONVERT_DATETIME_FORMAT(detect_date).month_day() # 統一日期格式
                    detect_date_rm_symbol = detect_date.replace('。', '').replace('，', '').replace(':', '').replace('?', '') # 刪除標點符號
                    del detect_date # 刪除用不到的變數
                    
                    # 對日期時間的答案
                    if folder_temp in last_answer_time_df_group_name:
                        df_datetime = last_answer_datetime_df.loc[last_answer_time_df_group[folder_temp]]
                        df_datetime_index = df_datetime.groupby('檔名').groups[filename][nun_page]
                        item = df_datetime['日期時間(實際)'][df_datetime_index]
                        if type(item) == dt:
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
                if detect_title in last_answer_title_list:
                    final_matched_title.append('OK')
                    final_detect_title.append(detect_title)
                    draw.text(xy=(50, 50), text=str(detect_title),
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
                
                final_full_folder.append(folder)
                final_full_dir.append(full_dir)
                final_full_dir_page.append(nun_page)
                final_full_dir_ext.append(filename_ext)
        else:
            pass
    
    if len(final_full_folder) != 0:
        df = pd.DataFrame(
            {
                "資料夾名稱": final_full_folder,
                "檔案路徑": final_full_dir,
                "頁碼": final_full_dir_page,
                "附檔名": final_full_dir_ext,
                "切ROI時間(/sec)": final_full_process_run_time,
                "標題命中與否": final_matched_title,
                "標題偵測結果": final_detect_title,
                "標題偵測時間(/sec)": final_full_title_run_time,
                "日期命中與否": final_matched_time,
                "日期偵測結果": final_detect_time,
                "日期偵測時間(/sec)": final_full_datetime_run_time
            }
        )
        # 用excel紀錄辨識狀況
        writer = pd.ExcelWriter(save_dir + '/' + folder_name + '.xlsx', engine='xlsxwriter') # 建立xlsx文件
        df.to_excel(writer, sheet_name='Sheet1', index=False) # 寫入資料
        worksheet = writer.sheets['Sheet1']
        for col_idx, col in enumerate(df.columns):
            if col != '檔案路徑':
                column_len = max(df[col].astype(str).map(len).max(), len(col))*1.8 # 找出每個欄位最長的字串長度
                worksheet.set_column(col_idx, col_idx, column_len + 3)
            else:
                pass
        writer.close() # 關閉檔案
    else:
        pass
