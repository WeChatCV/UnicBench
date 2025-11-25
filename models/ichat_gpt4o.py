# -*- coding: utf-8 -*-

import argparse
import requests
import time
import json
import hmac
import hashlib
import base64
from typing import Union, Optional, Tuple, List
from io import BytesIO

def encode_pil_image(pil_image):
    # Create an in-memory binary stream
    image_stream = BytesIO()
    
    # Save the PIL image to the binary stream in JPEG format (you can change the format if needed)
    pil_image.save(image_stream, format='JPEG')
    
    # Get the binary data from the stream and encode it as base64
    image_data = image_stream.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    return base64_image

class Ichat_GPT4o:
    def __init__(self, model="gpt-4o"):
        # 鉴权参数
        self.app_id = "app2fmp3ihlc5e1irlr"
        self.app_key = "EZmlffmLrtdVxoWoFjmsqQMODJjpuwdC"
        self.source = "mmbasevision_gpt4v"
        self.model = model
        self.use_encode = False # 该参数暂时针对GEdit_Bench

    def forward(self, final_prompt, max_retries=10):
        for retry in range(max_retries):
            response = self.get_parsed_output(final_prompt)
            if response is not None:
                return response
            else:
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2  # 指数退避：2秒, 4秒, 6秒...
                    print(f"Failed to get response (attempt {retry + 1}/{max_retries})")
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)


    def fetch_data(self, image, prompt_template):
        auth, timestamp = self.calcAuthorization(self.source, self.app_key)
        headers = {
            "X-AppID": self.app_id,
            "X-Source": self.source,
            "X-Timestamp": str(timestamp),
            "X-Authorization": auth,
        }
        if image is None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_template},
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_template},
                    ] + [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_image(img_base64)}"}} for img_base64 in image
                    ]
                }
            ]

        data = {
            "model": self.model,
            "temperature": 0.1,
            "messages": messages,
            "max_tokens": 2048,
        }
        url = "http://ichat.woa.com/api/chat_completions"
        r_content = ""
        try:
            r = requests.post(url=url, headers=headers, data=json.dumps(data))
            r_content = r.content
            res_json = r.json()
        except Exception as e:
            print(f"Error Occur, content: {r_content}, exception: {str(e)}")
            res_json = None

        msg = res_json.get('msg', None)
        if msg is None or msg != 'success':
            print(f"Error Occur, msg: {msg}")
            return None
        return res_json['response']

    def calcAuthorization(self, source, appkey):
        timestamp = int(time.time())
        signStr = "x-timestamp: %s\nx-source: %s" % (timestamp, source)
        sign = hmac.new(
            appkey.encode("utf-8"), signStr.encode("utf-8"), hashlib.sha256
        ).digest()
        return sign.hex(), timestamp
    
    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if image_links is None:
            print("img_base64_list is None")
            return None
        else:
            if not isinstance(image_links, list):
                image_links = [image_links]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                    ] + [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_image(img_base64)}"}} for img_base64 in image_links
                    ]
                }
            ]
            # messages = [
            #     {
            #         "role": "user",
            #         "content": [
            #             {"type": "text", "text": text_prompt},
            #         ] + [
            #             {"type": "image_url", "image_url": {"url": img_base64}} for img_base64 in image_links
            #         ]
            #     }
            # ]
            return messages
    
    def get_parsed_output(self, prompt):
        auth, timestamp = self.calcAuthorization(self.source, self.app_key)
        headers = {
            "X-AppID": self.app_id,
            "X-Source": self.source,
            "X-Timestamp": str(timestamp),
            "X-Authorization": auth,
        }

        data = {
            "model": self.model,
            "temperature": 0.1,
            "messages": prompt,
            "max_tokens": 2048,
        }
        url = "http://ichat.woa.com/api/chat_completions"
        r_content = ""
        try:
            r = requests.post(url=url, headers=headers, data=json.dumps(data))
            r_content = r.content
            res_json = r.json()
        except Exception as e:
            print(f"Error Occur, content: {r_content}, exception: {str(e)}")
            res_json = None

        msg = res_json.get('msg', None)
        if msg is None or msg != 'success':
            print(f"Error Occur, msg: {msg}")
            return None
        return res_json['response']

HONGBAO_TEMPLATE = """1. 图片描绘了一群穿着卡通服装的角色围绕在一个坐在金元宝上的小孩周围，背景是橙色的，带有庆祝氛围的装饰，如灯笼和红包。整体风格是可爱卡通风，色彩鲜艳。“安信资管”位于图片顶部中央，字体为白色，现代无衬线风格，大小适中。下面是较小的英文“ESSENCE ASSET MANAGEMENT”，也是白色无衬线字体。中间部分有一个红色横幅，上面写着“业绩长虹节节高”，字体为白色，传统手写风格，居中排版。底部左右分别有两个红色立式横幅，左边写着“龙腾虎跃”，右边写着“再创辉煌”，字体为白色，手写风格。
2. 图片展示了三个戴着可爱牛角帽子的卡通小孩围绕着一张桌子，桌子上有一个显示器和一些编程相关的元素。背景是渐变的紫色，上面有一些悬挂的装饰物，整体风格活泼可爱，颜色鲜艳。图中最大的文字是“少儿编程”，位于图片中央的显示器上，字体较大，颜色为红色，字体类型为无衬线体。显示器上方有“WOTOJO 挖土机®”的文字，字体较小，颜色为白色，位置居中偏上。显示器下方有“Python”和“C++”的文字，分别在显示器的右侧和左侧，字体中等大小，颜色分别为绿色和橙色，字体类型为无衬线体。左下角有“<>”的符号，字体较小，颜色为蓝色。背景上方有“WOTOJO 挖土机®”和“WWW.WOTOJO.COM”的文字，字体较小，颜色为白色，字体类型为无衬线体，位置居中偏上。
3. 这是一幅具有现代风格的商业宣传图片，背景为红色，中央有一个带有牛角装饰的圆形舞台，上面有品牌标志和名称。舞台周围点缀着金色的元宝，整体色调为红色和金色，充满节日气氛。图片中有三处文字。最上方中间有一行文字“生而无畏·爱而自由”，字体为无衬线体，白色，字母大小适中，位于一个红色的横幅内。两侧对称地各有一个“福”字，字体为无衬线体，白色，字母大小较小，位于金色的装饰下方。舞台正中央有品牌名称“PANDORA”，字体为无衬线体，金色，字母较大，位于品牌标志下方。
4. 这是一张以红色为背景的图片，背景上有一些浅色的中国传统云纹装饰图案，整体风格具有浓厚的中国传统文化元素。图片中间竖排的文字是“暴富发财”，字体较大，采用了黑色的仿宋体，位于图片中央。左上角有一行较小的文字“@一个正经的程序员”，字体为白色，位于图片的左上角，风格为现代常规字体。
5. 图片中有两只卡通企鹅，一只蓝色，一只粉色，背景是渐变的紫色到橙色，地上有一个金元宝，背景有聚光灯效果。图片上方有大号红色手写风格的“2021”字样，位于图片的上部中央，字体较大，颜色为红色，风格为手写体。金元宝上有较小的黄色“2021”字样，位于图片的中央靠下部分，字体较小，颜色为黄色，风格为圆润的无衬线体。
6. 这是一幅以红色为主色调的新年主题插画，人物头戴带有虎纹的面具，背景有鞭炮和灯笼装饰，整体风格充满节日气氛。“虎年大吉”文字位于图片上方，字体为圆形印章风格，颜色为黑色，背景为橙色圆圈，排列成一行。面具上的“发”字位于人物面具的正中间，字体为手写风格，颜色为金色。红包上的“发”字位于红包的左上角，字体为手写风格，颜色为黑色。
我们有一个文本到图像的生成模型，以上是一些训练数据中的文本。我们现在看到用户输入的文本与训练数据的分布不相似，导致模型表现变差。我希望你帮我重写用户的输入文本来提升模型表现：(a) 请将用户输入的文本改写成先描述图片整体风格，然后描述图像的背景内容，再描述图像的前景内容。最后依次描述图像中的文字，要包含文字的位置、大小、字体、风格和颜色。上面是一些样例。用户输入文本中的引号内容请不要拆分。(b) 我们的模型能够识别中文双引号“”中的文字，在图像上生成“”中的文字。用户有时候会不写双引号或者用英文双引号，我希望你能帮我把用户要写的字用中文双引号括上或者把英文双引号换成中文双引号。请用下面的模板格式作为输出：{"Rewrite_caption":""}。用户的输入为：
"""


def demo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt4.1")
    args = parser.parse_args()

    text = "一只小狗在跑，背景歇着“射雕英雄传”"
    bp1 = time.time()
    ichat = Ichat_GPT4o(model=args.model)

    input_text = HONGBAO_TEMPLATE + text

    res = ichat.fetch_data(input_text)
    print(res)
    # msg = res_json['msg']
    # if msg != "success":
    #     print('error')
    #     rewrite_caption = text
    # else:
    #     try:
    #         response = res_json['response']
        
    #         data = json.loads(response)
    #         rewrite_caption = data['Rewrite_caption']
    #     except Exception as e:
    #         print(f'decode error: {e}')
    #         rewrite_caption  = text

    # bp2 = time.time()
    # print(f"res: {rewrite_caption}, time_diff: {bp2 - bp1}")

if __name__ == "__main__":
    demo()
