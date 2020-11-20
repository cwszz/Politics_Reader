# import docx
import json
# from pdfminer.pdfparser import PDFParser, PDFDocument
# from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
# from pdfminer.converter import PDFPageAggregator
# from pdfminer.layout import LTTextBoxHorizontal, LAParams
# from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
import os
import re
import pdfplumber


# 读取word，写入json
# def read_word():
#     # 打开文档
#     file = docx.Document("E:\\Putian\\Dataset\\人才政策待遇.docx")
#     # 创建字典
#     word_dict = {'title': file.paragraphs[0].text}
#     # index = 0
#     # for i in range(len(file.paragraphs)):
#     #     if (i != 0) & (file.paragraphs[i].text != ""):
#     #         word_dict["paragraph"+str(index)] = file.paragraphs[i].text
#     #     if file.paragraphs[i].text != "":
#     #         index += 1
#     content = ""
#     symbol = ('。', '：', '；')
#     for i in range(len(file.paragraphs)):
#         if (i != 0) & (file.paragraphs[i].text != ""):
#             if file.paragraphs[i].text.endswith(symbol) is not True:
#                 file.paragraphs[i].text += "。"
#             content += file.paragraphs[i].text
#     word_dict["article"] = content
#     for para in file.paragraphs:
#         print(para.text)
#     return word_dict


def find_all_file(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.pdf'):
                fullname = os.path.join(root, f)
                yield fullname


# def read_pdf_miner():
#     """
#     解析PDF文本，并保存到TXT文件中
#     :return: 读取的pdf文本
#     """
#     base = 'E:\\Putian\\Dataset\\'
#     for file_path in find_all_file(base):
#         print(file_path)
#     # file_path = r'E:\\Dataset\\数据样本\\【9】关于支持“双招双引”十条意见.pdf'
#         fp = open(file_path, 'rb')
#         # 用文件对象创建一个PDF文档分析器
#         parser = PDFParser(fp)
#         # 创建一个PDF文档
#         doc = PDFDocument()
#         # 连接分析器，与文档对象
#         parser.set_document(doc)
#         doc.set_parser(parser)
#
#         # 提供初始化密码，如果没有密码，就创建一个空的字符串
#         doc.initialize()
#
#         # 检测文档是否提供txt转换，不提供就忽略
#         if not doc.is_extractable:
#             raise PDFTextExtractionNotAllowed
#         else:
#             # 创建PDF，资源管理器，来共享资源
#             rsrcmgr = PDFResourceManager()
#             # 创建一个PDF设备对象
#             laparams = LAParams()
#             device = PDFPageAggregator(rsrcmgr, laparams=laparams)
#             # 创建一个PDF解释其对象
#             interpreter = PDFPageInterpreter(rsrcmgr, device)
#
#             # 循环遍历列表，每次处理一个page内容
#             # doc.get_pages() 获取page列表
#             for page in doc.get_pages():
#                 interpreter.process_page(page)
#                 # 接受该页面的LTPage对象
#                 layout = device.get_result()
#                 # 这里layout是一个LTPage对象 里面存放着 这个page解析出的各种对象
#                 # 一般包括LTTextBox, LTFigure, LTImage, LTTextBoxHorizontal 等等
#                 # 想要获取文本就获得对象的text属性，
#                 for x in layout:
#                     if isinstance(x, LTTextBoxHorizontal):
#                         with open(r'data_512.txt', 'a') as f:
#                             results = x.get_text()
#                             print(results)
#                             f.write(results)  # + "\n"


def read_pdf_text():
    base = 'E:\\Putian\\Dataset\\'
    file_path = 'E:\\Putian\\Dataset\\附件3福建省级高层次人才认定和支持办法实施细则（试行）.pdf'
    # for file_path in find_all_file(base):
    with pdfplumber.open(file_path) as pdf:
        content = ''
        for i in range(len(pdf.pages)):
            page = pdf.pages[i]
            # page_content = '\n'.join(page.extract_text().split('\n')[:-1])
            print(page.extract_text())
            page_content = ''.join(page.extract_text().split('\n')[:-1])
            content = content + page_content
        print(content)
        content_list = content.split('。')
        print(len(content_list))


def read_pdf_table():
    path = 'E:\\Putian\\Dataset\\附件1福建省级高层次人才认定条件（2020年版）.pdf'
    pdf = pdfplumber.open(path)

    for page in pdf.pages:
        print(page.extract_text())
        for pdf_table in page.extract_tables():
            table = []
            cells = []
            for row in pdf_table:
                if not any(row):
                    # 如果一行全为空，则视为一条记录结束
                    if any(cells):
                        table.append(cells)
                        cells = []
                elif all(row):
                    # 如果一行全不为空，则本条为新行，上一条结束
                    if any(cells):
                        table.append(cells)
                        cells = []
                    table.append(row)
                else:
                    if len(cells) == 0:
                        cells = row
                    else:
                        for i in range(len(row)):
                            if row[i] is not None:
                                cells[i] = row[i] if cells[i] is None else cells[i] + row[i]
            for row in table:
                print([re.sub('\s+', '', cell) if cell is not None else None for cell in row])
            print('---------- 分割线 ----------')
    pdf.close()


def read_txt_write_json():
    base = 'E:\\Putian\\Dataset\\txt\\'
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.txt'):
                fullname = os.path.join(root, f)
                # yield fullname
                with open(fullname) as file:
                    word_dict = {'title': f.replace('.txt', '')}
                    content = ""
                    symbol = ('。', '：', '；')
                    for line in file.readlines():
                        line = line.strip().strip('\n')
                        if line != "":
                            if line.endswith(symbol) is not True:
                                line += "。"
                            content += line
                    word_dict["article"] = content
                    print(content)
                    write_json(word_dict)


def write_json(word_dict):
    with open("./Data/dataset.jsonl", "a+", encoding='utf-8') as f:
        json.dump(word_dict, f, ensure_ascii=False)
        f.write('\n')
        f.close()
    print("写入文件完成...")


if __name__ == '__main__':
    # write_json(read_word())
    read_pdf_text()
    # read_txt_write_json()
