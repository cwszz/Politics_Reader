# 本文件专门处理直接从pdf读取的上下文
import pdfplumber
import os
import json

chinese_number_list_level3 = []
for chinese_number in range(1, 100):
    obj_1 = str(chinese_number) + "．"
    obj_2 = str(chinese_number) + "."
    chinese_number_list_level3.append(obj_1)
    chinese_number_list_level3.append(obj_2)
print(chinese_number_list_level3)
chinese_number_list_level1 = ["一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、", "十、"]
chinese_number_list_level2 = ["（一）", "（二）", "（三）", "（四）", "（五）", "（六）", "（七）", "（八）", "（九）", "（十）"]
# chinese_number_list_level3 = ['1．', '2．', '3．', '4．', '5．', '6．', '7．', '8．', '9．', '10．']
chinese_number_list_level4 = ["（1）", "（2）", "（3）", "（4）", "（5）", "（6）", "（7）", "（8）", "（9）", "（10）"]


def find_all_file(base, end=".pdf"):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith(end):
                fullname = os.path.join(root, f)
                yield fullname


def read_pdf_save_text():
    base = 'E:\\Putian\\Dataset\\分文档后的所有文档\\'
    # file_path = 'E:\\Putian\\Dataset\\分文档后的所有文档\\附件3福建省级高层次人才认定和支持办法实施细则（试行）.pdf'
    for file_path in find_all_file(base):
        with pdfplumber.open(file_path) as pdf:
            content = ''
            for i in range(len(pdf.pages)):
                page = pdf.pages[i]
                # page_content = '\n'.join(page.extract_text().split('\n')[:-1])
                print(page.extract_text())
                page_content = ''.join(page.extract_text().split('\n')[:-1])
                content = content + page_content
            print(content)
            # content_list = content.split('。')
            # print(len(content_list))
        save_path = "E:\\Putian\\Dataset\\分文档后的所有文档\\txt\\" + str(os.path.basename(file_path).split('.')[0])+'.txt'
        with open(save_path, "w") as f:
            f.write(content)


def read_txt(file_path):
    # read_path = "E:\\Putian\\Dataset\\分文档后的所有文档\\txt\\"
    # file_path = read_path + "附件3福建省级高层次人才认定和支持办法实施细则（试行）.txt"
    with open(file_path, "r") as f:
        content = f.read()
    return content, os.path.basename(file_path).split('.')[0]


def save_dict():
    full_dict = {"data": []}
    document_dict = {"document": []}
    # base = 'E:\\Putian\\Dataset\\分文档后的所有文档\\txt\\'

    # for file_path in find_all_file(base, end='.txt'):
    file_path = "E:\\Putian\\Dataset\\分文档后的所有文档\\txt\\20170331+关于实施人才“壶兰计划”的意见（莆委发[2017]2号）.txt"
    full_content, title = read_txt(file_path)
    full_context = {"title": title, "introduction": "", "context": []}
    hierarchical_dispose(full_content, full_context)
    document_dict["document"].append(full_context)
    full_dict["data"].append(document_dict)

    with open("E:\\Putian\\Dataset\\hierarchical_dataset.json", "w") as f:
        f.write(json.dumps(full_dict, indent=4, ensure_ascii=False) + '\n')
        print("加载入文件完成...")
    print("字典保存成功！")


def hierarchical_dispose(full_content, full_context):
    # 是否有一级标题
    index_list_level1 = judge_level1(full_content)

    # 此时存在一级标题
    if len(index_list_level1) > 1:
        level1_content = dispose_level1(index_list_level1, full_content, full_context)
        # 判断每个一级标题下的内容是否有二级标题
        for index_level1, content_per_level1 in enumerate(level1_content):
            index_list_level2 = judge_level2(content_per_level1)

            # 如果有二级标题
            if len(index_list_level2) > 1:
                # 并且二级标题前不存在三级标题，才能表示有二级标题
                if len(judge_level3(content_per_level1[0:index_list_level2[1]])) <= 1:
                    level = 2
                    level2_content = dispose_level2(index_list_level2, content_per_level1, full_context, level)
                    for index_level2, content_per_level2 in enumerate(level2_content):
                        # 判断每个二级标题下的内容是否有三级标题
                        index_list_level3 = judge_level3(content_per_level2)

                        # 如果二级标题下有三级标题
                        if len(index_list_level3) > 1:
                            level = 3
                            level3_content = dispose_level3(
                                index_list_level3, content_per_level2, full_context, level, index_level1)
                            for index_level3, content_per_level3 in enumerate(level3_content):
                                # 在每个二级标题下的内容包含三级标题的情况下判断是否有四级标题
                                index_list_level4 = judge_level4(content_per_level3)

                                # 如果三级标题下有四级标题
                                if len(index_list_level4) > 1:
                                    level = 4
                                    level4_content = dispose_level4(
                                        index_list_level4, content_per_level3, full_context,
                                        level, index_level1, index_level2)
                                    # 由于只有4级标题，因此不再向下级标题判断；
                                    # 直接将当前一个三级标题对应的内容中所包含的全部四级标题对应的内容放到字典中
                                    put_context(level4_content, full_context, level,
                                                index_level1, index_level2, index_level3)

                                # 如果三级标题下没有四级标题
                                else:
                                    level = 3
                                    # 将当前一个三级标题对应的内容放到字典中
                                    put_paragraph(content_per_level3, full_context, level,
                                                  index_level1, index_level2)

                        # 在每个二级标题下的内容不包含三级标题的情况下 包含有四级标题
                        elif len(judge_level4(content_per_level2)) > 1:
                            index_list_level4 = judge_level4(content_per_level2)
                            level = 3  # 此时四级标题其实是三级标题
                            level4_content = dispose_level4(
                                index_list_level4, content_per_level2, full_context, level, index_level1)
                            # 由于只有4级标题，因此不再向下级标题判断；
                            # 直接将当前一个二级标题对应的内容中所包含的全部四级标题对应的内容放到字典中
                            put_context(level4_content, full_context, level, index_level1, index_level2)

                        # 在每个二级标题下的内容不包含三级标题的情况下 也不含有四级标题
                        else:
                            level = 2
                            # 将当前一个二级标题对应的内容放到字典中
                            put_paragraph(content_per_level2, full_context, level, index_level1)

            # 如果该一级标题下无二级标题，判断每个一级标题下的内容是否有三级标题
            # 有三级标题
            if len(judge_level3(content_per_level1)) > 1:
                index_list_level3 = judge_level3(content_per_level1)
                level = 2
                level3_content = dispose_level3(
                    index_list_level3, content_per_level1, full_context, level, index_level1)
                for index_level3, content_per_level3 in enumerate(level3_content):

                    # 判断每个三级标题下的内容是否有四级标题
                    index_list_level4 = judge_level4(content_per_level3)

                    # 有四级标题
                    if len(index_list_level4) > 1:
                        level = 3
                        level4_content = dispose_level4(
                            index_list_level4, content_per_level3, full_context, level, index_level1)
                        # 由于只有4级标题，因此不再向下级标题判断；
                        # 直接将当前一个三级标题对应的内容中所包含的全部四级标题对应的内容放到字典中
                        put_context(level4_content, full_context, level, index_level1, index_level3)
                    # 没有四级标题
                    else:
                        level = 2
                        # 将当前一个三级标题对应的内容放到字典中
                        put_paragraph(content_per_level3, full_context, level, index_level1)

            # 如果该一级标题下无二级和三级标题，则判断是否有四级标题
            # 有四级标题
            elif len(judge_level4(content_per_level1)) > 1:
                index_list_level4 = judge_level4(content_per_level1)
                level = 2
                level4_content = dispose_level4(index_list_level4, content_per_level1, full_context, level)
                # 由于只有4级标题，因此不再向下级标题判断；
                # 直接将当前一个一级标题对应的内容中所包含的全部四级标题对应的内容放到字典中
                put_context(level4_content, full_context, level, index_level1)
            # 没有四级标题
            else:
                level = 1
                # 将当前一个一级标题对应的内容放到字典中
                put_paragraph(content_per_level1, full_context, level, index_level1)

    # 在原文内容不含一级标题的情况下判断原文内容是否有二级标题
    elif len(judge_level2(full_content)) > 1:
        index_list_level2 = judge_level2(full_content)
        level = 1
        level2_content = dispose_level2(index_list_level2, full_content, full_context, level)
        for index_level2, content_per_level2 in enumerate(level2_content):
            # 判断每个二级标题下的内容是否有三级标题
            index_list_level3 = judge_level3(content_per_level2)

            # 如果二级标题下有三级标题
            if len(index_list_level3) > 1:
                level = 2
                level3_content = dispose_level3(
                    index_list_level3, content_per_level2, full_context, level)
                for index_level3, content_per_level3 in enumerate(level3_content):
                    # 在每个二级标题下的内容包含三级标题的情况下判断是否有四级标题
                    index_list_level4 = judge_level4(content_per_level3)

                    # 如果三级标题下有四级标题
                    if len(index_list_level4) > 1:
                        level = 3
                        level4_content = dispose_level4(
                            index_list_level4, content_per_level3, full_context, level, index_level2)
                        # 由于只有4级标题，因此不再向下级标题判断；
                        # 直接将当前一个三级标题对应的内容中所包含的全部四级标题对应的内容放到字典中
                        put_context(level4_content, full_context, level, index_level2, index_level3)

                    # 如果三级标题下没有四级标题
                    else:
                        level = 2
                        # 将当前一个三级标题对应的内容放到字典中
                        put_paragraph(content_per_level3, full_context, level, index_level2)

            # 在每个二级标题下的内容不包含三级标题的情况下 包含有四级标题
            elif len(judge_level4(content_per_level2)) > 1:
                index_list_level4 = judge_level4(content_per_level2)
                level = 2  # 此时四级标题其实是二级标题
                level4_content = dispose_level4(index_list_level4, content_per_level2, full_context, level)
                # 由于只有4级标题，因此不再向下级标题判断；
                # 直接将当前一个二级标题对应的内容中所包含的全部四级标题对应的内容放到字典中
                put_context(level4_content, full_context, level, index_level2)

            # 在每个二级标题下的内容不包含三级标题的情况下 也不含有四级标题
            else:
                level = 1
                # 将当前一个二级标题对应的内容放到字典中
                put_paragraph(content_per_level2, full_context, level)

    # 在原文内容不含一级和二级标题的情况下判断原文内容是否有三级标题
    elif len(judge_level3(full_content)) > 1:
        index_list_level3 = judge_level3(full_content)
        level = 1
        level3_content = dispose_level3(index_list_level3, full_content, full_context, level)
        for index_level3, content_per_level3 in enumerate(level3_content):
            # 包含三级标题的情况下判断是否有四级标题
            index_list_level4 = judge_level4(content_per_level3)

            # 如果三级标题下有四级标题
            if len(index_list_level4) > 1:
                level = 2
                level4_content = dispose_level4(index_list_level4, content_per_level3, full_context, level)
                # 由于只有4级标题，因此不再向下级标题判断；
                # 直接将当前一个三级标题对应的内容中所包含的全部四级标题对应的内容放到字典中
                put_context(level4_content, full_context, level, index_level3)

            # 如果三级标题下没有四级标题
            else:
                level = 1
                # 将当前一个三级标题对应的内容放到字典中
                put_paragraph(content_per_level3, full_context, level)

    # 在原文内容不含一级、二级和三级标题的情况下判断原文内容是否有四级标题
    # 如果原上下文有四级标题
    elif len(judge_level4(full_content)) > 1:
        index_list_level4 = judge_level4(full_content)
        level = 1
        level4_content = dispose_level4(index_list_level4, full_context, full_context, level)
        # 由于只有4级标题，因此不再向下级标题判断；
        # 直接将原上下文所包含的全部四级标题对应的内容放到字典中
        put_context(level4_content, full_context, level)
    print("当前文档处理完成！")


def judge_level1(content):
    """
    # 判断原文件中是否含有一级标题
    :param content: 原文件组成的上下文
    :return:返回一级标题所在上下文中的位置
    """
    index_list_level1 = [0]
    for headline_level1 in chinese_number_list_level1:
        # 上下文中存在多个相同一级标题时，如“一、”，则find返回第一个所在的索引，这也符合标题的定位
        index = content.find(headline_level1)
        # 一级标题存在上下文中，但后续序号在上下文中的位置必须在前继的后面
        if (index != -1) & (index > index_list_level1[-1]):
            index_list_level1.append(index)
        elif index == -1:
            break
    # print(len(index_list_level1))
    return index_list_level1


def dispose_level1(index_list_level1, full_content, full_context):
    """
    当存在一级标题时，则对上下文进行处理：构造导语字典元素以及由各个一级标题对应的内容组成的列表
    :param index_list_level1:一级标题在上下文中的索引
    :param full_content:原上下文
    :param full_context:字典
    :return:各个一级标题对应的内容组成的列表
    """
    # 处理导语部分
    introduction = full_content[0:index_list_level1[1]]
    full_context["introduction"] = introduction

    level1 = []  # 一级标题对应的内容
    # 处理正文
    for number, index in enumerate(index_list_level1):
        if number == 0:
            continue
        # 如果不是最后一个
        if number != len(index_list_level1) - 1:
            level1.append(full_content[index:index_list_level1[number + 1]])
        else:
            level1.append(full_content[index:])
    print("一级标题处理完成！")
    return level1


def judge_level2(content):
    """
    # 判断原文件中是否含有二级标题
    :param content: 一个级别标题对应的上下文或者原文组成的上下文
    :return:返回二级标题所在上下文中的位置
    """
    index_list_level2 = [0]
    for headline_level2 in chinese_number_list_level2:
        index = content.find(headline_level2)
        if (index != -1) & (index > index_list_level2[-1]):
            index_list_level2.append(index)
        elif index == -1:
             break
    # print(len(index_list_level2))
    return index_list_level2


def dispose_level2(index_list_level2, content, full_context, level):
    """
    当存在二级标题时：
    若一级标题也存在，对一个一级标题对应的上下文：添加嵌套字典以及构造由各个二级标题对应的内容组成的列表
    若一级标题不存在，对原上下文进行处理：构造导语字典元素以及由各个二级标题对应的内容组成的列表
    :param index_list_level2:二级标题在一级标题对应的上下文中的索引
    :param content:一个级别标题对应的上下文或者原上下文
    :param full_context:字典
    :param level:当前级别标题在原文中的级别，比如不存在一级标题时，则二级标题在上下文中标题级别就是1
    :return:各个二级标题对应的内容组成的列表
    """
    if level == 1:
        # 不存在一级标题，此时二级标题实际在原上下文中是一级标题
        # 构造导语字典元素
        introduction = content[0:index_list_level2[1]]
        full_context["introduction"] = introduction
    elif level == 2:
        # 存在一级标题，此时二级标题实际在原上下文中是二级标题
        context = {}
        title_level1 = content[0:index_list_level2[1]]
        context["title"] = title_level1
        context["context"] = []
        full_context["context"].append(context)

    # 构造由各个二级标题对应的内容组成的列表
    level2 = []  # 二级标题对应的内容
    # 处理正文
    for number_level2, index in enumerate(index_list_level2):
        if number_level2 == 0:
            continue
        # 如果不是最后一个
        if number_level2 != len(index_list_level2) - 1:
            level2.append(content[index:index_list_level2[number_level2 + 1]])
        else:
            level2.append(content[index:])
    print("二级标题处理完成！")
    return level2


def judge_level3(content):
    """
    # 判断原文件中是否含有三级标题
    :param content: 一个级别标题对应的上下文或者原文组成的上下文
    :return:返回三级标题所在上下文中的位置
    """
    index_list_level3 = [0]
    for headline_level3 in chinese_number_list_level3:
        index = content.find(headline_level3)
        if (index != -1) & (index > index_list_level3[-1]):
            index_list_level3.append(index)
        # elif index == -1:
        #     break
    # print(len(index_list_level3))
    return index_list_level3


def dispose_level3(index_list_level3, content, full_context, level, index_level1=0):
    """
    当存在三级标题时：
    若一级标题与二级标题都存在，此时level=3，则对一个二级标题对应的上下文进行处理：添加嵌套字典

    若一级标题存在，二级标题不存在，此时level=2，则对一个一级标题对应的上下文进行处理：添加嵌套字典
    若一级标题不存在，二级标题存在，此时level=2，则对一个二级标题对应的上下文进行处理：添加嵌套字典

    若一级标题和二级标题都不存在，此时level=1，则对原上下文进行处理：构造导语字典元素

    最后同一构造由各个三级标题对应的内容组成的列表
    :param index_list_level3:三级标题在一级标题对应的上下文中的索引
    :param content:一个级别标题对应的上下文或者原上下文
    :param full_context:字典
    :param level:当前级别标题在原文中的级别，比如不存在一级标题时，则二级标题在上下文中标题级别就是1
    :param index_level1=0：当level=3，此时对哪个一级标题对应的内容进行处理
    :return:各个三级标题对应的内容组成的列表
    """
    # 处理三级标题部分
    level3 = []  # 三级标题对应的内容
    context = {}
    if level == 1:
        # 此时不存在一级标题，则三级标题实际在原上下文中是一级标题
        # 处理导语部分
        introduction = content[0:index_list_level3[1]]
        full_context["introduction"] = introduction
    elif level == 2:
        # 此时三级标题实际在原上下文中是二级标题
        title_level2 = content[0:index_list_level3[1]]
        context["title"] = title_level2
        context["context"] = []
        full_context["context"].append(context)
    elif level == 3:
        # 此时三级标题实际在原上下文中是三级标题
        title_level2 = content[0:index_list_level3[1]]
        context["title"] = title_level2
        context["context"] = []
        full_context["context"][index_level1]["context"].append(context)
    # 处理正文
    for number_level3, index in enumerate(index_list_level3):
        if number_level3 == 0:
            continue
        # 如果不是最后一个
        if number_level3 != len(index_list_level3) - 1:
            level3.append(content[index:index_list_level3[number_level3 + 1]])
        else:
            level3.append(content[index:])
    print("三级标题处理完成！")
    return level3


def judge_level4(content):
    """
    # 判断原文件中是否含有四级标题
    :param content: 一个级别标题对应的上下文或者原文组成的上下文
    :return:返回四级标题所在上下文中的位置
    """
    index_list_level4 = [0]
    for headline_level4 in chinese_number_list_level4:
        index = content.find(headline_level4)
        if (index != -1) & (index > index_list_level4[-1]):
            index_list_level4.append(index)
        elif index == -1:
            break
    # print(len(index_list_level4))
    return index_list_level4


def dispose_level4(index_list_level4, content, full_context, level, index_level1=0, index_level2=0):
    """
    当存在四级标题时：
    若一级标题、二级标题和三级标题都存在，此时level=4，则对一个三级标题对应的上下文进行处理：添加嵌套字典

    若一级标题和二级标题存在，三级标题不存在，此时level=3，则对一个二级标题对应的上下文进行处理：添加嵌套字典
    若一级标题和三级标题存在，二级标题不存在，此时level=3，则对一个三级标题对应的上下文进行处理：添加嵌套字典
    若二级标题和三级标题存在，一级标题不存在，此时level=3，则对一个三级标题对应的上下文进行处理：添加嵌套字典

    当一级标题存在，二级标题和三级标题不存在，此时level=2，则对一个一级标题对应的上下文进行处理：添加嵌套字典
    当二级标题存在，一级标题和三级标题不存在，此时level=2，则对一个二级标题对应的上下文进行处理：添加嵌套字典
    当三级标题存在，一级标题和二级标题不存在，此时level=2，则对一个三级标题对应的上下文进行处理：添加嵌套字典

    若一级标题、二级标题和三级标题都不存在，此时level=1，则对原上下文进行处理：构造导语字典元素

    最后同一构造由各个四级标题对应的内容组成的列表
    :param index_list_level4:三级标题在一级标题对应的上下文中的索引
    :param content:一个级别标题对应的上下文或者原上下文
    :param full_context:字典
    :param level:当前级别标题在原文中的级别，比如不存在一级标题时，则二级标题在上下文中标题级别就是1
    :param index_level1=0：当level=3或level=4，此时对哪个一级标题对应的内容进行处理
    :param index_level2=0：当level=4，此时对哪个二级标题对应的内容进行处理
    :return:各个四级标题对应的内容组成的列表
    """
    # 处理四级标题部分
    level4 = []  # 四级标题对应的内容
    context = {}
    if level == 1:
        # 此时四级标题实际在原上下文中是一级标题
        # 处理导语部分
        introduction = content[0:index_list_level4[1]]
        full_context["introduction"] = introduction
    elif level == 2:
        # 此时四级标题实际在原上下文中是二级标题
        title_level3 = content[0:index_list_level4[1]]
        context["title"] = title_level3
        context["context"] = []
        full_context["context"].append(context)
    elif level == 3:
        # 此时四级标题实际在原上下文中是三级标题
        title_level3 = content[0:index_list_level4[1]]
        context["title"] = title_level3
        context["context"] = []
        full_context["context"][index_level1]["context"].append(context)
    elif level == 4:
        # 此时四级标题实际在原上下文中是四级标题
        title_level3 = content[0:index_list_level4[1]]
        context["title"] = title_level3
        context["context"] = []
        full_context["context"][index_level1]["context"][index_level2]["context"].append(context)
    # 处理正文
    for number_level4, index in enumerate(index_list_level4):
        if number_level4 == 0:
            continue
        # 如果不是最后一个
        if number_level4 != len(index_list_level4) - 1:
            level4.append(content[index:index_list_level4[number_level4 + 1]])
        else:
            level4.append(content[index:])
    print("四级标题处理完成！")
    return level4


def put_context(level_content_list, full_context, level, index_level1=0, index_level2=0, index_level3=0):
    """
    将所有当前级别标题对应的上下文放入字典中
    :param level_content_list:所有当前级别标题对应的上下文构成的列表
    :param full_context: 字典
    :param level: 当前级别标题在原文中的级别
    :param index_level1: 第一标题级别的第几个句子包含当前标题级别
    :param index_level2: 第二标题级别的第几个句子包含当前标题级别
    :param index_level3: 第二标题级别的第几个句子包含当前标题级别
    :return:
    """
    if level == 1:
        for content in level_content_list:
            full_context["context"].append(
                {"title": "", "context": content})
    elif level == 2:
        for content in level_content_list:
            full_context["context"][index_level1]["context"].append(
                {"title": "", "context": content})
    elif level == 3:
        for content in level_content_list:
            full_context["context"][index_level1]["context"][index_level2]["context"].append(
                {"title": "", "context": content})
    elif level == 4:
        for content in level_content_list:
            full_context["context"][index_level1]["context"][index_level2]["context"][index_level3]["context"].append(
                {"title": "", "context": content})


def put_paragraph(paragraph, full_context, level, index_level1=0, index_level2=0, index_level3=0):
    """
    对于当前标题级别对应的内容，若不包含下一级标题，则将其放入字典中
    :param paragraph: 当前标题级别对应的内容
    :param full_context: 字典
    :param level: 当前级别标题在原文中的级别
    :param index_level1: 第一标题级别的第几个句子包含当前标题级别
    :param index_level2: 第二标题级别的第几个句子包含当前标题级别
    :param index_level3: 第二标题级别的第几个句子包含当前标题级别
    :return:
    """
    if level == 1:
        context = {"title": "", "context": paragraph}
        full_context["context"].append(context)
    elif level == 2:
        context = {"title": "", "context": paragraph}
        full_context["context"][index_level1]["context"].append(context)
    elif level == 3:
        context = {"title": "", "context": paragraph}
        full_context["context"][index_level1]["context"][index_level2]["context"].append(context)
    elif level == 4:
        context = {"title": "", "context": paragraph}
        full_context["context"][index_level1]["context"][index_level2]["context"][index_level3]["context"].append(
            context)


if __name__ == '__main__':
    # read_pdf_save_text()
    save_dict()
