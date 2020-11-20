import os
import json

chinese_number_list_level3 = []
for chinese_number in range(1, 100):
    obj_1 = str(chinese_number) + "．"
    obj_2 = str(chinese_number) + "."
    chinese_number_list_level3.append(obj_1)
    chinese_number_list_level3.append(obj_2)
print(chinese_number_list_level3)
chinese_number_list_level1 = ["一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、", "十、",
                              "十一、", "十二、", "十三、", "十四、", "十五、", "十六、", "十七、", "十八、", "十九、", "二十、"]
chinese_number_list_level2 = ["（一）", "（二）", "（三）", "（四）", "（五）", "（六）", "（七）", "（八）", "（九）", "（十）",
                              "（十一）", "（十二）", "（十三）", "（十四）", "（十五）", "（十六）", "（十七）", "（十八）",
                              "（十九）", "（二十）"]
# chinese_number_list_level3 = ['1．', '2．', '3．', '4．', '5．', '6．', '7．', '8．', '9．', '10．']
chinese_number_list_level4 = ["（1）", "（2）", "（3）", "（4）", "（5）", "（6）", "（7）", "（8）", "（9）", "（10）",
                              "（11）", "（12）", "（13）", "（14）", "（15）", "（16）", "（17）", "（18）", "（19）", "（20）"]


def find_all_file(base, end=".pdf"):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith(end):
                fullname = os.path.join(root, f)
                yield fullname


def read_txt(file_path):
    # read_path = "E:\\Putian\\Dataset\\分文档后的所有文档\\txt\\"
    # file_path = read_path + "附件3福建省级高层次人才认定和支持办法实施细则（试行）.txt"
    with open(file_path, "r") as f:
        content = f.read()
    return content, os.path.basename(file_path)


def save_txt():
    base = 'E:\\Putian\\Dataset\\手动处理txt\\'
    for file_path in find_all_file(base, end='.txt'):
        full_content, file_name = read_txt(file_path)
        new_content = dispose_txt(full_content)

        new_file_path = "E:\\Putian\\Dataset\\txt_documents\\"
        new_file_path += file_name
        with open(new_file_path, 'w') as f:
            f.writelines(new_content)
            print("txt保存成功！")


def dispose_txt(full_content):
    # print(full_content)
    content = full_content.split('\n')
    new_content = []
    for line in content:
        line = line.strip()
        line = ''.join(line.split())
        print(line)
        if line != "":
            new_content.append(line + '\n')
    return new_content


if __name__ == "__main__":
    save_txt()
