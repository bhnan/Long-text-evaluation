import re
import os

class TextProcessor:
    def __init__(self, file_path):
        self.file_path = os.path.normpath(file_path)
        self.sections = []

    def process(self):
        try:
            current_section = None
            current_subsection = None
            current_subsubsection = None
            current_paragraphs = []

            with open(self.file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if self._is_main_title(line):
                        # 新章节
                        if current_section:
                            self._add_paragraphs_to_current_level(current_section, current_paragraphs)
                            self.sections.append(current_section)
                        current_section = {"title": line, "paragraphs": [], "subsections": []}
                        current_subsection = None
                        current_subsubsection = None
                        current_paragraphs = []
                    elif self._is_subtitle(line):
                        # 二级标题
                        self._add_paragraphs_to_current_level(current_section, current_paragraphs)
                        current_subsection = {"title": line, "paragraphs": [], "subsubsections": []}
                        current_section["subsections"].append(current_subsection)
                        current_subsubsection = None
                        current_paragraphs = []
                    elif self._is_subsubtitle(line):
                        # 三级标题
                        self._add_paragraphs_to_current_level(current_subsection, current_paragraphs)
                        current_subsubsection = {"title": line, "paragraphs": []}
                        current_subsection["subsubsections"].append(current_subsubsection)
                        current_paragraphs = []
                    elif line:
                        # 非空行，添加到当前段落
                        current_paragraphs.append(line)
                    else:
                        # 空行，表示段落结束
                        self._add_paragraphs_to_current_level(current_subsubsection or current_subsection or current_section, current_paragraphs)
                        current_paragraphs = []

            # 处理最后的段落和章节
            self._add_paragraphs_to_current_level(current_subsubsection or current_subsection or current_section, current_paragraphs)
            if current_section:
                self.sections.append(current_section)

            return self.sections
        except OSError as e:
            print(f"无法打开文件: {self.file_path}")
            print(f"错误信息: {str(e)}")
            return []  # 或者根据需要返回适当的值

    def _add_paragraphs_to_current_level(self, current_level, paragraphs):
        if current_level is not None and paragraphs:
            if "paragraphs" not in current_level:
                current_level["paragraphs"] = []
            current_level["paragraphs"].append(" ".join(paragraphs))

    def _is_main_title(self, line):
        return re.match(r'^[零一二三四五六七八九十]+、', line)

    def _is_subtitle(self, line):
        return re.match(r'^[1-9]\d*\s', line)

    def _is_subsubtitle(self, line):
        return re.match(r'^[1-9]\d*\.[1-9]\d*\s', line)

    def get_section_count(self):
        return len(self.sections)

    def get_total_paragraph_count(self):
        count = 0
        for section in self.sections:
            count += len(section['paragraphs'])
            for subsection in section['subsections']:
                count += len(subsection['paragraphs'])
                for subsubsection in subsection['subsubsections']:
                    count += len(subsubsection['paragraphs'])
        return count

    def print_summary(self):
        print(f"总章节数: {self.get_section_count()}")
        print(f"总段落数: {self.get_total_paragraph_count()}")
        print("\n章节摘要:")
        for i, section in enumerate(self.sections, 1):
            print(f"  章节 {i}: {section['title']}")
            print(f"    段落数: {len(section['paragraphs'])}")
            for j, para in enumerate(section['paragraphs'], 1):
                print(f"      段落 {j}: {para[:100]}...")
            for j, subsection in enumerate(section['subsections'], 1):
                print(f"    二级标题 {i}.{j}: {subsection['title']}")
                print(f"      段落数: {len(subsection['paragraphs'])}")
                for k, para in enumerate(subsection['paragraphs'], 1):
                    print(f"        段落 {k}: {para[:100]}...")
                for k, subsubsection in enumerate(subsection['subsubsections'], 1):
                    print(f"      三级标题 {i}.{j}.{k}: {subsubsection['title']}")
                    print(f"        段落数: {len(subsubsection['paragraphs'])}")
                    for l, para in enumerate(subsubsection['paragraphs'], 1):
                        print(f"          段落 {l}: {para[:100]}...")
            print()

# 使用示例
if __name__ == "__main__":
    file_path = r"长文本文件\a.txt"  # 使用原始字符串
    processor = TextProcessor(file_path)
    processed_sections = processor.process()
    processor.print_summary()
