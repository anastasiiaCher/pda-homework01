import pandas as pd
import re
from typing import Tuple


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    fsuir = data[data['факультет'] == 'факультет систем управления и робототехники'].copy()
    return len(fsuir), fsuir['группа'].nunique(), fsuir


# Задача 2
def find_homonymous_students(df: pd.DataFrame) -> Tuple[bool, int, pd.Series, str]:
    """
    Проверяет наличие однофамильцев на ФСУиР, их количество, распределение по курсам
    и определяет группу с наибольшим числом однофамильцев.
    Возвращает:
     - логическое значение (наличие однофамильцев)
     - общее количество однофамильцев
     - серию с числом однофамильцев по курсам
     - группу с максимальным числом однофамильцев
    """
    df = df.copy()
    df['фамилия'] = df['фио'].str.split().str[0]
    
    surname_counts = df['фамилия'].value_counts()
    repeated_surnames = surname_counts[surname_counts > 1].index.tolist()
    
    has_homonyms = len(repeated_surnames) > 0
    
    if not has_homonyms:
        return False, 0, pd.Series(dtype=int), None
        
    homonyms_df = df[df['фамилия'].isin(repeated_surnames)]
    total_homonyms = len(homonyms_df)
    homonyms_per_course = homonyms_df.groupby('курс').size()
    max_homonym_group = homonyms_df.groupby('группа').size().idxmax()
    
    return has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group


# Задача 3
def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    if pd.isna(patronym):
        return 'unknown'
    if re.search(r'(ович|евич|ич)$', patronym, re.IGNORECASE):
        return 'male'
    elif re.search(r'(овна|евна|ична|инична)$', patronym, re.IGNORECASE):
        return 'female'
    return 'unknown'


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
     - количество студентов без отчества
     - серию с распределением студентов по полу
    """
    df = df.copy()
    
    def extract_patronym(fio):
        parts = fio.split()
        if len(parts) >= 3:
            last_word = parts[-1]
            if re.search(r'(ович|евич|ич|овна|евна|ична|инична)$', last_word, re.IGNORECASE):
                return last_word
        return None
        
    df['отчество'] = df['фио'].apply(extract_patronym)
    
    students_without_patronym = df['отчество'].isna().sum()
    
    with_patronym = df[df['отчество'].notna()].copy()
    with_patronym['пол'] = with_patronym['отчество'].apply(gender_identification)
    
    gender_counts = with_patronym['пол'].value_counts()
    
    return students_without_patronym, gender_counts
    
# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    faculty_counts = data['факультет'].value_counts()
    
    max_faculty = (faculty_counts.idxmax(), faculty_counts.max())
    min_faculty = (faculty_counts.idxmin(), faculty_counts.min())
    
    return faculty_counts, max_faculty, min_faculty


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    course_counts = data.groupby('курс').size()
    
    mean_students = pd.Series([course_counts.mean()], index=['mean'])
    median_students = pd.Series([course_counts.median()], index=['median'])
    
    return mean_students, median_students


# Задача 6
def most_popular_name(data: pd.DataFrame) -> Tuple[str, str, str, int, float]:
    """
    Определяет самое популярное имя, группу с наибольшим количеством студентов с этим именем,
    факультет, курс и долю таких студентов в общем числе.
    Возвращает результат в следующем порядке:
     1. самое частое имя
     2. группа
     3. факультет
     4. доля
    """
    data = data.copy()
    data['имя'] = data['фио'].str.split().str[1]
    
    name_counts = data['имя'].value_counts()
    popular_name = name_counts.idxmax()
    
    name_df = data[data['имя'] == popular_name]
    
    group_counts = name_df.groupby('группа').size()
    top_group = group_counts.idxmax()
    
    group_info = data[data['группа'] == top_group].iloc[0]
    faculty = group_info['факультет']
    course = group_info['курс']
    
    name_ratio = round(len(name_df) / len(data), 2)
    
    return popular_name, top_group, faculty, course, name_ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    data = data.copy()
    data['имя'] = data['фио'].str.split().str[1]
    
    name_counts = data['имя'].value_counts()
    unique_names = name_counts[name_counts == 1].index.tolist()
    p_unique_names = [n for n in unique_names if str(n).startswith('П')]
    
    result = data[data['имя'].isin(p_unique_names)][['фио', 'факультет', 'курс']]
    
    return result


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    data = data.copy()
    
    def get_gender(fio):
        parts = fio.split()
        if len(parts) >= 3:
            last = parts[-1].lower()
            if last.endswith(('ович', 'евич', 'ич')):
                return 'male'
            elif last.endswith(('овна', 'евна', 'ична', 'инична')):
                return 'female'
        if len(parts) >= 2:
            name = parts[1]
            if name.endswith(('а', 'я', 'ия')):
                return 'female'
            return 'male'
        return 'unknown'
        
    data['пол'] = data['фио'].apply(get_gender)
    
    third_course = data[data['курс'] == '3-й']
    
    avg_by_faculty = third_course.groupby('факультет')['средний_балл'].mean()
    best_faculty = avg_by_faculty.idxmax()
    
    best_faculty_3rd = third_course[third_course['факультет'] == best_faculty]
    gender_avg = best_faculty_3rd.groupby('пол')['средний_балл'].mean()
    
    best_gender = gender_avg.idxmax()
    best_grade = int(round(gender_avg.max()))
    
    return best_faculty, best_gender, best_grade


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    df_sorted = data.sort_values('ису').reset_index(drop=True)
    
    for i in range(len(df_sorted) - 4):
        nums = df_sorted.loc[i:i+4, 'ису'].values
        if all(nums[j+1] - nums[j] == 1 for j in range(4)):
            return df_sorted.loc[i:i+4, ['фио', 'факультет', 'курс', 'группа']]
            
    return pd.DataFrame(columns=['фио', 'факультет', 'курс', 'группа'])


if __name__ == "__main__":
    data = fsuir
    
    # Задача 1
    num_students, num_groups, fsuir = filter_fsuir_students(data)
    print(f"1. Студентов на ФСУиР: {num_students}, Групп: {num_groups}")
    
    # Задача 2
    has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group = find_homonymous_students(fsuir)
    print(f"\n2. Есть однофамильцы: {has_homonyms}, Всего: {total_homonyms}, Группа с максимумом: {max_homonym_group}")
    print(f"На каждом курсе:\n{homonyms_per_course}")
    
    # Задача 3
    students_without_patronym, gender_counts = analyze_patronyms(fsuir)
    print(f"\n3. Студентов без отчества: {students_without_patronym}")
    print("Распределение по полу:", gender_counts.to_dict())
    
    # Задача 4
    faculty_counts, max_faculty, min_faculty = faculty_statistics(data)
    print(f"\n4. Факультет с наибольшим числом студентов: {max_faculty}")
    print(f"Факультет с наименьшим числом студентов: {min_faculty}")
    
    # Задача 5
    mean_students, median_students = course_statistics(data)
    print(f"\n5. Среднее число студентов на курсах: {mean_students.values[0]:.2f}")
    print(f"Медианное число студентов на курсах: {median_students.values[0]:.2f}")
    
    # Задача 6
    popular_name, name_group, faculty, course, name_ratio = most_popular_name(data)
    print(f"\n6. Самое популярное имя: {popular_name}, Группа: {name_group}, Факультет: {faculty}, Курс: {course}")
    print(f"Доля студентов с этим именем: {name_ratio}")
    
    # Задача 7
    result_7 = find_students_with_name_starting_P(data)
    print("\n7. Студенты с именем, начинающимся на П и встречающимся ровно один раз:")
    print(result_7.to_string(index=False))
    
    # Задача 8
    fac, best_gender, best_grade = highest_avg_grade_faculty(data)
    print(f"\n8. Факультет с высоким средним баллом 3-го курса: {fac}")
    print(f"Пол с наивысшим средним баллом: {best_gender}, Средний балл: {best_grade}")
    
    # Задача 9
    result_9 = find_consecutive_students(data)
    print("\n9. Первые 5 студентов с подряд идущими табельными номерами:")
    print(result_9.to_string(index=False))
