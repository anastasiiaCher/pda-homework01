import pandas as pd
import re
from typing import Tuple


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    fsuir = data[data["факультет"] == "факультет систем управления и робототехники"]
    num_students = len(fsuir)
    num_groups = fsuir["группа"].nunique()
    return num_students, num_groups, fsuir


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
 surname = df[df.duplicated(subset=["surname"], keep=False)]
    total_count = len(surname)
    if total_count == 0:
        return False, 0, pd.Series(dtype=int), "Нет однофамильцев"

    course_count = surname["курс"].value_counts()
    top_group = surname["группа"].value_counts().idxmax()
    return True, total_count, course_count, top_group
    

# Задача 3
def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    if pd.isna(patronym) or not isinstance(patronym, str) or not patronym.strip():
         return 'empty'
    
    patronym_lower = patronym.lower().strip()
    
    male_endings = ('ович', 'евич', 'ич')
    female_endings = ('овна', 'евна', 'ична', 'инична')
    
    if patronym_lower.endswith(male_endings):
        return 'male'
    elif patronym_lower.endswith(female_endings):
        return 'female'
    else:
        return 'other'


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
     - количество студентов без отчества
     - серию с распределением студентов по полу 
    """
    patronym_col = 'patronim'
    df['gender_category'] = df[patronym_col].apply(gender_identification)
    all_counts = df['gender_category'].value_counts().to_dict()
    
    students_without_patronym = all_counts.get('empty', 0)
    foreign_or_unclear = all_counts.get('other', 0)
    
    gender_counts = {
        'male': all_counts.get('male', 0),
        'female': all_counts.get('female', 0)
    }

    return students_without_patronym, foreign_or_unclear, gender_counts


# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    faculty_col = 'факультет'
        
    counts_series = data[faculty_col].value_counts()
    faculty_counts_df = counts_series.reset_index()
    faculty_counts_df.columns = ['Факультет', 'Количество студентов']

    max_faculty_name = counts_series.idxmax()
    max_count = counts_series.max()
    max_faculty = (max_faculty_name, max_count)
    
    min_faculty_name = counts_series.idxmin()
    min_count = counts_series.min()
    min_faculty = (min_faculty_name, min_count)
    
    return faculty_counts_df, max_faculty, min_faculty


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    course_col = 'курс'
    group_col = 'группа' 
    group_sizes = data.groupby([course_col, group_col]).size().reset_index(name='count')
    
    mean_students = group_sizes.groupby(course_col)['count'].mean()
    median_students = group_sizes.groupby(course_col)['count'].median()
    
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
    name_col = 'name'
    group_col = 'группа'
    faculty_col = 'факультет'
    course_col = 'курс'
    
    popular_name = data[name_col].value_counts().idxmax()
    
    students_with_popular_name = data[data[name_col] == popular_name]
    
    name_group = students_with_popular_name[group_col].value_counts().idxmax()
    
    # Так как студенты одной группы учатся на одном факультете и курсе, беру первую строку этой группы
    group_info = data[data[group_col] == name_group].iloc[0]
    faculty = group_info[faculty_col]
    course = int(str(group_info[course_col]).split('-')[0])
    
    name_ratio = round(len(students_with_popular_name) / len(data), 2)
    
    return popular_name, name_group, faculty, course, name_ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    name_counts = data['name'].value_counts()
    
    unique_names = name_counts[name_counts == 1].index
    
    mask = data['name'].isin(unique_names) & data['name'].str.startswith(('П', 'п'))
    
    result_df = data.loc[mask, ['фио', 'факультет', 'курс']]
    
    return result_df


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    course_col = 'курс'
    faculty_col = 'факультет'
    grade_col = 'средний_балл'
    patronym_col = 'patronim'  
  
    third_course = data[course_col].astype(str).str.startswith('3')
    data_3rd_course = data[third_course].copy() 
    
    faculty_avg_grades = data_3rd_course.groupby(faculty_col)[grade_col].mean()
    best_faculty = faculty_avg_grades.idxmax()
    
    best_fac_students = data_3rd_course[data_3rd_course[faculty_col] == best_faculty].copy()
    
    best_fac_students['gender'] = best_fac_students[patronym_col].apply(gender_identification)
    valid_genders = best_fac_students['gender'].isin(['male', 'female'])
    valid_students = best_fac_students[valid_genders]
    
    gender_avg_grades = valid_students.groupby('gender')[grade_col].mean()
    
    
    best_gender = gender_avg_grades.idxmax()
    best_grade_exact = gender_avg_grades.max()
    best_grade_rounded = int(round(best_grade_exact))
    
    return best_faculty, best_gender, best_grade_rounded


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    isu_col = 'ису'
    fio_col = 'фио'
    faculty_col = 'факультет'
    course_col = 'курс'
    group_col = 'группа'
    
    df_sorted = data.sort_values(by=isu_col).reset_index(drop=True)
    
    for i in range(len(df_sorted) - 4):
        window = df_sorted.iloc[i:i+5]
        isu_values = window[isu_col].tolist()
        
        if isu_values[4] - isu_values[0] == 4 and len(set(isu_values)) == 5:
            return window[[isu_col,fio_col, faculty_col, course_col, group_col]]
            
    return pd.DataFrame(columns=[isu_col,fio_col, faculty_col, course_col, group_col])


if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")
    data["surname"] = data["фио"].str.split(" ").str[0]
    data["name"] = data["фио"].str.split(" ").str[1]
    data["patronim"] = data["фио"].str.split(" ").str[2]
    
    # Задача 1
    num_students, num_groups, fsuir = filter_fsuir_students(data)
    print(f"Студентов на ФСУиР: {num_students}, Групп: {num_groups}")
    
    # Задача 2
    has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group = find_homonymous_students(fsuir)
    print(f"Есть однофамильцы: {has_homonyms}, Всего: {total_homonyms}, Группа с максимумом: {max_homonym_group}")
    print(f"На каждом курсе: {homonyms_per_course}")
    
    # Задача 3
    students_without_patronym,foreign_or_unclear, gender_counts = analyze_patronyms(fsuir)
    print(f"Студентов без отчества: {students_without_patronym}")
    print("Распределение по полу:", gender_counts)
    print("Иностранные студенты",foreign_or_unclear)
    
    # Задача 4
    faculty_counts, max_faculty, min_faculty = faculty_statistics(data)
    print(f"Факультет с наибольшим числом студентов: {max_faculty}")
    print(f"Факультет с наименьшим числом студентов: {min_faculty}")
    
    # Задача 5
    mean_students, median_students = course_statistics(data)
    print("Среднее число студентов на курсах:", mean_students)
    print("Медианное число студентов на курсах:", median_students)
    
    # Задача 6
    popular_name, name_group, faculty, course, name_ratio = most_popular_name(data)
    print(f"Самое популярное имя: {popular_name}, Группа: {name_group}, Факультет: {faculty}, Курс: {course}")
    print(f"Доля студентов с этим именем: {name_ratio}")
    
    # Задача 7
    result_7 = find_students_with_name_starting_P(data)
    print("Студенты с именем, начинающимся на П и встречающимся ровно один раз:")
    print(result_7)
    
    # Задача 8
    fac, best_gender, best_grade = highest_avg_grade_faculty(data)
    print(f"Факультет с высоким средним баллом 3-го курса: {fac}")
    print(f"Пол с наивысшим средним баллом: {best_gender}, Средний балл: {best_grade}")
    
    # Задача 9
    result_9 = find_consecutive_students(data)
    print("Первые 5 студентов с подряд идущими табельными номерами:")
    print(result_9)
