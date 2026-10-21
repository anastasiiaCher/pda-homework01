import pandas as pd
import re
from typing import Tuple
import numpy as np 

df = pd.read_csv('isu_fake_data.csv')
df_suir = df[df['факультет'].str.contains('факультет систем управления и робототехники', na = False, case = False)]

# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    count_suir = len(df_suir)
    groups_suir = df_suir['группа'].nunique()
    return count_suir, groups_suir, df_suir


# Задача 2
def find_homonymous_students(df: pd.DataFrame) -> Tuple[bool, int, pd.Series, str]:
    df_suir['фамилия'] = df_suir['фио'].str.split().str[0]
    surname_counts = df_suir['фамилия'].value_counts()
    duplicates = surname_counts[surname_counts>1].index
    df_odnofam = df_suir[df_suir['фамилия'].isin(duplicates)]
    df_course_odnofam = df_odnofam.groupby('курс').size().sort_index()
    group_odnofam = df_odnofam.groupby('группа').size()
    group_max_odnofam_name = group_odnofam.idxmax()
    vsevo = df_course_odnofam.sum()
    return True if len(duplicates)>0 else False, vsevo, df_course_odnofam, group_max_odnofam_name
    

# Задача 3
def gender_identification(patronym: str) -> str:
    pass
def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    df_suir['отчество'] = df_suir['фио'].str.split().str[2]
    no_otchestvo = df_suir['отчество'].isna().sum()
    df_girls = df_suir[df_suir['отчество'].str.contains('на', na=False)]
    df_boys = df_suir[df_suir['отчество'].str.contains('ич', na = False)]
    gender_counts = pd.DataFrame([[len(df_girls), len(df_boys), len(df_suir)-len(df_girls)-len(df_boys)-no_otchestvo]], 
                                 columns = ['девушек','парней','не определено по отчеству'])
    return no_otchestvo, gender_counts


# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    students_faculty = df.groupby('факультет').size()
    max = students_faculty.idxmax()
    min = students_faculty.idxmin()
    return students_faculty, max, min


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    students_courses = df.groupby('курс').size()
    return students_courses.groupby(level=0).mean(), students_courses.groupby(level=0).median()


# Задача 6
def most_popular_name(data: pd.DataFrame) -> Tuple[str, str, str, int, float]:
    df['имя'] = df['фио'].str.split().str[1]
    names = df.groupby('имя').size()
    df_most_popular_name = df[df['имя'] == names.idxmax()][['курс', 'факультет', 'группа']]
    mpn_groups = df_most_popular_name.groupby('группа').size()
    df_vsp = df[df['группа'] == mpn_groups.idxmax()].iloc[0]
    course = df_vsp['курс']
    fac = df_vsp['факультет']
    name_ratio = len(df_most_popular_name)/len(df)
    return names.idxmax(), mpn_groups.idxmax(),  fac, course, f'{name_ratio:.2f}'

# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    df['первая буква имени'] = df['фио'].str.split().str[1].str[0]
    name_count = df['имя'].value_counts()
    one_name = name_count[name_count == 1].index
    df_one_name = df[df['имя'].isin(one_name)][df['первая буква имени'] == 'П']
    df_p = df_one_name[['фио', 'факультет', 'курс']]
    return df_p

# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    df['отчество'] = df['фио'].str.split().str[2]
    uslov = [df['отчество'].str.contains('на', na =  False), df['отчество'].str.contains('ич', na =  False)] 
    choise = ['ж', 'м']
    df['пол'] = np.select(uslov, choise, default='не определен')
    faculty_mean_max = df[df['курс'] == '3-й'].groupby('факультет')['средний_балл'].mean().idxmax()
    
    mask_g = (
        (df['пол'] == 'ж') & 
        (df['факультет'] == faculty_mean_max) & 
        (df['курс'] == '3-й')
    )  
    mask_b = (
        (df['пол'] == 'м') &
        (df['факультет'] == faculty_mean_max) &
        (df['курс'] == '3-й')
    )
    
    girls_mean_3 = round(df.loc[mask_g, 'средний_балл'].mean())
    boys_mean_3 = round(df.loc[mask_b, 'средний_балл'].mean())

    best_gender_3 = ''
    best_grade_3 = []
    if girls_mean_3 > boys_mean_3:
        best_gender_3 += 'ж'
        best_grade_3 = girls_mean_3
    else:
        best_gender_3 += 'м'
        best_grade_3 = boys_mean_3
    return faculty_mean_max, best_gender_3, int(best_grade_3)

# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.sort_values('ису')
    passt = sorted_df['ису'].shift(-4) - sorted_df['ису'] == 4
    first_passt = sorted_df[passt]['ису'].iloc[0]
    df_first_odnogr = sorted_df[sorted_df['ису'].between(first_passt, first_passt+4)][['ису', 'фио', 'факультет', 'курс', 'группа']]
    return df_first_odnogr


if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")
    
    # Задача 1
    num_students, num_groups, fsuir = filter_fsuir_students(data)
    print(f"\nЗадача 1:\nСтудентов на ФСУиР: {num_students}, Групп: {num_groups}\n\n")
    
    # Задача 2
    has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group = find_homonymous_students(fsuir)
    print("Задача 2:")
    print(f"Есть однофамильцы: {has_homonyms}, Всего: {total_homonyms}, Группа с максимумом: {max_homonym_group}")
    print(f"На каждом курсе: {homonyms_per_course}\n\n")
    
    # Задача 3
    students_without_patronym, gender_counts = analyze_patronyms(fsuir)
    print('Задача 3:')
    print(f"Студентов без отчества: {students_without_patronym}")
    print("Распределение по полу:\n", gender_counts, '\n\n')
    
    # Задача 4
    faculty_counts, max_faculty, min_faculty = faculty_statistics(data)
    print('Задача 4:')
    print(f"Факультет с наибольшим числом студентов: {max_faculty}")
    print(f"Факультет с наименьшим числом студентов: {min_faculty}\n\n")
    
    # Задача 5
    mean_students, median_students = course_statistics(data)
    print('Задача 5:')
    print("Среднее число студентов на курсах:", mean_students)
    print("Медианное число студентов на курсах:", median_students, '\n\n')
    
    # Задача 6
    popular_name, name_group, faculty, course, name_ratio = most_popular_name(data)
    print('Задача 6:')
    print(f"Самое популярное имя: {popular_name},\nГруппа: {name_group},\nФакультет: {faculty},\nКурс: {course}")
    print(f"Доля студентов с этим именем: {name_ratio}\n\n")
    
    # Задача 7
    result_7 = find_students_with_name_starting_P(data)
    print('Задача 7: ')
    print("Студенты с именем, начинающимся на П и встречающимся ровно один раз:")
    print(result_7)
    print('\n\n')
    
    # Задача 8
    fac, best_gender, best_grade = highest_avg_grade_faculty(data)
    print('Задача 8: ')
    print(f"Факультет с высоким средним баллом 3-го курса: {fac}")
    print(f"Пол с наивысшим средним баллом: {best_gender}, Средний балл: {best_grade}\n\n")


    # Задача 9
    result_9 = find_consecutive_students(data)
    print('Задача 9: \n')
    print("Первые 5 студентов с подряд идущими табельными номерами:")
    print(result_9)
