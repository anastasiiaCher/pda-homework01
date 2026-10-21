import pandas as pd
import re
from typing import Tuple

data = pd.read_csv("isu_fake_data.csv")
data.head()
# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    fsuir = data[data['факультет'] == 'факультет систем управления и робототехники']
    num_students = len(fsuir)
    num_groups = fsuir['группа'].nunique()
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
    namesakes = df[df.duplicated('surname', keep=False)]
    has_homonyms = not namesakes.empty
    total_homonyms = len(namesakes)
    homonyms_per_course = namesakes['курс'].value_counts() if has_homonyms else pd.Series(dtype=int)
    max_homonym_group = namesakes.groupby('группа')['surname'].count().idxmax() if has_homonyms else ""
    return has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group


# Задача 3
def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    if pd.isna(patronym):
        return 'unknown'
    if str(patronym).endswith(('ович', 'евич', 'ич')):
        return 'male'
    if str(patronym).endswith(('овна', 'евна', 'ична', 'инична')):
        return 'female'
    return 'unknown'


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
     - количество студентов без отчества
     - серию с распределением студентов по полу
    """
    students_without_patronym = df['patronim'].isna().sum()
    genders = df['patronim'].apply(gender_identification)
    gender_counts = genders[genders != 'unknown'].value_counts()
    return students_without_patronym, gender_counts


# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    faculty_counts_df = data['факультет'].value_counts().reset_index()
    faculty_counts_df.columns = ['факультет', 'количество']
    max_faculty = (faculty_counts_df.iloc[0]['факультет'], int(faculty_counts_df.iloc[0]['количество']))
    min_faculty = (faculty_counts_df.iloc[-1]['факультет'], int(faculty_counts_df.iloc[-1]['количество']))
    return faculty_counts_df, max_faculty, min_faculty


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    group_sizes = data.groupby(['курс', 'факультет']).size().reset_index(name='size')
    mean_students = group_sizes.groupby('курс')['size'].mean()
    median_students = group_sizes.groupby('курс')['size'].median()
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
    popular_name = data['name'].value_counts().idxmax()
    name_df = data[data['name'] == popular_name]
    name_group = name_df.groupby('группа').size().idxmax()
    group_info = name_df[name_df['группа'] == name_group].iloc[0]
    faculty = group_info['факультет']
    course = int(group_info['курс'])
    name_ratio = round(len(name_df) / len(data), 2)
    return popular_name, name_group, faculty, course, name_ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    name_counts = data['name'].value_counts()
    unique_p_names = name_counts[(name_counts == 1) & (name_counts.index.str.startswith('П'))].index
    return data[data['name'].isin(unique_p_names)][['фио', 'факультет', 'курс']]


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    course3 = data[data['курс'] == 3].copy()
    best_fac = course3.groupby('факультет')['средний_балл'].mean().idxmax()
    best_fac_df = course3[course3['факультет'] == best_fac].copy()
    best_fac_df['gender'] = best_fac_df['patronim'].apply(gender_identification)
    gender_scores = best_fac_df[best_fac_df['gender'] != 'unknown'].groupby('gender')['средний_балл'].mean()
    best_gender = gender_scores.idxmax()
    best_grade = int(round(gender_scores.max()))
    return best_fac, best_gender, best_grade

# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    df_sorted = data.sort_values('ису').drop_duplicates(subset=['ису']).reset_index(drop=True)
    df_sorted['block'] = (df_sorted['ису'].diff() != 1).cumsum()
    block_sizes = df_sorted.groupby('block').size()
    target_block = block_sizes[block_sizes >= 5].index[0]
    result_df = df_sorted[df_sorted['block'] == target_block].head(5)
    return result_df[['ису', 'фио', 'факультет', 'курс', 'группа']].reset_index(drop=True)

if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")
    data['курс'] = data['курс'].astype(str).str.extract(r'(\d+)').astype(int)
    data["surname"] = data['фио'].str.split().str[0]
    data["name"] = data['фио'].str.split().str[1]
    data["patronim"] = data['фио'].str.split().str[2]

    # Задача 1
    num_students, num_groups, fsuir = filter_fsuir_students(data)
    print(f"Студентов на ФСУиР: {num_students}, Групп: {num_groups}")

    # Задача 2
    has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group = find_homonymous_students(fsuir)
    print(f"Есть однофамильцы: {has_homonyms}, Всего: {total_homonyms}, Группа с максимумом: {max_homonym_group}")
    print(f"На каждом курсе: {homonyms_per_course}")

    # Задача 3
    students_without_patronym, gender_counts = analyze_patronyms(fsuir)
    print(f"Студентов без отчества: {students_without_patronym}")
    print("Распределение по полу:", gender_counts)

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
