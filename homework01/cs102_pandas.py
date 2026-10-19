import pandas as pd
import re
from typing import Tuple
# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    fsuir = data[data['факультет'].str.contains(
        r'ФСУиР|систем управления и робототехники',
        case=False,
        na=False
    )
    ].copy()
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    return len(fsuir), fsuir['группа'].nunique(), fsuir

# Задача 2
def find_homonymous_students(df: pd.DataFrame) -> Tuple[bool, int, pd.Series, str]:
    counts = df['surname'].value_counts()
    surnames = counts[counts>1].index
    homonyms = df[df['surname'].isin(surnames)].copy()
    per_course = homonyms.groupby('курс').size()
    if homonyms.empty:
        max_group = ''
    else:
        max_group = homonyms['группа'].value_counts().idxmax()
    """
    Проверяет наличие однофамильцев на ФСУиР, их количество, распределение по курсам
    и определяет группу с наибольшим числом однофамильцев.
    Возвращает:
     - логическое значение (наличие однофамильцев)
     - общее количество однофамильцев
     - серию с числом однофамильцев по курсам
     - группу с максимальным числом однофамильцев
    """
    return len(homonyms) > 0, len(homonyms), per_course, max_group


# Задача 3
def gender_identification(patronym: str) -> str:
    if pd.isna(patronym):
        return 'unknown'
    patronym = str(patronym).lower().strip()
    if re.search(r'(овна|евна|ична|инична)$', patronym):
        return 'female'
    elif re.search(r'(ович|евич|ич)$', patronym):
        return 'male'
    else:
        return 'unknown'
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    genders = df['patronim'].apply(gender_identification)
    without_patronym = (genders == 'unknown').sum()
    gender_counts = genders[genders != 'unknown'].value_counts()
    """
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
     - количество студентов без отчества
     - серию с распределением студентов по полу
    """
    return without_patronym, gender_counts


# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    counts = data['факультет'].value_counts()
    max_faculty = (counts.idxmax(), counts.max())
    min_faculty = (counts.idxmin(), counts.min())
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    return counts.to_frame('Количество студентов'), max_faculty, min_faculty


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    students = data.groupby('курс').size()
    mean_students = students.groupby(level=0).mean()
    median_students = students.groupby(level=0).median()
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    return mean_students, median_students


# Задача 6
def most_popular_name(data: pd.DataFrame) -> Tuple[str, str, str, int, float]:
    counts = data['name'].value_counts()
    popular_name = counts.idxmax()
    students = data[data['name'] == popular_name]
    groups = students.groupby(
        ['группа', 'факультет', 'курс']
    ).size()
    group, faculty, course = groups.idxmax()
    ratio = round(counts.max()/len(data), 2)
    """
    Определяет самое популярное имя, группу с наибольшим количеством студентов с этим именем,
    факультет, курс и долю таких студентов в общем числе.
    Возвращает результат в следующем порядке:
     1. самое частое имя
     2. группа
     3. факультет
     4. доля
    """
    return popular_name, group, faculty, course, ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    counts = data['name'].value_counts()
    res = data[
        data['name'].str.startswith('П', na = False)
        & data['name'].map(counts).eq(1)
    ]
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    return res[['фио', 'факультет', 'курс']]


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    third_course = data[data['курс'].astype(str).str.startswith('3')]
    averages = third_course.groupby('факультет')['средний_балл'].mean()
    faculty = averages.idxmax()
    students = third_course[third_course['факультет'] == faculty].copy()
    students['пол'] = students['patronim'].apply(
        gender_identification
    )
    gender_avg = students.groupby('пол')['средний_балл'].mean()
    gender_avg = gender_avg.drop('unknown', errors='ignore')
    gender = gender_avg.idxmax()
    if gender == 'female':
        gender = 'Девушки'
    else:
        gender = 'Парни'
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    return faculty, gender, round(gender_avg.max())


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data['ису'] = pd.to_numeric(data['ису'], errors='coerce')
    data = data.sort_values('ису').reset_index(drop=True)
    consecutive = (data['ису'].diff().eq(1) & data['курс'].eq(data['курс'].shift(1)))
    found = consecutive.rolling(4).sum().eq(4)
    if not found.any():
        return pd.DataFrame(columns = ['фио', 'факультет', 'курс', 'группа'])
    end = found[found].index[0]
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    return data.loc[
        end - 4:end,
        ['фио', 'факультет', 'курс', 'группа']
    ].reset_index(drop=True)

#if __name__ == "__main__":
data = pd.read_csv("isu_fake_data.csv")
fio = data['фио'].fillna("").str.split(n=2, expand=True)
data["surname"] = fio[0]
data["name"] = fio[1]
data["patronim"] = fio[2]

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
