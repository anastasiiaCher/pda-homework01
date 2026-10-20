import pandas as pd
import re
from typing import Tuple


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    fsuir = data[data["факультет"] == "факультет систем управления и робототехники"].copy()
    return len(fsuir), fsuir["группа"].nunique(), fsuir


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
    surname_counts = df.groupby("surname")["surname"].transform("size")
    is_homonym = surname_counts > 1
    total_homonyms = int(is_homonym.sum())

    course_counts = df.groupby(["курс", "surname"])["surname"].transform("size")
    homonyms_per_course = df.loc[course_counts > 1].groupby("курс").size()

    group_counts = df.groupby(["группа", "surname"])["surname"].transform("size")
    homonyms_per_group = df.loc[group_counts > 1].groupby("группа").size()
    max_homonym_group = homonyms_per_group.idxmax()

    return total_homonyms > 0, total_homonyms, homonyms_per_course, max_homonym_group
    

# Задача 3
def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    if not isinstance(patronym, str) or not patronym.strip():
        return "unknown"

    value = patronym.strip().lower()
    if value.endswith(("овна", "евна", "инична", "ична")):
        return "female"
    if value.endswith(("ович", "евич", "ич")):
        return "male"
    return "unknown"


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
     - количество студентов без отчества
     - серию с распределением студентов по полу 
    """
    patronyms = df["patronim"].fillna("").astype(str).str.strip()
    without_patronym = patronyms == ""
    gender_counts = patronyms[~without_patronym].map(gender_identification).value_counts()
    return int(without_patronym.sum()), gender_counts


# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    faculty_counts = (
        data["факультет"]
        .value_counts()
        .rename_axis("факультет")
        .reset_index(name="количество")
    )
    largest = faculty_counts.iloc[0]
    smallest = faculty_counts.iloc[-1]
    return (
        faculty_counts,
        (largest["факультет"], int(largest["количество"])),
        (smallest["факультет"], int(smallest["количество"])),
    )


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    students_per_faculty = data.groupby(["курс", "факультет"]).size()
    mean_students = students_per_faculty.groupby("курс").mean()
    median_students = students_per_faculty.groupby("курс").median()
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
    popular_name = data["name"].value_counts().idxmax()
    named = data[data["name"] == popular_name]
    name_group = named["группа"].value_counts().idxmax()
    group_info = named.loc[named["группа"] == name_group].iloc[0]
    name_ratio = round(len(named) / len(data), 2)
    return popular_name, name_group, group_info["факультет"], group_info["курс"], name_ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    name_counts = data["name"].value_counts()
    unique_names = name_counts[name_counts == 1].index
    mask = data["name"].isin(unique_names) & data["name"].str.startswith("П")
    return data.loc[mask, ["фио", "факультет", "курс"]]


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    third_year = data[data["курс"] == "3-й"]
    faculty = third_year.groupby("факультет")["средний_балл"].mean().idxmax()

    faculty_students = third_year[third_year["факультет"] == faculty].copy()
    faculty_students["gender"] = faculty_students["patronim"].map(gender_identification)
    gender_means = (
        faculty_students[faculty_students["gender"].isin(["female", "male"])]
        .groupby("gender")["средний_балл"]
        .mean()
    )
    best_gender = gender_means.idxmax()
    return faculty, best_gender, int(round(gender_means.loc[best_gender]))


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    ordered = data.sort_values("ису")
    run_id = ordered["ису"].diff().ne(1).cumsum()
    run_sizes = run_id.groupby(run_id).transform("size")
    first_run_id = run_id[run_sizes >= 5].iloc[0]
    return ordered.loc[run_id == first_run_id, ["фио", "факультет", "курс", "группа", "ису"]].head(5)


if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")
    # data["surname"] = "put your code here"
    # data["name"] = "put your code here"
    # data["patronim"] = "put your code here"
    
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
