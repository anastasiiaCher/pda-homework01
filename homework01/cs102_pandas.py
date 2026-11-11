import pandas as pd
import re
from typing import Tuple

FSUIR = "факультет систем управления и робототехники"


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    fsuir = data[data["факультет"] == FSUIR]

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
    odnofam = df[df.duplicated("surname", keep=False)]
    total_odnofam = len(odnofam)
    has_odnofam = total_odnofam > 0

    odnofam_in_course = df[df.duplicated(["курс", "surname"], keep=False)]
    odnofam_per_course = odnofam_in_course.groupby("курс").size()

    max_odnofam_group = odnofam.groupby("группа").size().idxmax()

    return has_odnofam, total_odnofam, odnofam_per_course, max_odnofam_group


# Задача 3
def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    if patronym.endswith("ич"):
        return "male"
    if patronym.endswith(("овна", "евна", "ична")):
        return "female"
    return "unknown"


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
     - количество студентов без отчества
     - серию с распределением студентов по полу
    """
    without_patronym = df[df["patronim"] == ""]
    students_without_patronym = len(without_patronym)

    with_patronym = df[df["patronim"] != ""]
    gender = with_patronym["patronim"].apply(gender_identification)
    gender_counts = gender.value_counts()

    return students_without_patronym, gender_counts


# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    counts = data.groupby("факультет").size()

    max_faculty = (counts.idxmax(), int(counts.max()))
    min_faculty = (counts.idxmin(), int(counts.min()))

    faculty_counts = counts.reset_index(name="число_студентов")
    return faculty_counts, max_faculty, min_faculty


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    students = data.groupby(["курс", "факультет"]).size()

    mean_students = students.groupby("курс").mean().round(1)
    median_students = students.groupby("курс").median()
    return mean_students, median_students


# Задача 6
def most_popular_name(data: pd.DataFrame) -> Tuple[str, str, str, str, float]:
    """
    Определяет самое популярное имя, группу с наибольшим количеством студентов с этим именем,
    факультет, курс и долю таких студентов в общем числе.
    Возвращает результат в следующем порядке:
     1. самое частое имя
     2. группа
     3. факультет
     4. курс
     5. доля
    """
    popular_name = data.groupby("name").size().idxmax()

    teski = data[data["name"] == popular_name]
    name_group = teski.groupby("группа").size().idxmax()

    first_student = data[data["группа"] == name_group].iloc[0]
    faculty = first_student["факультет"]
    course = first_student["курс"]

    name_ratio = round(len(teski) / len(data), 2)
    return popular_name, name_group, faculty, course, name_ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    is_unique = ~data.duplicated("name", keep=False)
    starts_with_p = data["name"].str.startswith("П")

    result = data[is_unique & starts_with_p]
    return result[["фио", "факультет", "курс"]]


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    third_course = data[data["курс"] == "3-й"]

    faculty_grades = third_course.groupby("факультет")["средний_балл"].mean()
    fac = faculty_grades.idxmax()

    students = third_course[third_course["факультет"] == fac].copy()
    students["gender"] = students["patronim"].apply(gender_identification)
    students = students[students["gender"] != "unknown"]

    gender_grades = students.groupby("gender")["средний_балл"].mean()
    best_gender = gender_grades.idxmax()
    best_grade = int(round(gender_grades.max()))
    return fac, best_gender, best_grade


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    students = data.sort_values("ису").drop_duplicates("ису").reset_index(drop=True)

    is_end_of_run = students["ису"].diff(4) == 4

    end = is_end_of_run.idxmax()
    result = students.iloc[end - 4 : end + 1]
    return result[["фио", "факультет", "курс", "группа", "ису"]]


if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")
    fio_parts = data["фио"].str.split()
    data["surname"] = fio_parts.str[0]
    data["name"] = fio_parts.str[1]
    data["patronim"] = fio_parts.str[2].fillna("")

    # Задача 1
    num_students, num_groups, fsuir = filter_fsuir_students(data)
    print(f"Студентов на ФСУиР: {num_students}, Групп: {num_groups}")

    # Задача 2
    has_odnofam, total_odnofam, odnofam_per_course, max_odnofam_group = find_homonymous_students(fsuir)
    print(f"Есть однофамильцы: {has_odnofam}, Всего: {total_odnofam}, Группа с максимумом: {max_odnofam_group}")
    print(f"На каждом курсе: {odnofam_per_course}")

    # Задача 3
    students_without_patronym, gender_counts = analyze_patronyms(fsuir)
    print(f"Студентов без отчества: {students_without_patronym}")
    print("Распределение по полу:", gender_counts)

    # Задача 4
    faculty_counts, max_faculty, min_faculty = faculty_statistics(data)
    print(faculty_counts)
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
