import pandas as pd
from typing import Tuple


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    fsuir = data[data["факультет"] == "факультет систем управления и робототехники"]
    num_students = len(fsuir)
    num_groups = fsuir["группа"].nunique()
    return num_students, num_groups, fsuir


# Задача 2
def find_homonymous_students(df: pd.DataFrame) -> Tuple[bool, int, pd.Series, str]:
    homonyms = df[df.duplicated("surname", keep=False)]
    total_homonyms = len(homonyms)
    has_homonyms = total_homonyms > 0

    # однофамильцы внутри курса: фамилия повторяется на этом же курсе
    course_homonyms = df[df.duplicated(["surname", "курс"], keep=False)]
    homonyms_per_course = course_homonyms["курс"].value_counts().sort_index()

    max_homonym_group = homonyms["группа"].value_counts().idxmax()
    return has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group


# Задача 3
def gender_identification(patronym: str) -> str:
    if patronym.endswith("ич"):
        return "male"
    if patronym.endswith("на"):
        return "female"
    return "unknown"


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    students_without_patronym = int((df["patronim"] == "").sum())

    with_patronym = df[df["patronim"] != ""]
    gender_counts = with_patronym["patronim"].apply(gender_identification).value_counts()
    return students_without_patronym, gender_counts


# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    counts = data["факультет"].value_counts()
    faculty_counts = counts.reset_index()

    max_faculty = (counts.idxmax(), int(counts.max()))
    min_faculty = (counts.idxmin(), int(counts.min()))
    return faculty_counts, max_faculty, min_faculty


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    counts = data.groupby(["факультет", "курс"]).size()

    mean_students = counts.groupby("курс").mean().round(1)
    median_students = counts.groupby("курс").median()
    return mean_students, median_students


# Задача 6
def most_popular_name(data: pd.DataFrame) -> Tuple[str, str, str, str, float]:
    popular_name = data["name"].value_counts().idxmax()
    with_name = data[data["name"] == popular_name]

    name_group = with_name["группа"].value_counts().idxmax()

    group_row = data[data["группа"] == name_group].iloc[0]
    faculty = group_row["факультет"]
    course = group_row["курс"]

    name_ratio = round(len(with_name) / len(data), 2)
    return popular_name, name_group, faculty, course, name_ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    name_counts = data["name"].value_counts()
    unique_names = name_counts[name_counts == 1].index

    students = data[data["name"].isin(unique_names) & data["name"].str.startswith("П")]
    return students[["фио", "факультет", "курс"]]


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    third_course = data[data["курс"] == "3-й"]
    fac = third_course.groupby("факультет")["средний_балл"].mean().idxmax()

    students = third_course[third_course["факультет"] == fac]
    gender = students["patronim"].apply(gender_identification)

    # пол определяем по отчеству, студентов с неизвестным полом не учитываем
    known = students[gender != "unknown"]
    grades = known.groupby(gender)["средний_балл"].mean()

    best_gender = grades.idxmax()
    best_grade = int(round(grades.max()))
    return fac, best_gender, best_grade


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    students = data.sort_values("ису").drop_duplicates("ису")

    # если номер через 4 строки больше ровно на 4, значит 5 номеров идут подряд
    is_start = students["ису"].shift(-4) - students["ису"] == 4
    start = students[is_start]["ису"].iloc[0]

    result = students[students["ису"].between(start, start + 4)]
    return result[["фио", "факультет", "курс", "группа", "ису"]]


if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")
    fio = data["фио"].str.split()
    data["surname"] = fio.str[0]
    data["name"] = fio.str[1]
    data["patronim"] = fio.str[2].fillna("")

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
