import pandas as pd
import re
from typing import Tuple


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    fsuir = data[data["факультет"] == "факультет систем управления и робототехники"]
    num_students = len(fsuir)
    num_groups = fsuir["группа"].nunique()
    return num_students, num_groups, fsuir


# Задача 2
def find_homonymous_students(df: pd.DataFrame) -> Tuple[bool, int, pd.Series, str]:
    surname_counts = df["surname"].value_counts()
    homonym_surnames = surname_counts[surname_counts > 1].index

    homonyms_df = df[df["surname"].isin(homonym_surnames)]

    has_homonyms = len(homonyms_df) > 0
    total_homonyms = len(homonyms_df)

    homonyms_per_course = homonyms_df.groupby("курс").size()

    max_homonym_group = (
        homonyms_df.groupby("группа").size().idxmax()
        if not homonyms_df.empty else None
    )

    return has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group
    

# Задача 3
def gender_identification(patronym: str) -> str:
    if not isinstance(patronym, str) or patronym.strip() == "":
        return "unknown"
    p = patronym.strip().lower()
    if re.search(r"(овна|евна|ична|инична)$", p):
        return "female"
    if re.search(r"(ович|евич|ич)$", p):
        return "male"
    return "unknown"


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    without_patronym = df["patronim"].isna() | (df["patronim"].astype(str).str.strip() == "")
    num_without = int(without_patronym.sum())

    with_patronym = df[~without_patronym].copy()
    with_patronym["gender"] = with_patronym["patronim"].apply(gender_identification)
    gender_counts = with_patronym["gender"].value_counts()

    return num_without, gender_counts


# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    counts = data["факультет"].value_counts()
    faculty_counts = counts.reset_index()
    faculty_counts.columns = ["факультет", "количество"]

    max_faculty = (counts.idxmax(), int(counts.max()))
    min_faculty = (counts.idxmin(), int(counts.min()))

    return faculty_counts, max_faculty, min_faculty


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    per_course = data.groupby("курс").size()
    mean_students = pd.Series([per_course.mean()], index=["среднее"])
    median_students = pd.Series([per_course.median()], index=["медиана"])
    return mean_students, median_students


# Задача 6
def most_popular_name(data: pd.DataFrame) -> Tuple[str, str, str, int, float]:
    name_counts = data["name"].value_counts()
    popular_name = name_counts.idxmax()

    same_name = data[data["name"] == popular_name]

    group_counts = same_name.groupby("группа").size()
    name_group = group_counts.idxmax()

    group_row = same_name[same_name["группа"] == name_group].iloc[0]
    faculty = group_row["факультет"]
    course = group_row["курс"]

    ratio = round(len(same_name) / len(data), 2)

    return popular_name, name_group, faculty, course, ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    name_counts = data["name"].value_counts()
    unique_names = name_counts[name_counts == 1].index

    mask = data["name"].isin(unique_names) & data["name"].astype(str).str.startswith("П")
    result = data.loc[mask, ["фио", "факультет", "курс"]]
    return result


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    third = data[data["курс"] == "3-й"].copy()

    fac_avg = third.groupby("факультет")["средний_балл"].mean()
    best_faculty = fac_avg.idxmax()

    best_fac_students = third[third["факультет"] == best_faculty].copy()
    best_fac_students["gender"] = best_fac_students["patronim"].apply(gender_identification)

    gender_avg = best_fac_students.groupby("gender")["средний_балл"].mean()

    gender_avg = gender_avg[gender_avg.index.isin(["male", "female"])]
    best_gender = gender_avg.idxmax()
    best_grade = int(round(gender_avg.max()))

    return best_faculty, best_gender, best_grade


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    
    pass


if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")

    def split_fio(fio: str):
    if pd.isna(fio):
        return pd.Series([None, None, None])
    parts = fio.strip().split()
    surname = parts[0] if len(parts) > 0 else None
    name = parts[1] if len(parts) > 1 else None
    patronym = " ".join(parts[2:]) if len(parts) > 2 else None
    return pd.Series([surname, name, patronym])

    data[["surname", "name", "patronim"]] = data["фио"].apply(split_fio)
    
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
