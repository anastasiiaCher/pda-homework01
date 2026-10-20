import pandas as pd
import re
from typing import Tuple


def _ensure_fio_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Вспомогательная функция: добавляет столбцы surname, name, patronim, если их еще нет."""
    if not {"surname", "name", "patronim"}.issubset(df.columns):
        df = df.copy()
         fio_parts = df["фио"].str.split(" ", n=2, expand=True)
        df["surname"] = fio_parts[0]
        df["name"] = fio_parts[1]
        df["patronim"] = fio_parts[2] if 2 in fio_parts.columns else None
    return df


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    fsuir = data[
        data["факультет"].isin(
            [
                "факультет систем управления и робототехники",
                "Факультет систем управления и робототехники",
                "ФСУиР",
            ]
        )
    ]
    num_students = int(len(fsuir))
    num_groups = int(fsuir["группа"].nunique())
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
    df = _ensure_fio_columns(df)
    homonyms = df[df["surname"].duplicated(keep=False)]
    total_homonyms = int(len(homonyms))
    has_homonyms = bool(total_homonyms > 0)
    homonyms_per_course = homonyms["курс"].value_counts()
    max_homonym_group = str(homonyms["группа"].value_counts().idxmax())
    return has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group


# Задача 3
def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    if not isinstance(patronym, str) or pd.isna(patronym):
        return "unknown"
    patronym_clean = patronym.strip().lower()
    if re.search(r"(ович|евич|ич)$", patronym_clean):
        return "male"
    if re.search(r"(овна|евна|ична|инична)$", patronym_clean):
        return "female"
    return "unknown"


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
     - количество студентов без отчества
     - серию с распределением студентов по полу
    """
    df = _ensure_fio_columns(df)
    students_without_patronym = int(df["patronim"].isna().sum())
    genders = df["patronim"].dropna().apply(gender_identification)
    gender_counts = genders[genders != "unknown"].value_counts()
    return students_without_patronym, gender_counts


# Задача 4
def faculty_statistics(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    counts = data["факультет"].value_counts()
    max_faculty = (str(counts.idxmax()), int(counts.max()))
    min_faculty = (str(counts.idxmin()), int(counts.min()))
    return counts, max_faculty, min_faculty


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    course_faculty_counts = data.groupby(["курс", "факультет"]).size()
    mean_students = course_faculty_counts.groupby("курс").mean()
    median_students = course_faculty_counts.groupby("курс").median()
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
    data = _ensure_fio_columns(data)
    popular_name = str(data["name"].value_counts().idxmax())
    same_name_df = data[data["name"] == popular_name]
    name_group = str(same_name_df["группа"].value_counts().idxmax())
    group_row = same_name_df[same_name_df["группа"] == name_group].iloc[0]
    faculty = str(group_row["факультет"])
    course = int(group_row["курс"])
    name_ratio = round(len(same_name_df) / len(data), 2)
    return popular_name, name_group, faculty, course, name_ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    data = _ensure_fio_columns(data)
    starts_with_p = data[data["name"].str.startswith("П", na=False)]
    name_counts = starts_with_p["name"].value_counts()
    unique_p_names = name_counts[name_counts == 1].index
    result = starts_with_p[starts_with_p["name"].isin(unique_p_names)][
        ["фио", "факультет", "курс"]
    ]
    return result


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    data = _ensure_fio_columns(data)
    third_course = data[data["курс"] == 3].copy()
    best_faculty = str(
        third_course.groupby("факультет")["средний_балл"].mean().idxmax()
    )
    fac_students = third_course[third_course["факультет"] == best_faculty].copy()
    fac_students["gender"] = fac_students["patronim"].apply(gender_identification)
    fac_students = fac_students[fac_students["gender"].isin(["male", "female"])]
    gender_means = fac_students.groupby("gender")["средний_балл"].mean()
    best_gender = str(gender_means.idxmax())
    best_grade = round(float(gender_means.max()))
    return best_faculty, best_gender, best_grade


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    sorted_data = data.sort_values("ису").reset_index(drop=True)
    isu = sorted_data["ису"]
    is_consecutive = (
        (isu.shift(-1) - isu == 1)
        & (isu.shift(-2) - isu == 2)
        & (isu.shift(-3) - isu == 3)
        & (isu.shift(-4) - isu == 4)
    )
    start_idx = int(is_consecutive.idxmax())
    return sorted_data.iloc[start_idx : start_idx + 5][
        ["фио", "факультет", "курс", "группа"]
    ]


if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")
    fio_split = data["фио"].str.split(" ", n=2, expand=True)
    data["surname"] = fio_split[0]
    data["name"] = fio_split[1]
    data["patronim"] = fio_split[2]

    # Задача 1
    num_students, num_groups, fsuir = filter_fsuir_students(data)
    print(f"Студентов на ФСУиР: {num_students}, Групп: {num_groups}")

    # Задача 2
    has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group = (
        find_homonymous_students(fsuir)
    )
    print(
        f"Есть однофамильцы: {has_homonyms}, Всего: {total_homonyms}, Группа с максимумом: {max_homonym_group}"
    )
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
    print(
        f"Самое популярное имя: {popular_name}, Группа: {name_group}, Факультет: {faculty}, Курс: {course}"
    )
    print(f"Доля студентов с этим именем: {name_ratio}")

    # Задача 7
    result_7 = find_students_with_name_starting_P(data)
    print("Студенты с именем, начинающимся на П и встречающимся ровно один раз:")
    print(result_7)

    # Задача 8
    fac, best_gender, best_grade = highest_avg_grade_faculty(data)
    print(f"Факультет с высоким средним баллом 3-го курса: {fac}")
    print(
        f"Пол с наивысшим средним баллом: {best_gender}, Средний балл: {best_grade}"
    )

    # Задача 9
    result_9 = find_consecutive_students(data)
    print("Первые 5 студентов с подряд идущими табельными номерами:")
    print(result_9)
