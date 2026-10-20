import pandas as pd
import re
from typing import Tuple


FSUIR = "факультет систем управления и робототехники"


def add_name_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Разбивает столбец «фио» на фамилию, имя и отчество.
    Если отчества нет, в столбце patronim будет пустая строка.
    """
    data = data.copy()
    parts = data["фио"].str.split()
    data["surname"] = parts.str[0]
    data["name"] = parts.str[1]
    data["patronim"] = parts.str[2].fillna("")
    return data


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    fsuir = data[data["факультет"] == FSUIR].copy()
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
    if "surname" not in df.columns:
        df = add_name_columns(df)

    # однофамилецы
    homonyms = df[df.duplicated("surname", keep=False)]

    has_homonyms = not homonyms.empty
    total_homonyms = len(homonyms)
    # на каждом курсе: студенты, у которых есть однофамилец на этом же курсе
    same_course = df[df.duplicated(["курс", "surname"], keep=False)]
    homonyms_per_course = same_course["курс"].value_counts().sort_index()
    max_homonym_group = homonyms["группа"].value_counts().idxmax()
    return has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group


# Задача 3
def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    if re.search(r"(овна|евна|ична|инична)$", patronym):
        return "female"
    if re.search(r"(ич)$", patronym):
        return "male"
    return "unknown"


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
     - количество студентов без отчества
     - серию с распределением студентов по полу
    """
    if "patronim" not in df.columns:
        df = add_name_columns(df)

    genders = df["patronim"].apply(gender_identification)

    # без отчества — в ФИО только два слова
    students_without_patronym = int((df["patronim"] == "").sum())
    # у кого есть отчество: считаем парней и девушек
    gender_counts = genders[genders != "unknown"].value_counts()
    return students_without_patronym, gender_counts


# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    counts = data["факультет"].value_counts()
    faculty_counts = counts.rename_axis("факультет").reset_index(name="количество")

    max_faculty = (counts.idxmax(), int(counts.max()))
    min_faculty = (counts.idxmin(), int(counts.min()))
    return faculty_counts, max_faculty, min_faculty


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    # сколько студентов на каждом курсе каждого факультета
    per_faculty = data.groupby(["курс", "факультет"]).size()

    mean_students = per_faculty.groupby(level="курс").mean().round(1)
    median_students = per_faculty.groupby(level="курс").median()
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
    if "name" not in data.columns:
        data = add_name_columns(data)

    popular_name = data["name"].value_counts().idxmax()
    with_name = data[data["name"] == popular_name]

    name_group = with_name["группа"].value_counts().idxmax()
    group_row = with_name[with_name["группа"] == name_group].iloc[0]
    faculty = group_row["факультет"]
    course = group_row["курс"]

    name_ratio = round(len(with_name) / len(data), 2)
    return popular_name, name_group, faculty, course, name_ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    if "name" not in data.columns:
        data = add_name_columns(data)

    name_counts = data["name"].value_counts()
    unique_names = name_counts[name_counts == 1].index

    mask = data["name"].isin(unique_names) & data["name"].str.startswith("П")
    return data.loc[mask, ["фио", "факультет", "курс"]].reset_index(drop=True)


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    if "patronim" not in data.columns:
        data = add_name_columns(data)

    third = data[data["курс"] == "3-й"]
    fac = third.groupby("факультет")["средний_балл"].mean().idxmax()

    fac_third = third[third["факультет"] == fac]
    genders = fac_third["patronim"].apply(gender_identification)
    by_gender = fac_third[genders != "unknown"].groupby(genders)["средний_балл"].mean()

    best_gender = by_gender.idxmax()
    best_grade = int(round(by_gender.max()))
    return fac, best_gender, best_grade


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    df = data.sort_values("ису").reset_index(drop=True)

    step_is_one = df["ису"].diff() == 1
    end = df.index[step_is_one.rolling(4).sum() == 4][0]
    start = end - 4

    return df.loc[start:end, ["фио", "факультет", "курс", "группа", "ису"]]


if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")
    data = add_name_columns(data)

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
