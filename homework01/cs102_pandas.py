"""Лабораторная работа 1. Обработка данных студентов с помощью pandas.

Задание: https://github.com/anastasiiaCher/pda-homework01
Автор: Мертехин Михаил.
Поток: ВизМод 3.1.
Инструкция по сдаче:
https://docs.google.com/document/d/1YRIsBK4yLrC8wQttdxeaVfz5hsCCvyem00035ew4oFo/edit
Установка: python -m pip install pandas==2.2.2
Запуск: python cs102_pandas.py

Положите исходный isu_fake_data.csv рядом с программой. Если файла нет,
данные читаются из открытого репозитория, поэтому нужен интернет.

Для сдачи этот файл заменяет homework01/cs102_pandas.py в личном форке.
Название PR: «Лабораторная работа 1 — ВизМод 3.1 — Мертехин Михаил».
"""

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_data() -> pd.DataFrame:
    path = Path(__file__).with_name("isu_fake_data.csv")
    if not path.exists():
        path = (
            "https://raw.githubusercontent.com/anastasiiaCher/pda-homework01/"
            "602e875160b743cb1dedc19c26d5c871d0b9f07c/homework01/isu_fake_data.csv"
        )
    data = pd.read_csv(path)
    fio_parts = data["фио"].str.split()
    data["surname"] = fio_parts.str[0]
    data["name"] = fio_parts.str[1]
    data["patronim"] = fio_parts.str[2].fillna("")
    return data


# Задача 1. Студенты и группы ФСУиР.
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    fsuir = data[
        data["факультет"] == "факультет систем управления и робототехники"
    ].copy()
    return len(fsuir), fsuir["группа"].nunique(), fsuir


# Задача 2. Однофамильцы на ФСУиР.
def find_homonymous_students(df: pd.DataFrame) -> Tuple[bool, int, pd.Series, str]:
    homonyms = df[df["surname"].duplicated(keep=False)]
    per_course = homonyms.groupby("курс").size()
    max_group = homonyms["группа"].value_counts().idxmax() if not homonyms.empty else ""
    return not homonyms.empty, len(homonyms), per_course, max_group


# Задача 3. Пол по окончаниям отчества.
def gender_identification(patronym: str) -> str:
    if not isinstance(patronym, str):
        return "unknown"
    patronym = patronym.strip().lower()
    if patronym.endswith(("овна", "евна", "ична", "инична")):
        return "female"
    if patronym.endswith(("ович", "евич", "ич")):
        return "male"
    return "unknown"


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    without_patronym = df["patronim"].eq("")
    genders = df.loc[~without_patronym, "patronim"].apply(gender_identification)
    gender_counts = genders.value_counts().reindex(
        ["male", "female", "unknown"], fill_value=0
    )
    return int(without_patronym.sum()), gender_counts


# Задача 4. Численность факультетов.
def faculty_statistics(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    counts = data["факультет"].value_counts()
    table = counts.rename_axis("факультет").reset_index(name="число_студентов")
    max_faculty = (counts.idxmax(), int(counts.max()))
    min_faculty = (counts.idxmin(), int(counts.min()))
    return table, max_faculty, min_faculty


# Задача 5. Средняя и медианная численность курса по факультетам.
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    counts = data.groupby(["факультет", "курс"]).size()
    per_course = counts.groupby("курс")
    return per_course.mean(), per_course.median()


# Задача 6. Популярное имя и его доля во всём датасете.
def most_popular_name(data: pd.DataFrame) -> Tuple[str, str, str, str, float]:
    popular_name = data["name"].value_counts().idxmax()
    students = data[data["name"] == popular_name]
    group = students["группа"].value_counts().idxmax()
    group_info = students[students["группа"] == group].iloc[0]
    ratio = round(len(students) / len(data), 2)
    return popular_name, group, group_info["факультет"], group_info["курс"], ratio


# Задача 7. Имена на П, встретившиеся ровно один раз.
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    name_counts = data["name"].value_counts()
    mask = data["name"].str.startswith("П") & data["name"].map(name_counts).eq(1)
    return data.loc[mask, ["фио", "факультет", "курс"]].copy()


# Задача 8. Лучший средний балл третьего курса.
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    third_course = data[data["курс"] == "3-й"]
    faculty_means = third_course.groupby("факультет")["средний_балл"].mean()
    faculty = faculty_means.idxmax()
    students = third_course[third_course["факультет"] == faculty].copy()
    students["gender"] = students["patronim"].apply(gender_identification)
    known_gender = students[students["gender"].isin(["female", "male"])]
    gender_means = known_gender.groupby("gender")["средний_балл"].mean()
    best_gender = gender_means.idxmax()
    return faculty, best_gender, int(round(gender_means.loc[best_gender]))


# Задача 9. Первая по возрастанию цепочка из пяти номеров подряд.
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    students = data.sort_values("ису").drop_duplicates("ису").copy()
    sequence = students["ису"].diff().ne(1).cumsum()
    sequence_lengths = students.groupby(sequence)["ису"].transform("size")
    columns = ["ису", "фио", "факультет", "курс", "группа"]
    return students.loc[sequence_lengths >= 5, columns].head(5).copy()


def main() -> None:
    data = load_data()
    print(f"Всего записей: {len(data)}, факультетов: {data['факультет'].nunique()}")

    print("\nЗадача 1. ФСУиР")
    num_students, num_groups, fsuir = filter_fsuir_students(data)
    print(f"Студентов: {num_students}, групп: {num_groups}")

    print("\nЗадача 2. Однофамильцы")
    has_homonyms, total, per_course, group = find_homonymous_students(fsuir)
    print(f"Есть однофамильцы: {has_homonyms}, всего студентов: {total}")
    print(per_course.to_string())
    print(f"Группа с максимумом: {group}")

    print("\nЗадача 3. Отчества")
    without_patronym, gender_counts = analyze_patronyms(fsuir)
    print(f"Без третьего слова в ФИО: {without_patronym}")
    print("Среди записей с третьим словом:")
    print(gender_counts.to_string())
    print("male — парни, female — девушки, unknown — пол не определён по отчеству")

    print("\nЗадача 4. Численность факультетов")
    faculty_counts, max_faculty, min_faculty = faculty_statistics(data)
    print(faculty_counts.to_string(index=False))
    print(f"Больше всего: {max_faculty[0]} — {max_faculty[1]}")
    print(f"Меньше всего: {min_faculty[0]} — {min_faculty[1]}")

    print("\nЗадача 5. Численность курсов по факультетам")
    mean_students, median_students = course_statistics(data)
    statistics = pd.DataFrame(
        {"среднее": mean_students, "медиана": median_students}
    )
    print(statistics.to_string())

    print("\nЗадача 6. Самое популярное имя")
    name, group, faculty, course, ratio = most_popular_name(data)
    name_count = int(data["name"].eq(name).sum())
    percent = name_count / len(data) * 100
    print(f"Имя: {name}, студентов: {name_count}")
    print(f"Группа: {group}, факультет: {faculty}, курс: {course}")
    print(f"Доля, округлённая до двух знаков: {ratio:.2f}")
    print(f"Та же исходная доля в процентах: {percent:.2f}%")

    print("\nЗадача 7. Уникальные имена на П")
    result_7 = find_students_with_name_starting_P(data)
    print(result_7.to_string(index=False))
    print(f"Найдено: {len(result_7)}")

    print("\nЗадача 8. Средний балл третьего курса")
    faculty, gender, grade = highest_avg_grade_faculty(data)
    print(f"Факультет: {faculty}")
    print(f"Пол с более высоким средним баллом: {gender}, средний балл: {grade}")

    print("\nЗадача 9. Пять студентов с последовательными номерами")
    result_9 = find_consecutive_students(data)
    print(result_9[["фио", "факультет", "курс", "группа"]].to_string(index=False))
    print("Номера ИСУ для проверки:", result_9["ису"].tolist())


if __name__ == "__main__":
    main()
