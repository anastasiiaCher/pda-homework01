import pandas as pd
from typing import Tuple

male_end = ("ович", "евич", "ич")
fem_end = ("овна", "евна", "ична", "инична")

def add_name_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Разбивает «фио» на surname, name, patronim.
    Всё, что идёт после имени, считается отчеством (в т.ч. часть иностранного имени)."""
    parts = data["фио"].astype(str).str.strip().str.split(n=2, expand=True)
    parts = parts.reindex(columns=[0, 1, 2])
    data["surname"] = parts[0]
    data["name"] = parts[1]
    data["patronim"] = parts[2].fillna("").str.strip()
    return data

# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
   mask = data["факультет"].str.contains("робототехник", case=False, na=False)
    fsuir = data[mask].copy()
    return len(fsuir), fsuir["группа"].nunique(), fsuir


# Задача 2
def find_homonymous_students(df: pd.DataFrame) -> Tuple[bool, int, pd.Series, str]:
    mask = df["surname"].duplicated(keep=False)
    homonyms = df[mask]
    has_homonyms = bool(mask.any())
    total = int(mask.sum())
    per_course = homonyms.groupby("курс").size()
    max_group = homonyms["группа"].value_counts().idxmax() if has_homonyms else ""
    return has_homonyms, total, per_course, max_group

    

# Задача 3
def gender_identification(patronym: str) -> str:
    if not isinstance(patronym, str) or not patronym.strip():
        return "unknown"
    p = patronym.strip().lower()
    if p.endswith(FEMALE_ENDINGS):
        return "female"
    if p.endswith(MALE_ENDINGS):
        return "male"
    return "unknown"



def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    no_patronym = df["patronim"].str.strip() == ""
    with_patronym = df[~no_patronym]
    gender_counts = with_patronym["patronim"].apply(gender_identification).value_counts()
    return int(no_patronym.sum()), gender_counts


# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    counts = data.groupby("факультет").size().reset_index(name="количество")
    max_row = counts.loc[counts["количество"].idxmax()]
    min_row = counts.loc[counts["количество"].idxmin()]
    return (
        counts,
        (max_row["факультет"], int(max_row["количество"])),
        (min_row["факультет"], int(min_row["количество"])),
    )


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        group_sizes = data.groupby(["курс", "группа"]).size()
    by_course = group_sizes.groupby(level="курс")
    return by_course.mean(), by_course.median()


# Задача 6
def most_popular_name(data: pd.DataFrame) -> Tuple[str, str, str, int, float]:
    name_counts = data["name"].value_counts()
    popular = name_counts.idxmax()
    with_name = data[data["name"] == popular]
    best_group = with_name["группа"].value_counts().idxmax()
    row = data[data["группа"] == best_group].iloc[0]
    ratio = round(name_counts[popular] / len(data) * 100, 2)  # в процентах
    return popular, best_group, row["факультет"], int(row["курс"]), ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    counts = data["name"].value_counts()
    unique_names = counts[counts == 1].index
    mask = data["name"].isin(unique_names) & data["name"].str.startswith("П")
    return data.loc[mask, ["фио", "факультет", "курс"]]


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    third = data[data["курс"] == 3]
    fac = third.groupby("факультет")["средний_балл"].mean().idxmax()
    sub = third[third["факультет"] == fac].copy()
    sub["пол"] = sub["patronim"].apply(gender_identification)
    by_gender = sub[sub["пол"].isin(["male", "female"])].groupby("пол")["средний_балл"].mean()
    return fac, by_gender.idxmax(), int(round(by_gender.max()))

# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    s = data.sort_values("ису")
    new_run = (s["ису"].diff() != 1) | (s["курс"] != s["курс"].shift())
    run_id = new_run.cumsum()
    run_len = run_id.map(run_id.value_counts())
    candidates = run_id[run_len >= 5]
    if candidates.empty:
        return s.iloc[0:0][["фио", "факультет", "курс", "группа"]]
    first_run = candidates.iloc[0]
    return s[run_id == first_run].head(5)[["фио", "факультет", "курс", "группа"]]


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
    print(f"Факультет с наибольшим числом студентов: {max_faculty}")
    print(f"Факультет с наименьшим числом студентов: {min_faculty}")

    # Задача 5
    mean_students, median_students = course_statistics(data)
    print("Среднее число студентов на курсах:", mean_students)
    print("Медианное число студентов на курсах:", median_students)

    # Задача 6
    popular_name, name_group, faculty, course, name_ratio = most_popular_name(data)
    print(f"Самое популярное имя: {popular_name}, Группа: {name_group}, Факультет: {faculty}, Курс: {course}")
    print(f"Доля студентов с этим именем: {name_ratio}%")

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
