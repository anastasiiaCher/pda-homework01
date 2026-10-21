
import pandas as pd
from typing import Tuple


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Разделяет ФИО на фамилию, имя и отчество."""
    data = data.copy()

    fio = data["фио"].fillna("").astype(str).str.strip()
    parts = fio.str.split(r"\s+", n=2, expand=True)

    data["surname"] = parts[0].fillna("")
    data["name"] = parts[1].fillna("") if 1 in parts.columns else ""
    data["patronim"] = parts[2].fillna("") if 2 in parts.columns else ""

    data["ису_num"] = pd.to_numeric(data["ису"], errors="coerce")
    data["средний_балл"] = pd.to_numeric(
        data["средний_балл"], errors="coerce"
    )
    return data


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    faculty = data["факультет"].astype(str).str.lower()
    mask = (
        faculty.str.contains("систем управления и робототехники", regex=False)
        | faculty.str.contains("фсуир", regex=False)
    )
    fsuir = data.loc[mask].copy()
    return len(fsuir), fsuir["группа"].nunique(), fsuir


# Задача 2
def find_homonymous_students(
    df: pd.DataFrame
) -> Tuple[bool, int, pd.Series, str]:
    # Фамилии, которые встречаются более одного раза
    counts = df["surname"].value_counts()
    repeated = counts[counts > 1].index
    homonyms = df[df["surname"].isin(repeated)].copy()

    total = len(homonyms)
    per_course = homonyms.groupby("курс").size().sort_index()

    if homonyms.empty:
        max_group = ""
    else:
        group_counts = homonyms.groupby("группа").size()
        max_group = str(group_counts.idxmax())

    return total > 0, total, per_course, max_group


# Задача 3
def gender_identification(patronym: str) -> str:
    if not isinstance(patronym, str):
        return "unknown"

    patronym = patronym.strip().lower()
    if not patronym:
        return "unknown"

    if patronym.endswith(("овна", "евна", "ична", "инична")):
        return "female"
    if patronym.endswith(("ович", "евич", "ич")):
        return "male"

    return "unknown"


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    patronym = df["patronim"].fillna("").astype(str).str.strip()
    without_patronym = int(patronym.eq("").sum())

    gender = patronym.apply(gender_identification)
    gender_counts = gender[gender != "unknown"].value_counts()
    gender_counts = gender_counts.reindex(
        ["female", "male"], fill_value=0
    )

    return without_patronym, gender_counts


# Задача 4
def faculty_statistics(
    data: pd.DataFrame
) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    counts = data.groupby("факультет").size().sort_values(ascending=False)
    faculty_counts = counts.to_frame(name="количество_студентов")

    max_faculty = (str(counts.idxmax()), int(counts.max()))
    min_faculty = (str(counts.idxmin()), int(counts.min()))

    return faculty_counts, max_faculty, min_faculty


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    # Сначала считаем студентов в каждой группе,
    # затем среднее и медиану размера групп на каждом курсе.
    group_sizes = data.groupby(["курс", "группа"]).size()
    mean_students = group_sizes.groupby(level="курс").mean()
    median_students = group_sizes.groupby(level="курс").median()

    return mean_students, median_students


# Задача 6
def most_popular_name(
    data: pd.DataFrame
) -> Tuple[str, str, str, int, float]:
    name_counts = data["name"].replace("", pd.NA).dropna().value_counts()

    if name_counts.empty:
        raise ValueError("В датасете нет имён.")

    popular_name = str(name_counts.index[0])
    named_students = data[data["name"] == popular_name]

    group_counts = named_students.groupby("группа").size()
    name_group = group_counts.idxmax()

    # Получаем факультет и курс этой группы
    group_row = named_students[named_students["группа"] == name_group].iloc[0]
    faculty = str(group_row["факультет"])
    course = int(str(group_row["курс"]).split("-")[0])

    name_ratio = round(len(named_students) / len(data), 2) if len(data) else 0.0

    return popular_name, str(name_group), faculty, course, name_ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    name_counts = data["name"].value_counts()
    mask = (
        data["name"].str.startswith("П", na=False)
        & data["name"].map(name_counts).eq(1)
    )

    return data.loc[mask, ["фио", "факультет", "курс"]].copy()


# Задача 8
def highest_avg_grade_faculty(
    data: pd.DataFrame
) -> Tuple[str, str, int]:
    third_year = data[data["курс"].astype(str).str.startswith("3")].copy()

    faculty_means = third_year.groupby("факультет")["средний_балл"].mean()
    faculty = faculty_means.idxmax()

    faculty_students = third_year[third_year["факультет"] == faculty].copy()
    faculty_students["gender"] = faculty_students["patronim"].apply(
        gender_identification
    )

    gender_means = faculty_students[
        faculty_students["gender"].isin(["female", "male"])
    ].groupby("gender")["средний_балл"].mean()

    if gender_means.empty:
        return str(faculty), "unknown", 0

    best_gender = str(gender_means.idxmax())
    best_grade = int(round(gender_means.max()))

    return str(faculty), best_gender, best_grade


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["фио", "факультет", "курс", "группа"]

    df = data.copy()
    df["ису_num"] = pd.to_numeric(df["ису"], errors="coerce")
    df = df.dropna(subset=["ису_num"]).sort_values("ису_num").reset_index(drop=True)

    # Ищем начало последовательности из пяти номеров.
    consecutive = pd.Series(True, index=df.index)

    for offset in range(1, 5):
        consecutive &= df["ису_num"].shift(-offset).sub(df["ису_num"]).eq(offset)
        consecutive &= df["курс"].shift(-offset).eq(df["курс"])

    starts = df.index[consecutive]

    if len(starts) == 0:
        return pd.DataFrame(columns=columns)

    start = starts[0]
    return df.loc[start:start + 4, columns].copy()


if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")
    data = prepare_data(data)

    # Задача 1
    num_students, num_groups, fsuir = filter_fsuir_students(data)
    print(f"Студентов на ФСУиР: {num_students}, групп: {num_groups}")

    # Задача 2
    has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group = (
        find_homonymous_students(fsuir)
    )
    print(
        f"Есть однофамильцы: {has_homonyms}, всего: {total_homonyms}, "
        f"группа с максимумом: {max_homonym_group}"
    )
    print("На каждом курсе:", homonyms_per_course)

    # Задача 3
    students_without_patronym, gender_counts = analyze_patronyms(fsuir)
    print("Студентов без отчества:", students_without_patronym)
    print("Распределение по полу:", gender_counts)

    # Задача 4
    faculty_counts, max_faculty, min_faculty = faculty_statistics(data)
    print("Количество студентов по факультетам:")
    print(faculty_counts)
    print("Максимум:", max_faculty)
    print("Минимум:", min_faculty)

    # Задача 5
    mean_students, median_students = course_statistics(data)
    print("Среднее число студентов в группах на курсах:", mean_students)
    print("Медианное число студентов в группах на курсах:", median_students)

    # Задача 6
    popular_name, name_group, faculty, course, name_ratio = most_popular_name(data)
    print(
        f"Популярное имя: {popular_name}, группа: {name_group}, "
        f"факультет: {faculty}, курс: {course}"
    )
    print(f"Доля студентов с этим именем: {name_ratio:.2f}")

    # Задача 7
    print("Уникальные имена на П:")
    print(find_students_with_name_starting_P(data))

    # Задача 8
    fac, best_gender, best_grade = highest_avg_grade_faculty(data)
    print("Факультет с высоким средним баллом третьего курса:", fac)
    print("Пол с более высоким средним баллом:", best_gender)
    print("Средний балл:", best_grade)

    # Задача 9
    print("Первые 5 студентов с последовательными номерами:")
    print(find_consecutive_students(data))
