import pandas as pd
import re
from typing import Tuple

# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    fsuir = data[
        data["факультет"] == "факультет систем управления и робототехники"
    ].copy()

    num_students = len(fsuir)
    num_groups = fsuir["группа"].nunique()

    return num_students, num_groups, fsuir

# Задача 2
def find_homonymous_students(df: pd.DataFrame) -> Tuple[bool, int, pd.Series, str]:
    """
    Проверяет наличие однофамильцев на ФСУиР, их количество, распределение по курсам
    и определяет группу с наибольшим числом однофамильцев.
    """
    students = df.copy()
    students["фамилия"] = students["фио"].str.split().str[0]

    surname_counts = students["фамилия"].value_counts()
    repeated_surnames = surname_counts[surname_counts > 1].index
    homonyms = students[students["фамилия"].isin(repeated_surnames)]

    homonyms_per_course = homonyms.groupby("курс").size().sort_index()
    homonyms_per_group = homonyms.groupby("группа").size()

    max_homonym_group = (
        homonyms_per_group.idxmax() if not homonyms_per_group.empty else ""
    )

    return (
        not len(repeated_surnames) == 0,
        len(homonyms),
        homonyms_per_course,
        max_homonym_group,
    )

# Задача 3
def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    if not isinstance(patronym, str) or not patronym.strip():
        return "unknown"

    patronym = patronym.strip().lower()

    if patronym.endswith(("овна", "евна", "ична", "инична")):
        return "female"
    if patronym.endswith(("ович", "евич", "ич")):
        return "male"

    return "unknown"


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """
    Определяет количество студентов без отчества и распределение студентов по полу
    на основе отчества.
    """
    # Всё после фамилии и имени рассматриваем как отчество/оставшуюся часть имени.
    parts = df["фио"].fillna("").str.split(n=2, expand=True)
    if 2 in parts.columns:
        patronym = parts[2].fillna("").str.strip()
    else:
        patronym = pd.Series("", index=df.index)

    students_without_patronym = int((patronym == "").sum())

    # Студенты без отчества не включаются в распределение по полу.
    gender = patronym[patronym != ""].map(gender_identification)
    gender_counts = gender.value_counts().reindex(
        ["female", "male", "unknown"], fill_value=0
    )

    return students_without_patronym, gender_counts
    



# Задача 4
def faculty_statistics(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    faculty_counts = data.groupby("факультет").size().to_frame(
        name="количество студентов"
    )

    max_faculty_name = faculty_counts["количество студентов"].idxmax()
    min_faculty_name = faculty_counts["количество студентов"].idxmin()

    max_faculty = (
        max_faculty_name,
        int(faculty_counts.loc[max_faculty_name, "количество студентов"]),
    )
    min_faculty = (
        min_faculty_name,
        int(faculty_counts.loc[min_faculty_name, "количество студентов"]),
    )

    return faculty_counts, max_faculty, min_faculty



# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Сначала считает количество студентов на каждом факультете и курсе,
    затем вычисляет среднее и медиану по факультетам для каждого курса.
    """
    counts = data.groupby(["факультет", "курс"]).size().unstack(fill_value=0)

    mean_students = counts.mean(axis=0).sort_index()
    median_students = counts.median(axis=0).sort_index()

    return mean_students, median_students


# Задача 6
def most_popular_name(data: pd.DataFrame) -> Tuple[str, str, str, str, float]:
    """
    Определяет самое популярное имя, группу с наибольшим количеством студентов с этим именем,
    факультет, курс и долю таких студентов в общем числе.
    """
    parts = data["фио"].fillna("").str.split()
    names = parts.str[1].fillna("")
    name_counts = names[names != ""].value_counts()

    popular_name = name_counts.index[0]
    popular_count = int(name_counts.iloc[0])

    students = data.copy()
    students["имя"] = names
    students_with_name = students[students["имя"] == popular_name]

    group_counts = students_with_name.groupby(
        ["факультет", "курс", "группа"]
    ).size()

    faculty, course, group = group_counts.idxmax()
    name_ratio = round(popular_count / len(data), 2)

    return popular_name, group, faculty, course, name_ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П".
    Выводит их ФИО, факультет и курс.
    """
    students = data.copy()
    students["имя"] = students["фио"].fillna("").str.split().str[1].fillna("")

    name_counts = students.loc[students["имя"] != "", "имя"].value_counts()
    unique_names_starting_p = name_counts[
        (name_counts == 1) & name_counts.index.str.startswith("П")
    ].index

    result = students[students["имя"].isin(unique_names_starting_p)]
    return result[["фио", "факультет", "курс"]]
# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл которого выше.
    Возвращает факультет, пол и округленный средний балл.
    """
    third_course = data[data["курс"] == "3-й"].copy()
    faculty_means = third_course.groupby("факультет")["средний_балл"].mean()
    best_faculty = faculty_means.idxmax()

    students = third_course[third_course["факультет"] == best_faculty].copy()
    parts = students["фио"].fillna("").str.split(n=2, expand=True)
    patronym = parts[2].fillna("") if 2 in parts.columns else pd.Series("", index=students.index)
    students["пол"] = patronym.map(gender_identification)

    gender_means = students[students["пол"] != "unknown"].groupby("пол")[
        "средний_балл"
    ].mean()

    best_gender = gender_means.idxmax()
    best_grade = int(round(gender_means.max()))

    return best_faculty, best_gender, best_grade


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов с последовательными табельными номерами,
    обучающихся на одном курсе. Выводит ФИО, факультет, курс и группу.
    """
    students = data.sort_values("ису").copy()

    # Последовательность продолжается, если ИСУ увеличивается на 1 и курс тот же.
    consecutive = (
        students["ису"].diff().eq(1)
        & students["курс"].eq(students["курс"].shift())
    )
    students["цепочка"] = (~consecutive).cumsum()

    chain_sizes = students.groupby("цепочка").size()
    suitable_chains = chain_sizes[chain_sizes >= 5]

    if suitable_chains.empty:
        return students.iloc[0:0][["фио", "факультет", "курс", "группа"]]

    first_chain = suitable_chains.index[0]
    result = students[students["цепочка"] == first_chain].head(5)

    return result[["фио", "факультет", "курс", "группа"]]

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
