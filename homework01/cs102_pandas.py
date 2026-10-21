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
    #находим всех однофамильцев
    surname_counts = df.groupby("surname").size()
    homonym_surnames = surname_counts[surname_counts > 1].index
    homonyms = df[df["surname"].isin(homonym_surnames)]

    # считаем общее количество и наличие
    has_homonyms = not homonyms.empty
    total_homonyms = len(homonyms)

    # ищем дубликаты по паре ("курс", "surname")
    same_course = df[df.duplicated(["курс", "surname"], keep=False)]
    # Считаем, сколько таких людей на каждом курсе
    per_course = same_course["курс"].value_counts().sort_index()

    # ищем группу с максимумом
    group_counts = homonyms.groupby("группа").size().sort_values(ascending=False)
    max_group = group_counts.index[0]

    return has_homonyms, total_homonyms, per_course, max_group
    

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
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
     - количество студентов без отчества
     - серию с распределением студентов по полу 
    """
    without_patronym = len(df[df["patronim"] == ""])

    with_patronym = df[df["patronim"] != ""]
    genders = [gender_identification(p) for p in with_patronym["patronim"]]
    genders_df = pd.DataFrame({"пол": genders})
    counts = genders_df.groupby("пол").size()
    counts = counts[counts.index != "unknown"]
    return without_patronym, counts



# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    counts = data.groupby("факультет").size().sort_values(ascending=False)
    max_faculty = (counts.index[0], int(counts.iloc[0]))
    min_faculty = (counts.index[-1], int(counts.iloc[-1]))
    return counts, max_faculty, min_faculty


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    per_faculty = data.groupby(["курс", "факультет"]).size()
    mean = per_faculty.groupby("курс").mean().round(1)
    median = per_faculty.groupby("курс").median()
    return mean, median



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
    name_counts = data.groupby("name").size().sort_values(ascending=False)
    popular_name = name_counts.index[0]

    with_name = data[data["name"] == popular_name]
    group = with_name.groupby("группа").size().sort_values(ascending=False).index[0]
    
    row = data[data["группа"] == group].iloc[0]
    ratio = round(len(with_name) / len(data), 2)
    return popular_name, group, row["факультет"], row["курс"], ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    name_counts = data.groupby("name").size()
    once = name_counts[name_counts == 1].index
    names_p = [n for n in once if n.startswith("П")]
    return data[data["name"].isin(names_p)][["фио", "факультет", "курс"]]


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    third = data[data["курс"] == "3-й"]
    avg_by_faculty = third.groupby("факультет")["средний_балл"].mean().sort_values(ascending=False)
    faculty = avg_by_faculty.index[0]

    fac_data = third[third["факультет"] == faculty]
    genders = [gender_identification(p) for p in fac_data["patronim"]]
    table = pd.DataFrame({"пол": genders, "балл": list(fac_data["средний_балл"])})
    table = table[table["пол"] != "unknown"]

    avg_by_gender = table.groupby("пол")["балл"].mean().sort_values(ascending=False)
    return faculty, avg_by_gender.index[0], int(round(avg_by_gender.iloc[0]))


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    d = data.sort_values("ису")
    isu = list(d["ису"])
    course = list(d["курс"])

    for i in range(len(d) - 4):
        consecutive = all(isu[i + k + 1] - isu[i + k] == 1 for k in range(4))
        same_course = len(set(course[i:i + 5])) == 1
        if consecutive and same_course:
            return d.iloc[i:i + 5][["фио", "факультет", "курс", "группа", "ису"]]


if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")
    data["surname"] = [x.split()[0] for x in data["фио"]]
    data["name"] = [x.split()[1] for x in data["фио"]]
    data["patronim"] = [x.split()[2] if len(x.split()) > 2 else "" for x in data["фио"]]
    
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
