import os
import pandas as pd
import re
from typing import Tuple


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    # Оставляем только строки, где в названии факультета есть "робототехники"
    fsuir = data[data["факультет"].str.contains("робототехники")]
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
    # Фамилия -- первое слово в ФИО
    surnames = df["фио"].str.split().str[0]
    # Однофамильцы: все, у кого фамилия встречается больше одного раза
    homonyms = df[surnames.duplicated(keep=False)]
    # Сколько однофамильцев на каждом курсе
    per_course = homonyms["курс"].value_counts().sort_index()
    # Группа, в которой однофамильцев больше всего
    top_group = homonyms["группа"].value_counts().index[0]
    return len(homonyms) > 0, len(homonyms), per_course, top_group
    
 
# Задача 3
def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    # Если отчества нет, пол определить нельзя
    if patronym == "":
        return "unknown"
    # Смотрим на окончание последнего слова (отчество может состоять из нескольких слов)
    last_word = patronym.split()[-1].lower()
    if last_word.endswith(("овна", "евна", "ична")):
        return "female"
    if last_word.endswith(("ович", "евич", "ич")):
        return "male"
    return "unknown"
 
 
def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
     - количество студентов без отчества
     - серию с распределением студентов по полу 
    """
    # Отчество -- все слова ФИО, начиная с третьего (если их нет, получится пустая строка)
    patronyms = df["фио"].str.split().str[2:].str.join(" ")
    # Сколько студентов без отчества
    without_patronym = (patronyms == "").sum()
    # Определяем пол у студентов с отчеством и считаем, сколько каких
    genders = patronyms[patronyms != ""].apply(gender_identification)
    return int(without_patronym), genders.value_counts()
 
 
# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    # Считаем студентов по факультетам (список отсортирован от большего к меньшему)
    counts = data["факультет"].value_counts()
    table = counts.reset_index()
    table.columns = ["факультет", "студентов"]
    # Первый в списке -- максимум, последний -- минимум
    return table, (counts.index[0], int(counts.iloc[0])), (counts.index[-1], int(counts.iloc[-1]))
 
 
# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    # Сколько студентов на каждом факультете на каждом курсе
    counts = data.groupby(["факультет", "курс"]).size()
    # Для каждого курса берём среднее и медиану по факультетам
    return counts.groupby("курс").mean(), counts.groupby("курс").median()
 
 
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
    # Имя -- второе слово в ФИО
    names = data["фио"].str.split().str[1]
    name_counts = names.value_counts()
    popular_name = name_counts.index[0]
    # Группа, где больше всего студентов с этим именем
    best_group = data[names == popular_name]["группа"].value_counts().index[0]
    # Факультет и курс берём из первой строки этой группы (курс записан как "2-й", берём цифру)
    row = data[data["группа"] == best_group].iloc[0]
    # Доля в процентах, округляем до двух знаков
    ratio = round(name_counts.iloc[0] / len(data) * 100, 2)
    return popular_name, best_group, row["факультет"], int(row["курс"][0]), ratio
 
 
# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    # Имя -- второе слово в ФИО
    names = data["фио"].str.split().str[1]
    name_counts = names.value_counts()
    # Имена, которые встречаются ровно один раз
    single_names = name_counts[name_counts == 1].index
    # Оставляем студентов с таким именем, если оно начинается на "П"
    result = data[names.isin(single_names) & names.str.startswith("П")]
    return result[["фио", "факультет", "курс"]]
 
 
# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    # Только третий курс (в данных он записан как "3-й"); добавляем столбец с полом (по отчеству)
    third = data[data["курс"] == "3-й"].copy()
    third["gender"] = third["фио"].str.split().str[2:].str.join(" ").apply(gender_identification)
    # Факультет с самым высоким средним баллом
    avg_by_faculty = third.groupby("факультет")["средний_балл"].mean().sort_values(ascending=False)
    faculty = avg_by_faculty.index[0]
    # Средний балл девушек и парней этого факультета
    best = third[third["факультет"] == faculty]
    girls = best[best["gender"] == "female"]["средний_балл"].mean()
    boys = best[best["gender"] == "male"]["средний_балл"].mean()
    if girls > boys:
        return faculty, "female", int(round(girls))
    return faculty, "male", int(round(boys))
 
 
# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    # Сортируем по табельному номеру
    df = data.sort_values(by="ису").reset_index(drop=True)
    # True, если номер больше предыдущего ровно на 1 и курс тот же
    step = (df["ису"].diff() == 1) & (df["курс"] == df["курс"].shift())
    # 4 True подряд = 5 студентов с подряд идущими номерами; берём позицию последнего из них
    ends = df.index[step.astype(int).rolling(4).sum() == 4]
    if len(ends) == 0:
        return df.iloc[0:0][["фио", "факультет", "курс", "группа"]]
    return df.iloc[ends[0] - 4:ends[0] + 1][["фио", "факультет", "курс", "группа", "ису"]]


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "isu_fake_data.csv")
    data = pd.read_csv(file_path)

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
