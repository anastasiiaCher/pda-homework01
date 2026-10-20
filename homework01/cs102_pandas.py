import pandas as pd
import re
from typing import Tuple


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    d=data[data["факультет"]=="факультет систем управления и робототехники"].copy()
    return len(d),d["группа"].nunique(),d


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
    s=df["surname"] if "surname" in df else df["фио"].str.split().str[0]
    d=df[s.duplicated(keep=False)]
    g=d["группа"].value_counts()
    return not d.empty,len(d),d["курс"].value_counts().sort_index(),g.idxmax() if len(g) else ""
    

# Задача 3
def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    if not isinstance(patronym,str): return "unknown"
    p=patronym.strip().lower()
    if p.endswith(("овна","евна","инична","ична")): return "female"
    if p.endswith(("ович","евич","ич")): return "male"
    return "unknown"


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
     - количество студентов без отчества
     - серию с распределением студентов по полу 
    """
    p=df["patronim"] if "patronim" in df else df["фио"].str.split().str[2].fillna("")
    missing=p.fillna("").astype(str).str.strip().eq("").sum()
    genders=p.apply(gender_identification)
    return int(missing),genders[genders.isin(["male","female"])].value_counts()


# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    c=data["факультет"].value_counts()
    return c.rename_axis("факультет").reset_index(name="студенты"),(c.idxmax(),int(c.max())),(c.idxmin(),int(c.min()))


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    c=data.groupby(["курс","факультет"]).size().reset_index(name="студенты").groupby("курс")["студенты"]
    return c.mean().round(1),c.median()


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
    n=data["name"] if "name" in data else data["фио"].str.split().str[1]
    c=n.value_counts()
    name=c.idxmax()
    d=data[n==name]
    group=d["группа"].value_counts().idxmax()
    row=d[d["группа"]==group].iloc[0]
    return name,group,row["факультет"],row["курс"],round(c.max()/len(data),2)


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    n=data["name"] if "name" in data else data["фио"].str.split().str[1]
    return data.loc[n.map(n.value_counts()).eq(1)&n.str.startswith("П").fillna(False),["фио","факультет","курс"]]


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    d=data[data["курс"]=="3-й"]
    fac=d.groupby("факультет")["средний_балл"].mean().idxmax()
    d=d[d["факультет"]==fac].copy()
    p=d["patronim"] if "patronim" in d else d["фио"].str.split().str[2].fillna("")
    d["gender"]=p.apply(gender_identification)
    means=d[d["gender"]!="unknown"].groupby("gender")["средний_балл"].mean()
    return fac,means.idxmax(),int(round(means.max()))


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    d=data.sort_values("ису").reset_index(drop=True)
    hits=d["ису"].diff().eq(1).rolling(4).sum().eq(4)
    cols=["ису","фио","факультет","курс","группа"]
    if not hits.any(): return d.iloc[0:0][cols]
    last=hits[hits].index[0]
    return d.iloc[last-4:last+1][cols]


if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")
    data["surname"]=data["фио"].str.split().str[0]
    data["name"]=data["фио"].str.split().str[1]
    data["patronim"]=data["фио"].str.split().str[2].fillna("")
    
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
