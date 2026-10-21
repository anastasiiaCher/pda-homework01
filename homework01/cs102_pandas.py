import pandas as pd


# 1. ФСУиР
def filter_fsuir_students(data):
    fs = data[data["факультет"] == "факультет систем управления и робототехники"].copy()
    groups = fs['группа'].nunique()
    return len(fs), groups, fs


# однофамильцы
def find_homonymous_students(df):
    d = df.copy()
    d['фамилия'] = d['фио'].str.split().str[0]
    odno = d[d['фамилия'].duplicated(keep=False)]

    # отдельно по курсам (по всему факультету число чуть другое)
    po_kursam = d[d.duplicated(['курс', 'фамилия'], keep=False)]
    po_kursam = po_kursam.groupby('курс').size().reindex(d['курс'].drop_duplicates(), fill_value=0)
    group = odno.groupby("группа").size().idxmax()
    return (len(odno) > 0, len(odno), po_kursam, group)


# 3
def gender_identification(patronym):
    if not isinstance(patronym, str):
        return 'unknown'
    x = patronym.strip().casefold()
    if x.endswith(('овна', 'евна', 'ична', 'инична')):
        return 'female'
    if x.endswith(('ович', 'евич', 'ич')):
        return 'male'
    return 'unknown'


def analyze_patronyms(df):
    third = df['фио'].str.split().str[2]
    third.name = 'patronym_candidate'
    without = third.isna().sum()

    sex = third.dropna().map(gender_identification).value_counts()
    return int(without), sex


# по факультетам
def faculty_statistics(data):
    n = data.groupby('факультет').size()
    table = n.to_frame('количество_студентов').sort_values('количество_студентов', ascending=False)
    return table, (n.idxmax(), int(n.max())), (n.idxmin(), int(n.min()))


# 5
def course_statistics(data):
    sizes = data.groupby(['курс', 'факультет']).size()
    avg = sizes.groupby(level='курс').mean().round(1)
    med = sizes.groupby(level='курс').median()
    return avg,med


# самое частое имя
def most_popular_name(data):
    names = data['фио'].str.split().str[1]
    name = names.value_counts().idxmax()
    people = data[names == name]
    gr = people.groupby("группа").size().idxmax()
    row = people[people['группа'] == gr].iloc[0]
    return name, gr, row['факультет'], row['курс'], round(len(people) / len(data), 2)


# 7
def find_students_with_name_starting_P(data):
    names = data['фио'].str.split().str[1]
    n = names.value_counts()
    ok = (names.map(n) == 1) & names.str.startswith('П')
    return data.loc[ok, ['фио', 'факультет', 'курс']].copy()


# баллы у третьего курса
def highest_avg_grade_faculty(data):
    third = data[data['курс'] == '3-й']
    fac = third.groupby('факультет')['средний_балл'].mean().idxmax()
    d = third[third['факультет'] == fac].copy()
    d['пол'] = d['фио'].str.split().str[2].map(gender_identification)
    means = d[d['пол'].isin(['female', 'male'])].groupby('пол')['средний_балл'].mean()
    sex = means.idxmax()
    return fac, sex, round(float(means[sex]))


# первое место, где подряд идут пять номеров
def find_consecutive_students(data):
    d = data.sort_values('ису', kind='stable').reset_index(drop=True)
    next_num = (d['ису'].diff() == 1) & (d['курс'] == d['курс'].shift())
    ends = next_num.rolling(4).sum() == 4
    cols = ['ису', 'фио', 'факультет', 'курс', 'группа']
    if not ends.any():
        return d.iloc[0:0][cols]
    i = ends.idxmax()
    return d.loc[i - 4:i, cols].copy()


if __name__ == "__main__":
    from pathlib import Path

    data = pd.read_csv(Path(__file__).with_name("isu_fake_data.csv"))
    
    num_students, num_groups, fsuir = filter_fsuir_students(data)
    print(f"Студентов на ФСУиР: {num_students}, Групп: {num_groups}")

    yes, total, by_course, gr = find_homonymous_students(fsuir)
    print(f"Есть однофамильцы: {yes}, Всего: {total}, Группа с максимумом: {gr}")
    print('На каждом курсе:',by_course)

    # про отчества
    students_without_patronym, gender_counts = analyze_patronyms(fsuir)
    print('Студентов без отчества:', students_without_patronym)
    print("Распределение по полу:", gender_counts)

    faculty_counts, max_faculty, min_faculty = faculty_statistics(data)
    print("Количество студентов по факультетам:")
    print(faculty_counts.to_string())
    print("Факультет с наибольшим числом студентов:",max_faculty)
    print(f"Факультет с наименьшим числом студентов: {min_faculty}")
    
    mean_students, median_students = course_statistics(data)
    print("Среднее число студентов на курсах:", mean_students)
    print("Медианное число студентов на курсах:", median_students)
    
    
    popular_name, name_group, faculty, course, name_ratio = most_popular_name(data)
    print(f"Самое популярное имя: {popular_name}, Группа: {name_group}, Факультет: {faculty}, Курс: {course}")
    print('Доля студентов с этим именем:', name_ratio)
    
    result_7 = find_students_with_name_starting_P(data)
    print("Студенты с именем, начинающимся на П и встречающимся ровно один раз:")
    print(result_7.to_string(index=False))
    
    
    fac, best_gender, best_grade = highest_avg_grade_faculty(data)
    print('Факультет с высоким средним баллом 3-го курса:',fac)
    print(f"Пол с наивысшим средним баллом: {best_gender}, Средний балл: {best_grade}")
    
    result_9 = find_consecutive_students(data)
    print("Первые 5 студентов с подряд идущими табельными номерами:")
    print(result_9.to_string(index=False))
