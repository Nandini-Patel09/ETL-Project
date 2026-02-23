def calculate_average(students):
    return {name: round(sum(marks) / len(marks), 2) for name, marks in students.items()}

def find_top_performer(averages):
    return max(averages, key=averages.get)


students = {"John": [85, 78, 92], "Alice": [88, 79, 95], "Bob": [70, 75, 80]}
averages = calculate_average(students)
print("Average Marks:", averages)
print("Top Performer:", find_top_performer(averages))
