def search_by_disease(patients, disease):
    return [p["Name"] for p in patients if p["Disease"].lower() == disease.lower()]


patients = [
    {"Name": "Alice", "Age": 30, "Disease": "Flu"},
    {"Name": "Bob", "Age": 45, "Disease": "Diabetes"},
    {"Name": "Charlie", "Age": 35, "Disease": "Flu"}
]
print("Patients with Flu:", search_by_disease(patients, "Flu"))
