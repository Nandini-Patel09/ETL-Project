BASE_FARE = 50
PER_KM = 10

def calculate_fare(distance):
    return BASE_FARE + distance * PER_KM


# Example
trips = [5, 10, 3]
total = 0
for i, d in enumerate(trips, start=1):
    fare = calculate_fare(d)
    total += fare
    print(f"Trip {i}: ${fare}")
print("Total Fare:", f"${total}")
