def available_seats(total_seats, booked_seats):
    return [seat for seat in range(1, total_seats+1) if seat not in booked_seats]

def book_seat(booked, seat):
    if seat in booked:
        return "Seat already booked"
    booked.append(seat)
    return "Booking successful"

def cancel_seat(booked, seat):
    if seat in booked:
        booked.remove(seat)
        return "Cancellation successful"
    return "Seat not booked"


total = 10
booked = [2, 5, 7]
print("Available seats:", available_seats(total, booked))
print(book_seat(booked, 3))
print(cancel_seat(booked, 5))
print("Available seats:", available_seats(total, booked))
