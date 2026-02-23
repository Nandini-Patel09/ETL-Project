def positive_feedback(ratings):
    if not ratings:
        return "No ratings available"
    positive = sum(1 for r in ratings if r >= 4)
    return round((positive / len(ratings)) * 100, 2)


# Example
ratings = [5, 4, 3, 5, 2, 4, 1, 5]
print("Positive Feedback:", str(positive_feedback(ratings)) + "%")
