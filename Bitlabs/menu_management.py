def add_item(menu, item):
    menu.append(item)

def remove_item(menu, item):
    if item in menu:
        menu.remove(item)
    else:
        print(f"{item} not found in menu")

def check_item(menu, item):
    return f"{item} is available" if item in menu else f"{item} is not available"


menu = ["Pizza", "Burger", "Pasta", "Salad"]
add_item(menu, "Tacos")
remove_item(menu, "Salad")
print("Updated menu:", menu)
print(check_item(menu, "Pizza"))
