def number_to_thai_text(num):
    # Thai numerals and place values
    thai_digits = {
        0: "ศูนย์", 1: "หนึ่ง", 2: "สอง", 3: "สาม", 4: "สี่",
        5: "ห้า", 6: "หก", 7: "เจ็ด", 8: "แปด", 9: "เก้า"
    }
    thai_places = ["", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน", "ล้าน"]

    # Handle zero case
    if num == 0:
        return thai_digits[0]

    # Convert number to string and reverse it for easier place value processing
    num_str = str(num)
    digits = [int(d) for d in num_str]
    digits.reverse()  # Reverse to process from units to highest place

    result = []
    for i, digit in enumerate(digits):
        if digit == 0:
            continue  # Skip zeros
        # Special case for tens place
        if i == 1:
            if digit == 1:
                result.append(thai_places[i])  # "สิบ" for 10-19
            elif digit == 2:
                # Use "ยี่" instead of "สอง" for 20-29
                result.append("ยี่" + thai_places[i])
            else:
                result.append(thai_digits[digit] + thai_places[i])
        # Special case for units place
        elif i == 0 and digit == 1:
            # "เอ็ด" for 11 (when tens is 1) and 21 (when tens is 2)
            if len(digits) > 1 and (digits[1] == 1 or digits[1] == 2):
                result.append("เอ็ด")
            else:
                result.append(thai_digits[digit])
        else:
            result.append(thai_digits[digit] + thai_places[i])

    # Reverse back and join
    result.reverse()
    return "".join(result)

def replace_numbers_with_thai(text):
    import re
    
    # Function to convert matched number to Thai text
    def convert_match(match):
        num = int(match.group(0))
        return number_to_thai_text(num)
    
    # Replace all numbers in the text
    return re.sub(r'\b\d+\b', convert_match, text)

# Test the functions
if __name__ == "__main__":
    # Test number_to_thai_text
    test_numbers = [1562, 305, 10, 11, 12, 20, 21, 25, 29, 1001, 0, 120, 321]
    for num in test_numbers:
        print(f"{num} -> {number_to_thai_text(num)}")

    # Test replace_numbers_with_thai
    test_text = "ฉันมีเงิน 2560 บาท"
    result = replace_numbers_with_thai(test_text)
    print(f"\nOriginal: {test_text}")
    print(f"Converted: {result}")