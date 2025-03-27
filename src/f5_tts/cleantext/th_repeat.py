from pythainlp import word_tokenize

def process_thai_repeat(text):
    
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Process the tokenized words
    result = []
    i = 0
    while i < len(words):
        if i + 1 < len(words) and words[i + 1] == "ๆ":
            # If current word is followed by ๆ, repeat the current word
            result.append(words[i])
            result.append(words[i])  # Repeat the word
            i += 2  # Skip the ๆ
        else:
            result.append(words[i])
            i += 1
    
    # Join the words back together
    return "".join(result)

# Test the function
if __name__ == "__main__":
    # Example
    test_cases = [
        "วันที่ ฉันสนุกมากๆ",
        "ดีมากๆ",
        "บ้านสวยๆ",
        "เขียนเร็วๆ",
        "วันที่ ฉันสนุกมากๆ และกินอร่อยๆ"
    ]
    
    for text in test_cases:
        result = process_thai_repeat(text)
        print(f"Original: {text} -> Converted: {result}")