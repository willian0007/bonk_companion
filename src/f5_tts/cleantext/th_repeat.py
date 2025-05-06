from pythainlp import syllable_tokenize

def remove_symbol(text):
    symbols = ",{}[]().-_?/\\|!*%$&@#^<>+-\";:~\`="
    for symbol in symbols:
        text = text.replace(symbol, '')
    return text
    
def process_thai_repeat(text):
    
    cleaned_symbols = remove_symbol(text)

    words = syllable_tokenize(cleaned_symbols)
    
    result = []
    i = 0
    while i < len(words):
        if i + 1 < len(words) and words[i + 1] == "ๆ":
            result.append(words[i])
            result.append(words[i])  
            i += 2 
        else:
            result.append(words[i])
            i += 1
    
    return "".join(result)

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
