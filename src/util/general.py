

def parse_list_from_string(string: str) -> list[str]:
    substrings= string.split(",")
    return_list= []
    for element in substrings:
        return_list.append(element.strip())
    return return_list

