def jaccard_similarity(str1, str2):
    set1, set2 = set(str1.lower().split()), set(str2.lower().split())
    return len(set1 & set2) / len(set1 | set2)
