text = "Query processed: Man, woman and tree.. See visualization for details."
import string
clean_text = text.lower().translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
words = set(clean_text.split())
print(f"Words: {words}")

FEMALE_WORDS = {
    "she", "her", "hers", "herself", "woman", "women", "female", "girl", "girls",
    "mother", "daughter", "sister", "wife", "aunt", "niece", "grandmother",
    "lady", "ladies", "madam", "ms", "mrs", "queen", "princess", "businesswoman"
}

intersection = words & FEMALE_WORDS
print(f"Intersection: {intersection}")
print(f"Is 'woman' in words? {'woman' in words}")
print(f"Is 'woman' in FEMALE? {'woman' in FEMALE_WORDS}")
