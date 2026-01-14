from src.ethics import GenderBiasDetector

def test(text):
    print(f"\n--- Testing: '{text}' ---", flush=True)
    GenderBiasDetector.analyze(text)

test("Query processed: Man, woman and tree.. See visualization for details.")
test("Query processed: That man is interesting.. See visualization for details.")
