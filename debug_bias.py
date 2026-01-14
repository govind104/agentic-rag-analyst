from src.ethics import GenderBiasDetector

def test(text):
    print(f"Text: '{text}'", flush=True)
    result = GenderBiasDetector.analyze(text)
    print(f"  Male: {result['male_terms']}", flush=True)
    print(f"  Female: {result['female_terms']}", flush=True)
    print(f"  Bias Score: {result['bias_score']}", flush=True)
    print("-" * 20, flush=True)

test("Who is he")
test("Who is that man")
test("Man, woman and tree")
test("Query processed: Who is he. See visualization for details.")
