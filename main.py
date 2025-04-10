from models.ngram_model import ngram_model
from models.min_edit_dist import min_dist


def main():
    ngram_size = 5
    file = "data/big_data.txt"
    ml = ngram_model(file, ngram_size)
    text = "i havn gott "

    text = ("<s> " * (ngram_size - 1) + text + " </s>").split()
    print("text:", text)

    for i in range(ngram_size, len(text)):
        if text[i] not in ml.word_counts:
            context = text[i - ngram_size + 1 : i]
            wrong = text[i]
            print(wrong, context)
            corrections = [
                ngram[-1]
                for ngram in ml.ngram_counts
                if ngram[:-1] == tuple(context)  # Ensure context matches
            ]

            probs = [{word: min_dist(wrong, word).dist()} for word in corrections]
            sorted_probs = probs = sorted(
                [{word: min_dist(wrong, word).dist()} for word in corrections],
                key=lambda x: list(x.values())[
                    0
                ],  # Extract the distance value for sorting
            )
            print(f"Word '{wrong}' incorrect. Suggestions: {sorted_probs}")

            # Replace the word:
            if len(sorted_probs) > 0:
                text[i] = list(sorted_probs[0].keys())[0]
                print(text[i])

            print(text)


if __name__ == "__main__":
    main()
