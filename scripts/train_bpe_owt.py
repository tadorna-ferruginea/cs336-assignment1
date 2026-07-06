import pickle
import cProfile
import pstats
from cs336_basics.bpe import train_bpe


def main():
    # run the trainings with given parameters
    input_path = "data/owt_train.txt"
    vocabsize = 32000
    special_tokens = ["<|endoftext|>"]

    profiler = cProfile.Profile()

    profiler.enable()
    vocab, merges = train_bpe(input_path, vocabsize, special_tokens)
    profiler.disable()

    with open("output/owt_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("output/owt_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    with open("output/owt_profile.txt", "w") as f:
        pstats.Stats(profiler, stream=f).sort_stats("cumulative").print_stats(30)

    longest = max(vocab.values(), key=len)
    print(f"longest word in bytes: {longest!r}")
    print(f"longest word: {longest.decode('utf-8', errors='replace')}")
    print(f"longest word length in bytes: {len(longest)}")


if __name__ == "__main__":
    main()
