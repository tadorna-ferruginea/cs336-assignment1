.PHONY: train-ts train-owt clean

# train TinyStories with memory statistics
train-ts: data/TinyStoriesV2-GPT4-train.txt
	mkdir -p output
	/usr/bin/time -l uv run python scripts/train_bpe_tinystory.py 2>&1 | tee output/train_ts.log

# train OpenWebText with memory statistics
train-owt: data/owt_train.txt
	mkdir -p output
	/usr/bin/time -l uv run python scripts/train_bpe_owt.py 2>&1 | tee output/train_owt.log

# TinyStories data download
data/TinyStoriesV2-GPT4-train.txt:
	mkdir -p data
	curl -L -o $@ "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"

data/TinyStoriesV2-GPT4-valid.txt:
	mkdir -p data
	curl -L -o $@ "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

# OpenWebText data download (gzipped, needs gunzip)
data/owt_train.txt:
	mkdir -p data
	curl -L -o $@.gz "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
	gunzip $@.gz

data/owt_valid.txt:
	mkdir -p data
	curl -L -o $@.gz "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz"
	gunzip $@.gz

# download all data
.PHONY: data
data: data/TinyStoriesV2-GPT4-train.txt data/TinyStoriesV2-GPT4-valid.txt data/owt_train.txt data/owt_valid.txt

# clean output dir
clean:
	rm -f output/*.pkl output/*.txt output/*.log
