from datasets import load_dataset

dataset = load_dataset("andrrrei/saiga_pref_selfplay_best_sft")

dataset.save_to_disk("datasets/saiga_pref_selfplay_best_sft")
