import matplotlib.pyplot as plt


# RRF: MRR@10 by k
rrf_results = {
    "multilingual_rrf": {
        25: 0.8293198412698413,
        60: 0.8292365079365079,
        80: 0.8292365079365079,
    },
    "bge_rrf": {
        25: 0.8694952380952381,
        60: 0.8694952380952381,
        80: 0.8694952380952381,
    },
}

plt.figure(figsize=(8, 5))

for method, results in rrf_results.items():
    plt.plot(
        results.keys(),
        results.values(),
        marker="o",
        label=method,
    )

plt.title("MRR@10 by RRF k")
plt.xlabel("k")
plt.ylabel("MRR@10")
plt.xticks([25, 60, 80])
plt.ylim(0.80, 0.89)
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("rrf_mrr10.png", dpi=300)
plt.show()


# Weighted fusion: MRR@10 by weights
weight_labels = [
    "Lexical 0.75\nSemantic 0.25",
    "Lexical 0.50\nSemantic 0.50",
    "Lexical 0.25\nSemantic 0.75",
]

multilingual_weighted = [
    0.7269738095238095,
    0.7522325396825397,
    0.7358916666666666,
]

bge_weighted = [
    0.7955059523809525,
    0.8312944444444444,
    0.8238373015873016,
]

x = list(range(len(weight_labels)))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    [i - width / 2 for i in x],
    multilingual_weighted,
    width,
    label="multilingual_weighted",
)

plt.bar(
    [i + width / 2 for i in x],
    bge_weighted,
    width,
    label="bge_weighted",
)

plt.title("MRR@10 by Weighted Fusion Weights")
plt.xlabel("Fusion weights")
plt.ylabel("MRR@10")
plt.xticks(x, weight_labels)
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("weighted_fusion_mrr10.png", dpi=300)
plt.show()


# Individual retrieval methods
methods = ["multilingual", "bge", "lexical"]
mrr_values = [
    0.8311674603174602,
    0.9267563492063492,
    0.6489496031746033,
]

plt.figure(figsize=(7, 5))

bars = plt.bar(methods, mrr_values)

plt.title("MRR@10 for Individual Retrieval Methods")
plt.xlabel("Method")
plt.ylabel("MRR@10")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.bar_label(bars, fmt="%.3f", padding=3)
plt.tight_layout()
plt.savefig("individual_methods_mrr10.png", dpi=300)
plt.show()