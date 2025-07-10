import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE # t-SNE অ্যালগরিদম
from sklearn.datasets import load_digits # MNIST-এর মতো ডেটা সেট



def generate_data():
    # হাতে লেখা সংখ্যার ডেটা সেট লোড করো
    digits = load_digits()
    X = digits.data # ডেটা (64D)
    y = digits.target # ডেটার আসল লেবেল (0-9)

    # ডেটার আকার দেখে নাও
    print(f"মূল ডেটার আকার (Shape): {X.shape}") # আউটপুট: (1797, 64)
    print(f"লেবেলের আকার (Shape): {y.shape}")   # আউটপুট: (1797,)

    return X, y

def run_app():
    print("Welcome to the t-SNE application!")
    X, y = generate_data()
    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
    print(X)
    print("t-SNE রূপান্তর শুরু হচ্ছে... এতে কিছুটা সময় লাগতে পারে।")
    X_tsne = tsne.fit_transform(X)
    print(X_tsne)
    plt.figure(figsize=(12, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='Paired', s=50, alpha=0.8) # রূপান্তরিত ডেটা প্লট করো
    plt.colorbar(label='আসল সংখ্যা (0-9)') # কালার বার যোগ করো
    plt.title("t-SNE এর পর 2D ডেটা (হাতে লেখা সংখ্যা)")
    plt.xlabel("t-SNE কম্পোনেন্ট ১")
    plt.ylabel("t-SNE কম্পোনেন্ট ২")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    print(f"t-SNE এর পর ডেটার আকার (Shape): {X_tsne.shape}") # আউটপুট: (1797, 2)

if __name__ == "__main__":
    run_app()