
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons # বৃত্তাকার ডেটা তৈরি করার জন্য



# অর্ধ-চন্দ্রাকার ডেটা তৈরি করো
# n_samples: মোট ডেটা পয়েন্টের সংখ্যা
# noise: ডেটা পয়েন্টগুলোতে কতটা এলোমেলোতা থাকবে (আউটলায়ারের মতো)
X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)

# ডেটা কেমন দেখতে, একটু দেখে নাও
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
plt.title("তৈরি করা ডামি ডেটা (অনিয়মিত আকার)")
plt.xlabel("ফিচার ১")
plt.ylabel("ফিচার ২")
plt.show()



# DBSCAN মডেল তৈরি করো
# eps: একটি পয়েন্টের চারপাশে কত দূরত্বের মধ্যে অন্য পয়েন্ট থাকলে প্রতিবেশী ধরা হবে
# min_samples: একটি পয়েন্টকে কোর পয়েন্ট হতে হলে তার eps দূরত্বের মধ্যে ন্যূনতম কতগুলো প্রতিবেশী থাকতে হবে
dbscan = DBSCAN(eps=0.3, min_samples=5)

# মডেলকে ডেটার ওপর ফিট করো এবং প্রতিটি ডেটা পয়েন্টের ক্লাস্টার লেবেল বের করো
labels = dbscan.fit_predict(X)



plt.figure(figsize=(10, 8))

# ক্লাস্টার করা ডেটা পয়েন্টগুলো আঁকো
# নয়েজ পয়েন্টগুলো কালো রঙে দেখানো হবে (লেবেল -1)
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1: # নয়েজ পয়েন্ট
        # Use black for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=8, label=f'ক্লাস্টার {k}' if k != -1 else 'নয়েজ')

plt.title('DBSCAN ক্লাস্টারিং ফলাফল')
plt.xlabel('ফিচার ১')
plt.ylabel('ফিচার ২')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()