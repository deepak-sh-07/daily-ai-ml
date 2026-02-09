import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

 
orders = pd.read_csv("orders.csv")
#Task 1
# print(orders.groupby("order_hour").size())
# print(orders["order_id"].count())
# print(orders["delivery_time_min"].mean())
# print(orders["delivery_cost"].mean())
# selected = orders.groupby("order_hour").agg(
#     order_count = ("order_id","count"),
#     avg_delivery_time = ("delivery_time_min","mean"),
#     avg_delivery_cost = ("delivery_cost","mean")
# )
# print(selected)
# print("Busiest hour",selected["order_count"].idxmax())
# print("Slowest Delivery",selected["avg_delivery_time"].idxmax())


#Task 2

# print(orders)
# data = orders[["distance_km", "delivery_time_min", "delivery_cost"]].to_numpy()
# cost_per_km = data[:,2] / data[:,0]
# cost_per_min = data[:,2] / data[:,1]
# peak = (orders["order_hour"]>=17) & (orders["order_hour"]<=22)
# print(cost_per_km)
# print(cost_per_min)
# print("Avg cost per km (peak):", cost_per_km[peak].mean())
# print("Avg cost per km (non-peak):", cost_per_km[~peak].mean())

# print("Avg cost per min (peak):", cost_per_min[peak].mean())
# print("Avg cost per min (non-peak):", cost_per_min[~peak].mean())
# print(data)



#Task 3
# data = orders[["distance_km", "delivery_time_min", "delivery_cost"]].to_numpy()
# cost_per_km = data[:,2] / data[:,0]
# cost_per_min = data[:,2] / data[:,1]
# peak = (orders["order_hour"]>=17) & (orders["order_hour"]<=22)
# plt.figure(figsize=(12, 10))

# # 1. Distance vs Delivery Cost
# plt.subplot(2, 2, 1)
# plt.scatter(orders["distance_km"], orders["delivery_cost"])
# plt.xlabel("Distance (km)")
# plt.ylabel("Delivery Cost")
# plt.title("Distance vs Delivery Cost")

# # 2. Delivery Time vs Delivery Cost
# plt.subplot(2, 2, 2)
# plt.scatter(orders["delivery_time_min"], orders["delivery_cost"])
# plt.xlabel("Delivery Time (min)")
# plt.ylabel("Delivery Cost")
# plt.title("Delivery Time vs Delivery Cost")

# # 3. Histogram of Cost per Minute
# plt.subplot(2, 2, 3)
# plt.hist(cost_per_min, bins=10)
# plt.xlabel("Cost per minute")
# plt.ylabel("Count")
# plt.title("Distribution of Cost per Minute")

# # 4. Boxplot: Peak vs Non-peak Cost per Minute
# plt.subplot(2, 2, 4)
# plt.boxplot(
#     [cost_per_min[~peak], cost_per_min[peak]],
#     labels=["Non-peak", "Peak"]
# )
# plt.ylabel("Cost per minute")
# plt.title("Cost per Minute: Peak vs Non-peak")

# plt.tight_layout()
# plt.show()
