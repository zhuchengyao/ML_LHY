import numpy as np
import matplotlib.pyplot as plt


plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 10

total_sample = 1000
X = np.random.randn(total_sample)
Y = 2 * X + 1 + np.random.uniform(-0.3, 0.3, size=total_sample)

data = np.column_stack((X, Y))
# print(data[:10])

train_split = int(0.9*total_sample)
train_data, test_data = data[:train_split], data[train_split:]
print("随机选择一个0-899之间的数字作为样例" + str(np.random.choice(900)))
x, y = train_data[np.random.choice(train_split)]
print(f"x={x:.4f}, y={y:.4f}")

# 随机生成参数w和b
w = np.random.random()
b = np.random.random()


# 计算loss
y_hat = w * x + b
loss = np.square(y_hat - y)/2

dy_hat = y_hat - y
dw = x * dy_hat
db = dy_hat
print(f"loss: {loss:.4f}")
print(f"dy_hat:{dy_hat:.4f}, dw:{dw:.4f}, db:{db:.4f}")

# 用pytorch来计算梯度和自己手动计算的梯度作比较，看看是否一致
import torch
x_t, y_t = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

w_t = torch.tensor(w, dtype=torch.float, requires_grad=True)
b_t = torch.tensor(b, dtype=torch.float, requires_grad=True)

y_hat_t = w_t * x_t + b_t
y_hat_t.retain_grad()
loss_t = (y_t - y_hat_t)**2 /2
loss_t.backward()

print(f"loss: {loss_t.item():.4f}")
print(f"dy_hat:{y_hat_t.grad.item():.4f}, dw:{w_t.grad.item():.4f}, db:{b_t.grad.item():.4f}")


y_pred = np.array([w*x+b for x in test_data[:,0]])
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(test_data[:,0], test_data[:,1], alpha=0.5, c='b', marker='o')
ax.scatter(test_data[:,0], y_pred, alpha=0.5, c='g', marker='o')
plt.show()

# 看看梯度下降训练的效果
lr = 1e-2
step = 2

for i in range(step):
    for x, y in train_data:
        y_hat = w*x+b
        loss = np.square(y-y_hat)/2
        print(f"step:{i+1}, loss: {loss: .4f}")
        dy_hat = y_hat-y
        dw = x * dy_hat
        db = dy_hat

        w -= lr*dw
        b -= lr*db

y_pred = np.array([w*x+b for x in test_data[:,0]])
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(test_data[:,0], test_data[:,1], alpha=0.5, c='b', marker='o')
ax.scatter(test_data[:,0], y_pred, alpha=0.5, c='g', marker='o')
plt.show()