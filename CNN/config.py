import torch.cuda

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Classifier().to(device)

batch_size = 64

n_epochs = 8

patience = 5

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model. parameters(), lr=0.00025, weight_decay=1e-5)
