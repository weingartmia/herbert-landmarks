import matplotlib.pyplot as plt




def show_index(inx, dataloader):
    features,labels= next(iter(dataloader))
    print(f"Feature batch shape: {features.size()}")
    print(f"Labels batch shape: {labels.size()}")
    img = features[0].squeeze()
    label = labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


