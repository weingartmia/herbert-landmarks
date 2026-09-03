import matplotlib.pyplot as plt




def show_index(dataloader,index):
    features,labels= next(iter(dataloader))
    print(f"Feature batch shape: {features.size()}")
    print(f"Labels batch shape: {labels.size()}")
    img = features[index].squeeze()
    label = labels[index]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


