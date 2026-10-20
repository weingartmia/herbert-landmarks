import matplotlib.pyplot as plt
import numpy as np




def show_index(dataloader,index):
    features,labels= next(iter(dataloader))
    print(f"Feature batch shape: {features.size()}")
    print(f"Labels batch shape: {labels.size()}")
    img = features[index].squeeze()
    label = labels[index]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


def test_labels(loader):
    X, y = next(iter(loader))
    print(X)
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)   
    # std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    fig, axs = plt.subplots(4, 8, figsize=(20, 10))
    for ax, img, pts in zip(axs.flat, X, y):
      
        ax.imshow((img).permute(1,2,0).clamp(0,1))
        ax.scatter(pts[:,0]*224, pts[:,1]*224, c=["r","g","b","y"], s=25)  # thumb, index, palm, middle
        # ax.scatter(pts[0].item()*224,pts[1].item()*224, c="r", s=25)
        # ax.scatter(pts[2].item()*224,pts[3].item()*224, c="g", s=25)
        # ax.set_xlabel(pts[4])
        # ax.scatter(pts[5].item()*224,pts[6].item()*224, c="y", s=25)
        ax.axis("off")
    plt.show()   



def show_highest_distance(loader):
    labels, images=0,0
    for i,(X,y) in loader:
        labels.append(y)

   
    d = np.linalg.norm(labels[:, 0] - labels[:, 1], axis=1)
    print(d)
    img=y.numpy().flatten()
    idx = np.argsort(d)[-32:].flatten()
    print(idx.shape)
    fig, ax = plt.subplots(4,8)

    for i in idx:
        print(img[i])
        plt.imshow((img[i]).permute(1,2,0).clamp(0,1))
        ax.scatter(d[0:1,0] * 224, d[0:1,1] *224, c=["r,g"])
    plt.show()

 

